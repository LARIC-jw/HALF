
"""
-------------------------------------------------------------------------------
Description
-------------------------------------------------------------------------------
Main executing script to filter LakeSP time series after downloading individual
lake csv files from Hydrocron.

This script was modified from Customized-LakeSP-filter-vxx.py to allow joblib 
parallel per-lake: 
    1. Parallelize *within each basin* over lakes using joblib. 
    2. Return one output CSV per basin.
    3. No need to perform gauge validation as in Customized-LakeSP-filter-vxx.py

Initiated: 11/11/2024
Last update: 09/11/2025 (adds joblib.Parallel for per-lake parallelism; retains SLURM array across basins)
Contact: jidaw@illinois.edu

-------------------------------------------------------------------------------
Intput: per-lake time series csv files from Hydrocron
        (saved under `work_dir_Hydrocron` folder)
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Final output: per-basin filtered LakeSP time series
-------------------------------------------------------------------------------
For each basin ID in `unique_basin_ids`, the script writes a single CSV:

    Hydrcron_prior_<basin>.csv
    (saved under `work_dir_Hydrocron_filtered` folder)

Each row in this per-basin file corresponds to one LakeSP observation
(i.e., one prior lake at one SWOT acquisition time) that has a valid
time and WSE after basic screening.

The output schema is:

    - Original LakeSP / Hydrocron fields
      All columns read from the original per-lake Hydrocron CSV are
      preserved (e.g., lake_id, time, time_str, wse, wse_u, wse_std,
      xtrk_dist, quality_f, ice_clim_f, xovr_cal_q, crid, etc.).

    - Additional columns created by this script:
      * index_col
          Integer index (0..N-1) assigned after initial read for each
          lake file; used solely as a stable ID to track observations
          through filtering and to merge results back to the original
          DataFrame.

      * datetime
          Datetime version of `time`, converted from SWOT HR epoch
          (seconds since 2000-01-01 00:00:00 UTC) to a timezone-naive
          pandas datetime64[ns]. This is the primary time coordinate
          for analysis and plotting.

      * filter_flag
          Indicator of whether an observation is retained by the final
          filtering scenario for that lake:
              1  = retained (non-outlier; part of the final filtered
                   series for this lake)
              0  = rejected as an outlier (or not included in the final
                   filtered series)
          Implementation detail:
              - After the filtering attempt for a lake is completed, the
                surviving observations are those whose `index_col`
                appears in the final `df_filtered` dataframe. All other 
                rows in the lake’s original df are labeled filter_flag = 0.

      * wse_adjusted
          Filtered / adjusted WSE (in m) for retained observations:
              - For a given lake, `wse_adjusted` is copied from the
                final `df_filtered`, after applying the chosen filter
                scenario AND the cycle-adjustment step
                `sp_cycle_adjustment`.
              - If no cycle adjustment is needed or applied, then
                `wse_adjusted` simply equals `wse` for survivors.
              - For rows flagged as outliers (filter_flag = 0), or for
                any observation not present in `df_filtered`,
                `wse_adjusted` is left as NaN.
          Recommended usage:
              - Use `wse_adjusted` as the main filtered time series (with filter_flag == 1).
              - Use original `wse` only if you explicitly want to inspect
                raw LakeSP values.

      * n_while
      * n_while_r2
          Less important: supplementary information to track filtering process: 
          Iteration counts summarizing the internal convergence behavior
          of the filtering algorithm for each lake:
              - For strict / lenient attempts:
                    n_while    = number of main iterative passes of the
                                  customized filter
                    n_while_r2 = number of additional passes
                                  (if r2_filter = 'yes'; otherwise
                                  typically -2 or 0 by convention)
              - For Tukey/IQR attempt:
                    `apply_baseline_tukey_filter` returns a single
                    iteration count; the script stores it as:
                        [n_while, -2]
              - For '4_none' (no successful filter), both are set to 0.
          These are mainly diagnostic; they can be used to inspect how
          “hard” the filter had to work or to flag problematic lakes.

      * filter_attempt
          Categorical ranking label describing which filter scenario ended up 
          being used for that lake:
              - 1_strict: Strict criteria applied to the customized filter,
                          so that retained time series must be sufficiently long (e.g., 1 year)
                          and must not contain major gaps (e.g., 3 months or 1 season)
              - 2_lenient: Less strict criteria applied to the customized filter,
                          so there is no constraint on temporal span or gap, but retained 
                          time series must contain >=5 observations
              - 3_tukey (not recommended):   If attempt 1 or 2 still failed, 
                          apply only a Tukey (IQR) method on the baseline time series, and 
                          the retained time series must contain at least one observation.
              - 4_none:   If none of the first three attempts was successful,
                          this lake is left without any filtering. 
          Special case:
              - Lakes with 'no data' never enter the per-basin CSV at all;
                they are counted separately in basin-level stats.

-------------------------------------------------------------------------------
Basin-level coverage and “no data” lakes
-------------------------------------------------------------------------------
Within each basin, lakes fall into two groups:

    1) Lakes with usable SWOT WSE data:
       - These appear in the per-basin CSV, with:
            filter_attempt ∈ {'1_strict','2_lenient','3_tukey','4_none'}
       - Summary counts are reported at the end of each basin’s run:
            % strict, % lenient, % tukey, % no filter
            (over `lake_count_in_basin`)

    2) Lakes with no usable data ('no data'):
       - Either the lake’s Hydrocron file is missing or it contains no
         valid WSE/time pairs after initial screening.
       - These lakes are NOT included as rows in the per-basin CSV.
       - They are counted via `lake_no_data_in_basin` and reported as:
            % no data
         where the denominator is
            lake_count_in_basin + lake_no_data_in_basin.

Together, these diagnostics allow one to:
   - Focus analyses on `wse_adjusted` with filter_flag == 1;
   - Assess which lakes required relaxed criteria or Tukey fallback;
   - Quantify data availability and filtering success at the basin scale.
-------------------------------------------------------------------------------
"""


import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings (use with caution).
import os, datetime
import pandas as pd
import numpy as np

# --- joblib imports and HPC hygiene -------------------------------------
from joblib import Parallel, delayed

# Cap threads inside workers to avoid oversubscription
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

def _auto_n_jobs(default=1):
    """Derive worker count from env (N_JOBS or SLURM_CPUS_PER_TASK)."""
    v = os.environ.get("N_JOBS") or os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        n = int(v)
        return n if n > 0 else default
    except Exception:
        return default

N_JOBS = _auto_n_jobs(1)
JOBLIB_PREFER = os.environ.get("JOBLIB_PREFER", "processes")  # "processes" or "threads"
# -----------------------------------------------------------------------------

"""
Import customized functions:
        calibrate_heuristic_thresholds: Calibrate heuristic thresholds (max wse_std, max wse_u, and min xtrk_dist) before SP filtering.
        apply_customized_filter:       Apply heuristic thresholds to filter the SP time series.
        apply_baseline_tukey_filter:   Apply a simple baseline turkey filter to infer WSE variability (not suitable for phenology)
        sp_cycle_adjustment:           Reduce intra-cycle WSE inconsistencies in the SP time series caused by multiple orbit passes.
"""
from customized_functions import calibrate_heuristic_thresholds, apply_customized_filter, \
                                 apply_baseline_tukey_filter, sp_cycle_adjustment

# -------------------- Parameters ----------------------------------------------
start_time = "2023-07-11T00:00:00Z"  # 2023-07-21 is the start of the SWOT nominal orbit.
end_time   = "2025-07-08T00:00:00Z"
#The Cal/Val phase occurred from 30 March 2023, to 10 July 2023. The science phase began on 21 July 2023,
#following the transition of the spacecraft to its science orbit. https://doi.org/10.1029/2025GL114936

apply_low_pass_filter = 'yes'  # 'yes' strongly recommended
# The following parameters only matter if apply_low_pass_filter = 'yes'
evaluating_at_full_data = 'no' #'no' recommended
r2_filter = 'yes'              #'yes' recommended
filter_type = 'savgol' #lowess, wavelet, savgol, kalman, spline, median, hampel.
z_score_thresholds = [2.576, 3.5] #2.576(99% for two tails), 2.807(99.5%), 2.967(99.7%), 3.291(99.9%), 3.5(99.95%)
maximum_residual_spreads = [0.07, 0.05] #0.08 0.06 Amazon (622): [0.05, 0.03]
show_filtering_evolution = 'no' #for visualization only; caution: 'yes' may load many figures at the end of the script execution. 



# -------------------- Paths ---------------------------------------------------
# Retrieve absolute path of the script
script_path = os.path.abspath(__file__)
# Get the directory containing the script
work_dir = os.path.dirname(script_path)
print("Working directory:", work_dir)

# INPUT folder containing all Hydrocron lake CSV files: one csv file (Hydrocron downlaod) is a LakeSP time series per prior lake
work_dir_Hydrocron = os.path.join(work_dir, "LakeSP_Hydrocron_download_07082025")

# OUTPUT folder: where we want to save the stacked basin CSVs after filtering
work_dir_Hydrocron_filtered = os.path.join(work_dir, "LakeSP_Hydrocron_download_07082025_filtered")
# Ensure output folder exists
os.makedirs(work_dir_Hydrocron_filtered, exist_ok=True)

print("N_JOBS (joblib workers):", N_JOBS, "| prefer:", JOBLIB_PREFER)

print("----- Module Started -----")
print(datetime.datetime.now())

# Read lake IDs strictly based on the prior lake IDs in PLD v106 (used for LakeSP PIC2 up to 07/08/2025, from Roger).
all_lake_ids = np.load(os.path.join(work_dir, "all_lake_id_array_v106LakeSPvPIC2.npy"), allow_pickle=True)
all_lake_ids = all_lake_ids.tolist()  # Integer
unique_basin_ids = sorted({int(str(lake_id)[:3]) for lake_id in all_lake_ids})  # 272 basins; ids are integers
unique_basin_ids = [624]

# If running under a SLURM array, pick one basin by index
task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
if task_id is not None:
    task_idx = int(task_id)
    if 0 <= task_idx < len(unique_basin_ids):
        unique_basin_ids = [unique_basin_ids[task_idx]]
    else:
        raise IndexError(f"SLURM_ARRAY_TASK_ID {task_idx} out of range (n_basins={len(unique_basin_ids)})")

# For Python, the epoch time starts at 00:00:00 UTC on 1 January 1970.
# But for SWOT HR product, time of measurement is in seconds in the UTC time scale since 1 Jan 2000 00:00:00 UTC.
#time_delta = (datetime.datetime(2000, 1, 1, 0, 0) - datetime.datetime(1970, 1, 1, 0, 0)).total_seconds()

# Define fill values depending on variable type.
fill_text  = 'no_data'
fill_int4  = -999
fill_int9  = -99999999
fill_float = -999999999999

# -------------------- Per-lake worker function -------------------------------
def _process_one_lake(this_lake_id: int, this_basin: int):
    """
    Run the existing filtering steps for a single lake and return:
      - df (pandas.DataFrame) with filter outputs for this lake (original schema + wse_adjusted + flags)
      - filter scenario (one of 'strict','lenient','baseline tukey','no filter','no data')
    Returns (None, 'no data') if the source file doesn't exist or no usable data.
    """
    lake_csv = os.path.join(work_dir_Hydrocron, f"Hydrocron_prior_{this_lake_id}.csv")
    if not os.path.exists(lake_csv):
        return None, 'no data'
    try:
        # Read this lake csv file (from Hydrocron)
        df = pd.read_csv(lake_csv, sep=';')

        # Apply the filter chain below
        
        # _________Pre-process the LakeSP data________
        # Add index_col to preserve the original df ordering/reproducibility and later label outliers. 
        df = df.copy() # Make df an independent DataFrame and eliminate the SettingWithCopyWarning
        df['index_col'] = range(len(df))    
        
        # Drop invalid records for simplicity: time and WSE must both exist.
        df = df.loc[(df.time != fill_float) & (df.wse != fill_float)]   
        
        # Mask out invalid values for filtering metrics
        df.wse_u    = df.wse_u.mask(df.wse_u  == fill_float, np.nan)
        df.wse_std  = df.wse_std.mask(df.wse_std == fill_float, np.nan)
        df.xtrk_dist= df.xtrk_dist.mask(df.xtrk_dist == fill_float, np.nan)
        
        # Intialize an outlier label: # 1 = good; 0 = outlier
        df['filter_flag'] = 1          
        
        # Ensure chronological order
        df = df.sort_values('time', kind='mergesort')  # Sort df by time to be safe (mergesort keeps relative order for ties). 
        # - "time" is measurement in seconds in the UTC time scale since 1 Jan 2000 00:00:00 UTC.
        # - "time_str" is a string giving UTC time, in YYYY-MM-DDThh:mm:ssZ, where the Z suffix indicates UTC time.
        # Caution: "time" can has higher precision, and is therefore more precise than time_str.
        #          This leads to sometimes time_str values being duplicate. So always use "time" to sort the data.        
        
        # Convert 'time' to yyyy-dd-mm hh:mm:ss (datetime64[ns], timezone-naive for simplicity). All SP time is based on UTC consistently.    
        df['datetime'] = pd.to_datetime(df['time'], unit='s', origin=pd.Timestamp('2000-01-01 00:00:00')) # (drop utc=True, tz-aware). 
        #Note: this result will be equivalent to time_str in LakeSP, but more precise (e.g., nanosecond precision) that time_str.         
        #      Deprecate pd.to_datetime(df['time_str'], format='%Y-%m-%dT%H:%M:%SZ'), which is less precise
        
        # Initialize df_eval by dropping duplicate timestamps, if any
        df_eval = df.drop_duplicates(subset='time', keep='first').copy()
        # Note: df_eval.empty, i.e., no valid SWOT data available for this lake,
        #       is possible but will be handled by the following scripts. 
        #       When df_eval is empty, the processed df will also be empty. 
        # _________Pre-process completed________
        
        
        """    
        Calibrate heuristic quality thresholds based on long-term lake statistics
        
        Logic: 
            - 1. Rather than directly using the LakeSP summary quality flags for filtering, we leverage these flags 
                 to identify high-quality observations. 
            - 2. From these selected observations, we calibrate heuristic maximum thresholds for the following key 
                 metrics (determined by random forest test):
                • wse_std_threshold: Represents the minimum acceptable surface water level consistency across (wse_std) the lake.
                • wse_u_threshold: Represents the maximum acceptable uncertainty from data processing (wse_u).
                • xtrk_dist_threshold: Represents the minimum acceptable absolute distance to the central track (|xtrk_dist|).  
            - 3. Then, these thresholds are used to retrieve a heuristic baseline that consists of the observations 
                 satifying the thresholds. 
        Notes:
            - dark_frac is not used here for filtering WSE, as it is not directly involved in WSE computation.
            - The bitwise quality flag is not used for now because it is only available for PIC2 and PID0 data 
              (since October 2024).
            - However, we utilize the updated quality_f classification (values 0–3) available in PIC2 and PID0 to define 
              good observations.
            - The heuristic thresholds may need to be independent of both data version and orbit pass. As data quality improves 
                over time, thresholds calibrated for earlier data versions may no longer be suitable for newer versions. 
                Similarly, for lakes that span multiple passes, varying factors such as surrounding terrain can affect the 
                measurements differently across passes, making a single threshold inadequate. 
                To accommodate these variations, we have designed the function "calibrate_heuristic_thresholds" with built-in 
                flexibility. 
                Please refer to the function script for further details.
        """    
        # Define conservative_SQL as the criteria of high-quality LakeSP subset, from which thresholds are calibrated.
        # Selection criteria are broadly aligned with CNES baseline standards (Claire Pottier and Roger Fjørtoft, 
        # on behalf of the HR Cal/Val Team, SWOT ST Meeting 2024).
        # quality_f:
        #     - Prior to PIC2, observations flagged as 1 (bad) are excluded.
        #     - PIC2 and PID0: observations flagged as 2 (degraded) or 3 (bad) are excluded.
        #     - Caustion: Using LakeSP = 0 prior to PIC2 may over-reject observations.
        # xovr_cal_q:
        #     - Only good observations (0) are retained. 
        # ice_clim_f:
        #     - Observations with full ice cover (2) are considered ice-covered.
        #     - Observations with partial ice cover (1) are retained in ice-free to avoid under-detection.    
        # Throwback: In version 7 (superseded), the subset of 'xovr_cal_q < 1 & quality_f < 2 & ice_clim_f < 2' (ice-free only)
        #            was used to calibrate the thresholds. 
        #            This is lacking sufficient flexibility for customization, so in this updated version, we allow the user to 
        #            define its own rule, i.e., using the thresholds calibrated during either ice-free or ice-covered period (or both). 
        # Ice coverage conditions will be considered sophisticately in function "calibrate_heuristic_thresholds". 
        # Use: 
        # (1) xovr_cal_q < 1, and 
        # (2) either
        #     quality_f < 1 (for any crid), or
        #     quality_f == 1 and crid is "PIC2" or "PID0".  
        
        # Conservative SQL for threshold calibration
        conservative_SQL = (
            '(xovr_cal_q < 1) & ('
            '  (quality_f < 1) '
            '  | ( (quality_f == 1) & ((crid == "PIC2") | (crid == "PID0")) )'
            ')'
        )
        
        # Calibrate heuristic thresholds, which can be specific to pass, version, and ice-condition. 
        #  - wse_std (in m): Lake surface flatness, assuming errors lead to large wse_std
        #  - wse_u (in m): Algorithm processing uncertainty, assuming errors lead to large wse_u
        #  - xtrk_dist (in m, abs. 0-75000 m): Distance of lake polygon centroid from the nadir track, assuming errors lead to small |xtrk_dist|
           
        # Deprecated:    
        #    A lake split by the [-10 to 10] km central track may also have an |xtrk_dist value| < 10 km, 
        #    Yet its WSE value can be useful, as shown in many cases, such as 4340980733 (TGD) and 4530143033.
        #    So, an observation whose |xtrk_dist| < 10 km may not mean poor WSE, 
        #    and we cannot just exclude observations whose |xtrk_dist| < 10 km, either.
        #    In other wrods, while wse_std represents water consistency and wse_u represents algorithm uncertainty, both indicating quality, 
        #    small |xtrk_dist| may not always represent bad quality, so using it is deprecated for now. 
        
        # Calibrate heuristic thresholds
        # Caution: Applying pass and version groupings for wse_std or wse_u may lead to avoid over-rejection.
        #          Pass groupiong is theoretically needed for xtrk_dist as lake position varies in different passes. 
        df_heuristic_thresholds = calibrate_heuristic_thresholds(
            df_eval, conservative_SQL,
            by_crid_scenario=[False, False, False],  # [wse_std, wse_u, xtrk_dist]
            by_pass_id=[False, False, True],
            by_ice=[True, True, True]
        )
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(df_heuristic_thresholds)
        # Note: output df_heuristic_thresholds contains: 
        #       ['crid_scenario', 'pass_id', 'ice_condition', 'wse_std_threshold', 'wse_u_threshold', 'xtrk_distant_threshold'] 
        #       - crid_scenario has two scenarios: "PIC2_or_PID0" and "early_versions" (e.g., PIC0, PGC0);
        #       - ice_condition has three scenarios: "ice-free", "ice-covered", and "both". 
        #       See function calibrate_heuristic_thresholds for details. 
        # Prefer using by_crid_scenario = False for now, e.g., 7720028943, PIC2 data is too limited at this moment (2 pts). 


        """    
        Apply heuristic thresholds to the customized filter
           
        Logic:
            > The calibrated heuristic thresholds are first applied to the initial SP time series to retrieve a baseline subset.
            > A low-pass filter (e.g., LOWESS or Savitzky–Golay) is then fitted to the heuristic baseline, but evaluated 
                        against the initial SP time series to identify noises.
            > This procedure is repeated iteratively until convergence criteria are satisfied.
            > An optional round-2 low-pass filter is also available if r2_filter is set to "yes"
            Flexible parameter settings are supported throughout the process.   
        """    
        # Reasoning for threshold bounds:     
        # - max bound for wse_std_thresholdx = 3 or 5. 3 seems more conservative and accurate, but 5 m may allow more observations in.
        # - min bound for wse_std_threshold = 0
        # - max bound for wse_u_threshold = 0.5. 0.5 m is probably a bit too conservative. Probably needs optimization using validation data.
        # - min bound for wse_u_threshold = 0 or 0.1 The value of 0.1 m overall consistent with the science requirement. 
        # Note: Not capping mininum wse_u_threshold may lead to over-rejection. 
        #       The upper bounds may lack strong theoretical justification and may require future revision.!!!
        #       For reference, a wse_std_threshold exceeding 3-5 m is unlikely:
        #          - Lake Tefé, a ria lake with strong water-surface gradient during the dry season, shows a maximum 4–5 m WSE range.
        #          - Backwater superelevations for Lake Selingué and Shardara Reservoir are typically <3 m.
        
        # - max bond for xtrk_dist_threshold = 75000, which is the valid max.  
        # - min bond for xtrk_dist_threshold = 0, which is the valid min.  
        # Note: Do not enforce xtrk_dist for now (see above).    
        
        # More filtering rules:
        # Based on experimentation, it seems by_crid_scenario = True (with smaller thresholds for PIC2/PID0) does lead to some degree
        #    of over-rejection, but they are also often reasonable. 
        # Not setting wse_u_threshold min bound seems to improve noise: e.g., good observations in PIC2 often have wse_u < 0.1. 
        #    Overall, ice leads to larger wse_u due to increased interferometric noises.
        #    So, use 0 min bound for wse_u_threshold, but set up a higher min bound for freeze-up condition only (wse_u_ice_min).
        # wse_std_ice_min = 3 # Elevate min wse_std threshold for ice/freeze-up condition to allow for data
        # wse_u_ice_min = 0.1 # Elevate min wse_u threshold for ice/freeze-up conditions to allow for data. 0.1 m is also set to 
        #    include valid extremes. 
        
        # Rules (xtrk_dist deprecated)
        # Per-metric [wse_std, wse_u, xtrk_dist], with four unique rules 'ice-free','ice-covered','both, and 'not apply'
        rules_for_ice_free_data    = ['ice-free', 'ice-free', 'not apply']
        rules_for_ice_covered_data = ['ice-free', 'ice-free', 'not apply']

        # Attempt 1: strict
        filter_attempt = '1_strict'
        df_filtered, n_while_filtered, filter_status = apply_customized_filter(
            df_eval, df_heuristic_thresholds,
            # Bounds for applied thresholds
            wse_std_threshold_bounds=[0, 3],
            wse_u_threshold_bounds=[0, 0.5],
            xtrk_dist_threshold_bounds=[0, 75000],  # deprecated
            # Ice overrides
            wse_std_ice_min=3, #use 0 if not applying the override
            wse_u_ice_min=0.1, #use 0 if not applying the override
            # Time continuity
            allow_major_gap = 'no', # 'yes/no', to indicate if gap in the filtered time series is allowed.
            max_temporal_gap=90, #Maximum temporal gap (days) for filtering
            min_temporal_range=365, # Minimum tmeporal range (days) for filtering            
            # Per-metric rules
            rules_for_ice_free_data=rules_for_ice_free_data,
            rules_for_ice_covered_data=rules_for_ice_covered_data,
            # Aux
            gauge_df=None, #None if no gauge data is available. 
            plot_period=[start_time, end_time], #Defining start and end time for plotting.
            # Filtering options
            apply_low_pass_filter = apply_low_pass_filter, 
            evaluating_at_full_data = evaluating_at_full_data,
            r2_filter = r2_filter,
            filter_type = filter_type, 
            z_score_thresholds = z_score_thresholds, 
            maximum_residual_spreads = maximum_residual_spreads,
            show_filtering_evolution = show_filtering_evolution
        ) #filter_status = 'heuristic baseline' is included by 'strict' attempt: n_while_filtered = [-2,-2]
        
        # In case of no valid SWOT data for this lake (df_eval/df is empty)
        if filter_status == 'no data': #n_while_r2 is always -9 when n_while is -9. 
            filter_attempt = 'no data' # Overwrite 'strict' by 'no data'
        
        # Attempt 2: lenient
        if filter_status == 'fail': 
            filter_attempt = '2_lenient'
            df_filtered, n_while_filtered, filter_status = apply_customized_filter(
                df_eval, df_heuristic_thresholds,
                wse_std_threshold_bounds=[0.5, 5],
                wse_u_threshold_bounds=[0.1, 0.5],
                xtrk_dist_threshold_bounds=[0, 75000],
                wse_std_ice_min=5,
                wse_u_ice_min=0.1,
                allow_major_gap = 'yes', # 'yes/no', to indicate if gap in the filtered time series is allowed.
                max_temporal_gap=90,
                min_temporal_range=365,                
                rules_for_ice_free_data=rules_for_ice_free_data,
                rules_for_ice_covered_data=rules_for_ice_covered_data,
                gauge_df=None,
                plot_period=[start_time, end_time],
                apply_low_pass_filter = apply_low_pass_filter, 
                evaluating_at_full_data = evaluating_at_full_data,
                r2_filter = r2_filter,
                filter_type = filter_type, 
                z_score_thresholds = z_score_thresholds, 
                maximum_residual_spreads = maximum_residual_spreads,
                show_filtering_evolution = show_filtering_evolution
            )   
        
        # Attempt 3: baseline Tukey (IQR)
        if filter_status == 'fail': 
            # This attempt is only to infer WSE variability uncertainty, not for characterize phenology
            # So, the previous customized filter does not apply, and instead we use a simple baseline_tukey method
            filter_attempt = '3_tukey'
            # Define the baseline condition as a boolean mask
            baseline_SQL = '(quality_f == 0) & (xovr_cal_q == 0) & (ice_clim_f < 2)'
            # Remove remaining isolated extreme outliers using Tukey method (IQR method) 
            df_filtered, n_while_filtered, filter_status = apply_baseline_tukey_filter(
                df_eval, baseline_SQL,
                multiplier=3, lower_q=0.25, upper_q=0.75, iteration_n=5
            )
            # n_while_filtered returned from apply_baseline_tukey_filter is a number. 
            # Here we update n_while_filtered to be consistent with the size of other attempts
            n_while_filtered = [n_while_filtered, -2] 
        
        # No filter works for this lake: df_filtered remains empty
        if filter_status == 'fail': 
            filter_attempt = '4_none' 
            n_while_filtered = [0, 0]
               
        n_while = n_while_filtered[0]
        n_while_r2 = n_while_filtered[1]
            
        
        
        """
        Cycle adjustment: To reduce intra-cycle WSE inconsistencies caused by multiple orbit passes
        
        Logic: For large lakes spanning multiple SWOT orbit passes, WSE values within the same orbit cycle may show substantial 
               inconsistencies (e.g., zig-zag patterns) across different passes. The following three options are provided to 
               mitigate this issue:
                 - Option 1: Compute a cycle-averaged WSE time series.
                             Averaging all WSE values within each cycle can help eliminate intra-cycle inconsistencies.
                 - Option 2: Retain only observations from the pass that captures the largest observed lake area (area_total).
                             The representative pass is identified based on the highest median area_total across the time series.
                             Note: Both Option 1 and Option 2 yield one WSE value per cycle.
                 - Option 3 (recommended): Adjust each WSE value by removing its pass-specific bias relative to the overall WSE 
                             average across the time series. This approach preserves the original number of observations and has 
                             been shown to produce more reliable results.
                  Note that option 2 and option 3 will not run if intra-cycle WSE inconsistency is insignificant. 
        """
        # Cycle adjustment (Option 3 recommended)
        _, _, df_filtered = sp_cycle_adjustment(df_filtered)
        
        
        """
        Label filtered result (survivals) back to the original df 
        """  
        # Label survivals (non-outlier LakeSP observations) back to the original df through index_col 
        #     assign df.filter_flag to be 0 when df.index_col goes beyond df_filtered.index_col (i.e., outliers)
        #     filter_flag: 1 means good; 0 means outlier
        df.loc[~df['index_col'].isin(df_filtered['index_col']), 'filter_flag'] = 0

        # Merge 'wse_adjusted' from df_filtered into df by index_col
        df = df.merge(
            df_filtered[['index_col', 'wse_adjusted']],
            on='index_col', how='left'
        )
        # Note:
        #    wse_adjusted is only valid for filtered results (good observations)
        #    If no wse_adjusted was assigned (i.e., outliers, not in df_filtered), the value of df.wse_adjusted is left nan. 
        #    wse_adjusted will equal wse if no cycle adjustment is needed. 
        #    In summary, it's safe to just use wse_adjusted for representing filtered results.

        # Assign filter_attempt and stats
        df[['n_while', 'n_while_r2', 'filter_attempt']] = [n_while, n_while_r2, filter_attempt]

        return df, filter_attempt
        # In case of no valid SWOT data, the return will be an empty row, and filter_attempt "strict". 

    except Exception as e:
        print(f"[WARN] basin {this_basin} lake {this_lake_id}: {e}", flush=True)
        return None, 'no data'

# -------------------- Basin loop (arrays over basins; joblib within) ----------
for this_basin in unique_basin_ids:
    print('Processing basin:', str(this_basin), '...')

    # Collect lakes in this basin
    all_lake_ids_in_basin = [lake_id for lake_id in all_lake_ids if str(lake_id).startswith(str(this_basin))]

    # Parallel map over lakes
    results = Parallel(
        n_jobs=N_JOBS,
        prefer=JOBLIB_PREFER,
        batch_size="auto",
        verbose=10
    )([delayed(_process_one_lake)(lid, this_basin) for lid in all_lake_ids_in_basin])

    # Unpack results
    dfs = []
    lake_no_data_in_basin = 0 #Initialize count of lakes without valid SWOT data
    lake_count_in_basin = 0 #Initialize count of lakes with valid SWOT observations    
    lake_strict_in_basin = 0 #Initialize count of lakes (with valid SWOT data) for each filtering attempt
    lake_lenient_in_basin = 0
    lake_tukey_in_basin = 0
    lake_no_filter_in_basin = 0

    for df_res, attempt in results:
        if attempt == 'no data': #If this lake has no valid SWOT observations (df_res can be None or empty)
        # When df_res is None, it indicates there is no lake file; 
        # when df_res is empty, it means the Hydrocron lake file has no valid WSE data
        # Both situations are indicated by 'no data'. 
            lake_no_data_in_basin += 1 # Do not append any result
        else: #If this lake has valid SWOT observations.
            dfs.append(df_res)
            lake_count_in_basin += 1
            if attempt == '1_strict':
                lake_strict_in_basin += 1
            elif attempt == '2_lenient':
                lake_lenient_in_basin += 1
            elif attempt == '3_tukey':
                lake_tukey_in_basin += 1
            else: #remaining attempt should be '4_none':
                lake_no_filter_in_basin += 1

    if not dfs: #if dfs == []
        print('Basin:', str(this_basin), 'has no valid LakeSP records.')
        continue # Skip the current basin. 

    # Concatenate per-lake dataframes into one dataframe per basin
    df_this_basin = pd.concat(dfs, ignore_index=True)
    # Note: dfs is guaranteed non-empty here as 'continue' was enforced above otherwise. 

    # Save df_this_basin into a csv file (one CSV per basin)
    out_csv = os.path.join(work_dir_Hydrocron_filtered, f"Hydrocron_prior_{this_basin}.csv")
    df_this_basin.to_csv(out_csv, index=False)

    # Print statistics
    print('% strict (attempt 1):', (lake_strict_in_basin/lake_count_in_basin*100.0))
    print('% lenient (attempt 2):', (lake_lenient_in_basin/lake_count_in_basin*100.0))
    print('% tukey IQR (attempt 3):', (lake_tukey_in_basin/lake_count_in_basin*100.0))
    print('% no filter (attempt 4):', (lake_no_filter_in_basin/lake_count_in_basin*100.0))
    # The above four should add up to 100%. 
    print('% total (100):', ((lake_strict_in_basin + lake_lenient_in_basin + lake_tukey_in_basin \
          + lake_no_filter_in_basin) / lake_count_in_basin * 100.0), \
          str(lake_count_in_basin))
    print('% no data:', (lake_no_data_in_basin/ (lake_count_in_basin + lake_no_data_in_basin) * 100.0))    
    print('Basin:', str(this_basin), '. Filter completed and result saved to:', out_csv)

print("----- Module Completed -----")
print(datetime.datetime.now())
