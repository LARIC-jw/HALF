# -*- coding: utf-8 -*-
"""
Customized LakeSP WSE filter 
Created: 04/20/2025
Last updated: 07/04/2025
Authors: 
    Jida Wang (jidaw@illinois.edu); 
    Melanie Trudel (melanie.trudel@usherbrooke.ca)
Goals: 
    To balance (1) WSE noise removal and (2) a good representation of the all-year lake level hydrograph.   
General structure: 
    There are two major steps in this customized filtering process: 
        - Step 1 (herustic baseline): Subset good observations from full SP time series using a heuristic baseline filter
        - Step 2 (low-pass filtering): Further tidy up the good observations using a low-pass filter (e.g., LOWESS, Savitzky-Golay, etc).
"""




"""
Define global parameters:
    start_time (yyyy-mm-ddThh:mm:ssZ): Starting time of the WSE time series
    end_time (yyyy-mm-ddThh:mm:ssZ):   Ending time of the WSE time series
    
    work_dir (text):                   Directory containing gauge data and storing outputs
    
    SP_retrieval_method (tex):         Preferred method to retrieve the SWOT LakeSP time series: 
                                            - "Hydrocron" - from the Hydrocron API
                                            - "on-premise" - from the csv file previously saved in the local disk
    
    apply_low_pass_filter (text):      "yes" = both heuristic baseline (Step 1) and low-pass filtering (Step 2) will be executed;
                                       "no" = only heuristic baseline (Step 1) will be executed. 
    
    The following parameters only matter if apply_low_pass_filter is set to "yes":
    filter_type (text):                Low-pass filter type: lowess, wavelet, savgol, kalman, spline, median, and hampel.
    z_scores_threshold (float):        Z-score threshold for round-1 (more aggressive) filtering
    z_scores_threshold_r2 (float):     Z-score threshold for round-2 (less aggressive) filtering. This is only needed if r2_filter = 'yes'
    evaluating_at_full_data (text):    "yes" = evaluate outlier removal (z-score clipping) on full LakeSP data; 
                                       "no" = evaluate only on selected observations                                   
    recovering_observations (text):    "yes" = add good-quality observation back after round-1 filtering; "no" = otherwise
    r2_filter (text):                  "yes" = perform another round (round-2) of filtering to remove remaining noise; "no" = otherwise  
    show_filtering_evolution (text):   "yes" = plot how outlier filtering evolves through iteration; "no" = otherwise    
"""
# Global parameters
start_time = "2023-07-26T00:00:00Z" #2023-07-26 is the start of the SWOT nominal orbit.
end_time = "2025-05-31T00:00:00Z"

work_dir = r'D:\D\Research\Projects\SWOT\Initial_global_lakes\Codes\Codes_for_processing_LakeSP'

SP_retrieval_method = 'on-premise' #'Hydrocron' or 'on-premise'

apply_low_pass_filter = 'yes' #'yes' strongly recommended

# The following parameters only matter if apply_low_pass_filter = 'yes'
filter_type = 'savgol' #lowess, wavelet, savgol, kalman, spline, median, hampel.
z_score_threshold = 2.576 #2.576(99% for two tails), 2.807(99.5%), 2.967(99.7%), 3.291(99.9%), 3.5(99.95%)
z_scores_threshold_r2 = 3.5
evaluating_at_full_data = 'no' #'no' recommended
recovering_observations = 'yes' #'yes' recommended
r2_filter = 'yes' #yes recommended
show_filtering_evolution = 'no' #for visualization only; caution: 'yes' may load many figures at the end of the script execution. 




"""
Setup validation metadata
"""
import requests, pywt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d, PchipInterpolator, UnivariateSpline
from io import StringIO
from joblib import Parallel, delayed
from scipy.signal import savgol_filter, medfilt
from pykalman import KalmanFilter
from scipy.stats import spearmanr
import seaborn as sns

# Initialize a dataframe for tested lakes
test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])

#Read in gauge metadata
# Quebec lakes
df_CEHQ = pd.read_csv(work_dir+'/gauge_data/Conversion_EGM08_CEHQ.csv', sep=',', encoding='iso-8859-2')
for n in range(len(df_CEHQ)):
    new_row = pd.DataFrame([{
        'lake_id': df_CEHQ['lake_id (v_105)'][n], #integer
        'gauge_source': 'CEHQ',
        'gauge_dir': work_dir+'/Conversion_EGM08_CEHQ.csv',
        'gauge_id': str(df_CEHQ['station_id'][n]), #string
        'gauge_datum': df_CEHQ['Constante'][n] + df_CEHQ['GPSH'][n] + df_CEHQ['TRX_2024'][n] - df_CEHQ['geoid_swot'][n]
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
    
# Reservoirs in North America
# Load the original CSV file
df_NA = pd.read_csv(work_dir + "/gauge_data/NA-from-Merritt-Harlan/SWOTlake_gagedata_NorthAmerica.csv")
unique_pld_ids = df_NA["lake_id"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    if unique_pld_ids[n] not in [7320361003]: # Inconsistent gauge levels (even for the same gauge_id)
        new_row = pd.DataFrame([{
            'lake_id': unique_pld_ids[n], #integer
            'gauge_source': 'NA',
            'gauge_dir': work_dir + "/gauge_data/NA-from-Merritt-Harlan/SWOTlake_gagedata_NorthAmerica.csv",
            'gauge_id': 'NA', #String
            'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
            }])
        test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)

# Reservoirs in China
# Load the original CSV file
df_China = pd.read_csv(work_dir + "/gauge_data/China-from-Chunqiao-Song/Daily_water_level_for_Chinese_reservoirs.csv")
unique_pld_ids = df_China["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'China',
        'gauge_dir': work_dir + "/gauge_data/China-from-Chunqiao-Song/Daily_water_level_for_Chinese_reservoirs.csv",
        'gauge_id': 'China', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
    
# Reservoirs in West Africa
# Load the original CSV file
df_wf = pd.read_csv(work_dir + "/gauge_data/West-Africa-from-Manuela-Grippa/West_Africa_water_level_meters.csv")
unique_pld_ids = df_wf["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'West_Africa',
        'gauge_dir': work_dir + "/gauge_data/West-Africa-from-Manuela-Grippa/West_Africa_water_level_meters.csv",
        'gauge_id': 'West_Africa', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
    
# Lake Tefe in the Amazon
new_row = pd.DataFrame([{
    'lake_id': 6220321573, #PLD lake ID
    'gauge_source': 'Tefe',
    'gauge_dir': work_dir + '/gauge_data/Lake-Tefe-from-Ayan/LakeTefe_WaterLevel_6220321573.xlsx',
    'gauge_id': 'Tefe',
    'gauge_datum': np.nan
}])
test_lakes = pd.concat([test_lakes, new_row], ignore_index=True) # Reindex the resulting DataFrame with a fresh, sequential index

test_lakeIDs = np.array(test_lakes['lake_id']) # Copy lake_id into an array to allow flexibility for testing lakes without gauge data
 # ----------------If needed, read other lakes without gauge data for visual validation------------------
#test_lakeIDs = [7830011152] #[7830169633] #[7830169882] #[7730019043]
#test_lakeIDs = [7120117532, 7120118473, 7130656702, 7230192673, 7250053712]
#[7520023093] !!
#[7421020203] #[7250674793] #[7120197312] #[7120051812] #[7120010813] #[7110445003]
#test_lakeIDs = test_lakeIDs[0:30]
#test_lakeIDs = [7250051493] #[4350013953, 4340980673] #[4340276603] #[4340457112] #[4330037643] #[7251005212] #[4420009143]
#test_lakeIDs = [7251005983, 7251006483, 7260039322] #[4530746623]
#test_lakeIDs = [1320036143, 1720004433, 2310000033, 2310000143, 2310028773, 2310042313, 2440273723, 2440526623, \
#                2440537523, 2440587433, 2510021752, 2510096362, 2510096392, 2510118183, \
#                2510153912, 2510201732, 2510207233, 2830221613, 3120539933, 3220638553, 3220638593, 3220638613, \
#                3310015943, 3310341183, 4320000782, 4320000792, 4340299353, 4340509682, \
#                4340980803, 4340980873, 4360100373, 4360131693, 4420183353, 4450027763, 4530143033, 4530154843, \
#                4530179393, 4530185723, 4530433243, 4530612753, 4530745303, \
#                4540010073, 4560018033, 4620005313, 4620033793, 4910023252, 6220321573, 6220352303, 6220460253, \
#                6520002392, 6530029163, 6530029383, 6530063353, 6530064763, 6610047672, \
#                6610112613, 6620007912, 7121005653, 7121329183, 7212602202, 7212812023, 7222025243, 7231833283, \
#                7420422213, 7420739872]    
# ---------------------------------------------------------------------------------------------------
# Retrieve unique values in test_lakeIDs while preserving their original order
test_lakeIDs = pd.unique(test_lakeIDs)
# Note: In case a PLD lake IDs is redundant among different gauge sources, we will prefer the first gauge source (the above gauge sources
# have been ranked in an decreasing preference order).




"""
Functions: Do not change the functions unless necessary. 
    Basic functions:
        compute_rmse:              Computes root mean squared error (RMSE), np.nan robust. 
        compute_spearman:          Computes Spearman's rank correlation coefficient (rho)
        remove_tukey_outlier:      Removes outliers using a generalized Tukey method (IQR-based).
        calibrate_heuristic_thresholds: Calibrate heuristic thresholds (max wse_std and wse_u) before SP filtering.
        apply_heuristic_thresholds: Applies heuristic thresholds to subset the baseline for SP filtering.
        filter_ice_outliers:       Removes anomalies ice-covered/freeze-up observations.
        convert_to_daily_series:   Compute daily-interpolated WSEs from SWOT and gauge data over their overlapping time range. 
        signed_min_abs_residual:   Computes the signed residuals with the smallest absolute value across multiple smoothed estimates.
            
    Options of multiple low-pass filters: all allows parallel run. 
        filter_lowess:             LOWESS filter
        filter_savgol:             Savitzky-Golay filter
        filter_wavelet:            Wavelet-based denoising filter
        filter_hampel:             Hampel filter
        filter_spline:             UnivariateSpline filter
        filter_median:             Median filter
        filter_kalman:             Kalman filter
"""
# Define all functions
def compute_rmse(y, y_hat):
    """
    Computes root mean squared error (RMSE)
    
    Parameters:
        y and y_hat (numeric array-like): The two vectors to compare (order does not matter.)
        
    Returns: 
        rmse: Computed RMSE value, or np.nan if no valid data points exist.
    """
    y = np.array(y)
    y_hat = np.array(y_hat)    
    mask = ~np.isnan(y) & ~np.isnan(y_hat)  # Valid (non-NaN) pairs
    
    if np.sum(mask) == 0:
        return np.nan  # Return NaN if all data is invalid
    
    rmse = np.sqrt(np.mean((y[mask] - y_hat[mask])**2))
    return rmse

def compute_spearman(y, y_hat):
    """
    Computes Spearman's rank correlation coefficient (rho)
    
    Parameters:
        y and y_hat (numeric array-like): The two vectors to compare.
        
    Returns: 
        rho: Spearman’s rho, or np.nan if insufficient valid data.
    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    mask = ~np.isnan(y) & ~np.isnan(y_hat)
    
    if np.sum(mask) < 2:
        return np.nan  # Need at least two valid points to compute correlation
    
    rho, _ = spearmanr(y[mask], y_hat[mask])
    return rho

def remove_tukey_outliers(df, col='wse', multiplier=3, lower_q=0.25, upper_q=0.75):
    """
    Removes outliers from a DataFrame column using a generalized Tukey method (IQR-based),
    allowing customizable lower and upper quantile thresholds.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        col (str): Column name on which to perform outlier detection.
        multiplier (float): Multiplier for the IQR to define outlier bounds.
                            Common values: 1.5 (mild outliers), 3 (extreme outliers).
        lower_q (float): Lower quantile for computing the IQR (default = 0.25).
        upper_q (float): Upper quantile for computing the IQR (default = 0.75).

    Returns:
        pd.DataFrame: Filtered copy of the input DataFrame with outliers removed
                      (excluding rows where the target column is NaN).

    Notes:
        - Quantiles must be between 0 and 1, and lower_q < upper_q.
        - This method is robust to non-Gaussian distributions.
        - To preserve NaNs, modify the filtering condition accordingly.
    """
    # Validate quantiles
    if not (0 <= lower_q < upper_q <= 1):
        raise ValueError("Quantiles must be between 0 and 1, with lower_q < upper_q.")
    
    df = df.copy()

    # Compute the specified quantiles
    q_low = df[col].quantile(lower_q)
    q_high = df[col].quantile(upper_q)

    # Compute IQR based on custom quantiles
    iqr = q_high - q_low

    # Define bounds for outlier removal
    lower_bound = q_low - multiplier * iqr
    upper_bound = q_high + multiplier * iqr

    # Filter data within bounds
    df_filtered = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df_filtered, lower_bound, upper_bound

def calibrate_heuristic_thresholds(df, 
                                   by_crid_scenario=True, 
                                   by_pass_id=True, 
                                   wse_std_threshold_max=3, 
                                   wse_std_threshold_min=0,
                                   wse_u_threshold_max=0.5,
                                   wse_u_threshold_min=0.1):
    """
    Computes max wse_std and wse_u thresholds with optional grouping: 
        for each unique combination of crid_scenario and pass_id (if set to be True). 
    Thresholds are capped by user-defined min/max values.

    Parameters:
        df (pd.DataFrame): Input with ['crid', 'pass_id', 'wse_std', 'wse_u']
        by_crid_scenario (bool): If True, use crid_scenario for grouping
        by_pass_id (bool): If True, use pass_id for grouping
        wse_std_threshold_max/min (float): Max/min cap for wse_std threshold        
        wse_u_threshold_max/min (float): Max/min cap for wse_u threshold

    Returns:
        pd.DataFrame: Contains ['lake_id', 'crid_scenario', 'pass_id', 'wse_std_threshold', 'wse_u_threshold']
          > lake_id: PLD lake id of the input df. 
          > crid_scenario: "PIC2_or_PID0" or "early_versions" (e.g., PIC0, PGC0); or 'global' if df is empty
          > pass_id: SWOT orbit pass; or 'global' if df is empty
          > wse_std_threshold: the maximum wse_std threshold under this pass_id and crid_scenario combination
          > wse_u_threshold: the maximum wse_u threshold under this pass_id and crid_scenario combination
    """
    # Extract lake_id (assumed unique)
    lake_id = df['lake_id'].iloc[0] if 'lake_id' in df.columns and not df.empty else 'unknown'

    # Handle empty input case
    if len(df) == 0:
        return pd.DataFrame([{
            'lake_id': lake_id,
            'crid_scenario': 'global',
            'pass_id': 'global',
            'wse_std_threshold': max(wse_std_threshold_max, wse_std_threshold_min),
            'wse_u_threshold': max(wse_u_threshold_max, wse_u_threshold_min)
        }])

    df = df.copy()

    # Define CRID scenario
    df['crid_scenario'] = df['crid'].apply(lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions')

    # Define unique combinations
    combinations = df[['crid_scenario', 'pass_id']].drop_duplicates()

    # Compute global max thresholds with fallback
    global_max_std = df['wse_std'].max(skipna=True)
    global_max_std = min(max(global_max_std if pd.notna(global_max_std) else wse_std_threshold_max,
                             wse_std_threshold_min), wse_std_threshold_max)

    global_max_u = df['wse_u'].max(skipna=True)
    global_max_u = min(max(global_max_u if pd.notna(global_max_u) else wse_u_threshold_max,
                           wse_u_threshold_min), wse_u_threshold_max)

    # Decide grouping keys
    if by_crid_scenario and by_pass_id:
        group_keys = ['crid_scenario', 'pass_id']
    elif by_crid_scenario:
        group_keys = ['crid_scenario']
    elif by_pass_id:
        group_keys = ['pass_id']
    else:
        combinations['wse_std_threshold'] = global_max_std
        combinations['wse_u_threshold'] = global_max_u
        combinations.insert(0, 'lake_id', lake_id)
        return combinations

    # Group and aggregate
    grouped = df.groupby(group_keys).agg({
        'wse_std': 'max',
        'wse_u': 'max'
    }).reset_index().rename(columns={
        'wse_std': 'wse_std_threshold',
        'wse_u': 'wse_u_threshold'
    })

    # Merge with all combinations to ensure coverage
    merged = combinations.merge(grouped, on=group_keys, how='left')

    # Fill missing with global
    merged['wse_std_threshold'] = merged['wse_std_threshold'].fillna(global_max_std)
    merged['wse_u_threshold'] = merged['wse_u_threshold'].fillna(global_max_u)

    # Clip to enforce caps
    merged['wse_std_threshold'] = merged['wse_std_threshold'].clip(wse_std_threshold_min, wse_std_threshold_max)
    merged['wse_u_threshold'] = merged['wse_u_threshold'].clip(wse_u_threshold_min, wse_u_threshold_max)

    # Insert lake_id as the first column
    merged.insert(0, 'lake_id', lake_id)

    return merged 

def apply_heuristic_thresholds(df, thresholds_df, wse_std_threshold_max=3):
    """
    Applies heuristic thresholds to filter rows in `df` where both wse_std and wse_u
    are smaller than their thresholds (as defined in thresholds_df). 

    Hierarchical matching:
     > Full match: crid_scenario + pass_id
     > Fallback: crid_scenario only (use max of its thresholds)
     > Fallback: pass_id only (use max of its thresholds)
     > Global fallback: use global maximum thresholds from entire thresholds_df
     
    In other words, Tries full match: crid_scenario + pass_id
    If unmatched:
        > Tries fallback by crid_scenario only
        > Or by pass_id only
    If still unmatched:
        > Applies global maximum of thresholds
        
    Special case:
     > If ice_clim_f > 0 (ice covered), apply wse_std_threshold_max for wse_std.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns ['crid', 'pass_id', 'wse_std', 'wse_u', 'ice_clim_f']
        thresholds_df (pd.DataFrame): Output of calibrate_heuristic_thresholds()
        wse_std_threshold_max (float): Fixed threshold to apply if ice_clim_f > 0

    Returns:
        pd.DataFrame: Filtered rows that meet the heuristic threshold criteria, retaining only original columns.
    """
    thresholds_df = thresholds_df.copy()  # Prevent modifying the original
    thresholds_df.drop(columns=['lake_id'], inplace=True) # Drop lake_id to avoid column duplicates (suffix) in merging
    df = df.copy()
    original_columns = df.columns.tolist()

    # Assign crid_scenario
    df['crid_scenario'] = df['crid'].apply(lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions')

    # Create temporary string keys for type-safe merge: this is needed as thresholds_df.pass_id can be "global", 
    # which does not match the integer type of df.pass_id, leading to a potential error. 
    # The temporary string keys will later be deleted. 
    df['_pass_id_str'] = df['pass_id'].astype(str)
    df['_crid_scenario_str'] = df['crid_scenario'].astype(str)
    thresholds_df['_pass_id_str'] = thresholds_df['pass_id'].astype(str)
    thresholds_df['_crid_scenario_str'] = thresholds_df['crid_scenario'].astype(str)

    # Merge using stringified keys
    df = df.merge(
        thresholds_df,
        left_on=['_crid_scenario_str', '_pass_id_str'],
        right_on=['_crid_scenario_str', '_pass_id_str'],
        how='left',
        suffixes=('', '_thresh') #This only matters if there are duplicate fields, which should not happen. 
    )

    # Fallbacks for unmatched thresholds
    unmatched = df['wse_std_threshold'].isna() | df['wse_u_threshold'].isna()
    if unmatched.any():
        # Prepare max thresholds by crid_scenario
        max_by_crid = thresholds_df.groupby('crid_scenario').agg({
            'wse_std_threshold': 'max',
            'wse_u_threshold': 'max'
        }).reset_index()
        # Prepare max thresholds by pass_id
        max_by_pass = thresholds_df.groupby('pass_id').agg({
            'wse_std_threshold': 'max',
            'wse_u_threshold': 'max'
        }).reset_index()
        # Global max fallback
        global_max_std = thresholds_df['wse_std_threshold'].max()
        global_max_u = thresholds_df['wse_u_threshold'].max()

        for idx in df.index[unmatched]:
            row = df.loc[idx]
            fallback_std = fallback_u = None

            # Try crid_scenario fallback
            match_crid = max_by_crid[max_by_crid['crid_scenario'] == row['crid_scenario']]
            if not match_crid.empty:
                fallback_std = match_crid['wse_std_threshold'].values[0]
                fallback_u = match_crid['wse_u_threshold'].values[0]
            else:
                # Try pass_id fallback
                match_pass = max_by_pass[max_by_pass['pass_id'] == row['pass_id']]
                if not match_pass.empty:
                    fallback_std = match_pass['wse_std_threshold'].values[0]
                    fallback_u = match_pass['wse_u_threshold'].values[0]

            # If still none, apply global max
            if fallback_std is None or fallback_u is None:
                fallback_std = global_max_std
                fallback_u = global_max_u

            df.at[idx, 'wse_std_threshold'] = fallback_std
            df.at[idx, 'wse_u_threshold'] = fallback_u

    # Apply ice-clim override
    if 'ice_clim_f' in df.columns:
        df.loc[df['ice_clim_f'] > 0, 'wse_std_threshold'] = wse_std_threshold_max

    # Drop temp merge keys
    df.drop(columns=['_pass_id_str', '_crid_scenario_str'], inplace=True)

    # Apply filtering
    mask = (df['wse_std'] <= df['wse_std_threshold']) & \
           (df['wse_u'] <= df['wse_u_threshold'])

    return df.loc[mask, original_columns]

def filter_ice_outliers(df, remove_tukey_outliers, by_pass=True, by_crid_scenario=True,
                        multiplier=3, lower_q=0.25, upper_q=0.75, used_q='upper'):
    """
    Removes ice-covered/freeze-up observations where area_total or wse is a Tukey outlier
    relative to comparable ice-free observations, with optional grouping and directional outlier filtering.
    Output preserves the original row order and original columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame
        remove_tukey_outliers (function): Returns (filtered_df, lower_bound, upper_bound)
        by_pass (bool): Whether to group by pass_id
        by_crid_scenario (bool): Whether to group by crid_scenario
        multiplier (float): Tukey IQR multiplier
        lower_q (float): Lower quantile
        upper_q (float): Upper quantile
        used_q (str): Which bound(s) to use: 'upper', 'lower', or 'both'

    Returns:
        pd.DataFrame: Filtered DataFrame with ice-covered outliers removed,
                      preserving the original row order and original columns.
                      
    Logic: LakeSP observations tend to be more uncertain during freeze-up periods. This function provides
    an option to compare freeze-up observations (in both area_total and WSE) with ice-free observations, and 
    remove possible errors during the freeze-up period. 
    
    Caution: Some reservoirs can experience significant water level draw-downs during the freeze-up period. 
    So caveats are needed when using this function to remove negative anomalies, which could be true signals.
    Therefore, we provide an option for the filtering direction ("used_q"), and we recommend using filtering
    positive anomalies as high lake water area or WSE during the freeze-up period are less likely and are probably errors. 
    """
    df = df.copy()
    original_columns = df.columns.tolist()

    # Define crid_scenario
    df['crid_scenario'] = df['crid'].apply(lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions')

    # Split into ice-covered and ice-free subsets: use fully ice-covered to avoid over-rejection. 
    # Note: in some cases, there are ice-covered measurements that are in high WSEs but valid. 
    df_ice = df[df['ice_clim_f'] > 1].copy()
    df_noice = df[df['ice_clim_f'] <= 1].copy()

    # If no ice-free data exists, return original DataFrame (original columns only)
    if df_noice.empty:
        return df[original_columns]

    rows_to_keep = []
    
    # Loop through each ice-covered observation
    for idx, row in df_ice.iterrows():
        condition = pd.Series(True, index=df_noice.index)

        if by_pass and row['pass_id'] in df_noice['pass_id'].values:
            condition &= (df_noice['pass_id'] == row['pass_id'])

        if by_crid_scenario and row['crid_scenario'] in df_noice['crid_scenario'].values:
            condition &= (df_noice['crid_scenario'] == row['crid_scenario'])

        reference_group = df_noice[condition]
        if reference_group.empty:
            reference_group = df_noice

        # Check if there's enough data to compute bounds
        valid_area = reference_group['area_total'].dropna()
        valid_wse = reference_group['wse'].dropna()
        if len(valid_area) < 2 or len(valid_wse) < 2:
            # Not enough data for meaningful Tukey bounds — retain the point
            rows_to_keep.append(idx)
            continue

        # Tukey bounds
        _, area_lb, area_ub = remove_tukey_outliers(reference_group, col='area_total',
                                                    multiplier=multiplier,
                                                    lower_q=lower_q, upper_q=upper_q)
        _, wse_lb, wse_ub = remove_tukey_outliers(reference_group, col='wse',
                                                  multiplier=multiplier,
                                                  lower_q=lower_q, upper_q=upper_q)

        val_area = row['area_total']
        val_wse = row['wse']

        # Apply bounds conditionally
        is_area_ok = pd.notna(val_area)
        if used_q == 'both':
            is_area_ok &= (area_lb <= val_area <= area_ub)
        elif used_q == 'upper':
            is_area_ok &= (val_area <= area_ub)
        elif used_q == 'lower':
            is_area_ok &= (val_area >= area_lb)

        is_wse_ok = pd.notna(val_wse)
        if used_q == 'both':
            is_wse_ok &= (wse_lb <= val_wse <= wse_ub)
        elif used_q == 'upper':
            is_wse_ok &= (val_wse <= wse_ub)
        elif used_q == 'lower':
            is_wse_ok &= (val_wse >= wse_lb)

        if is_area_ok and is_wse_ok:
            rows_to_keep.append(idx)

    # Combine and return only original columns in original order
    df_ice_filtered = df_ice.loc[rows_to_keep]
    df_combined = pd.concat([df_ice_filtered, df_noice])
    return df_combined.loc[df.index.intersection(df_combined.index)].sort_index()[original_columns]

def convert_to_daily_series (df, gauge_df, 
                      time_col='datetime',
                      gauge_time_col='gauge_datetime',
                      wse_col='wse',
                      wse_filtered_col='wse_adjusted',
                      gauge_wse_col='gauge_wse',
                      interp_method='linear'):
    """
    Compute daily-interpolated WSE time series from SWOT (raw and adjusted) and
    gauge data over their overlapping date range. 

    Returns NaN for all outputs if either df or gauge_df is empty.
    
    This function was updated on 07/01/2025, from previous "compute_daily_variability"
    """
    
    # Check for empty inputs
    if df is None or gauge_df is None or df.empty or gauge_df.empty:
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    # Copy and convert timestamps
    df = df.copy()
    gauge_df = gauge_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    gauge_df[gauge_time_col] = pd.to_datetime(gauge_df[gauge_time_col])

    # Align to daily floor
    df['date'] = df[time_col].dt.floor('D')
    gauge_df['date'] = gauge_df[gauge_time_col].dt.floor('D')

    # Daily averages
    wse_daily = df.groupby('date')[wse_col].mean()
    wse_filtered_daily = df.groupby('date')[wse_filtered_col].mean()
    gauge_daily = gauge_df.groupby('date')[gauge_wse_col].mean()

    # Ensure datetime64[ns] index
    wse_daily.index = pd.to_datetime(wse_daily.index)
    wse_filtered_daily.index = pd.to_datetime(wse_filtered_daily.index)
    gauge_daily.index = pd.to_datetime(gauge_daily.index)

    # Check again for valid overlapping range
    if wse_daily.empty or wse_filtered_daily.empty or gauge_daily.empty:
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    start_date = max(wse_daily.index.min(), gauge_daily.index.min())
    end_date = min(wse_daily.index.max(), gauge_daily.index.max())

    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Interpolate and fill
    def safe_interp(series):
        return (series.reindex(date_range)
                      .interpolate(method=interp_method, limit_direction='both')
                      .bfill()
                      .ffill())

    wse_interp = safe_interp(wse_daily)
    wse_filtered_interp = safe_interp(wse_filtered_daily)
    gauge_interp = safe_interp(gauge_daily)

    return {
        'daily_wse': wse_interp, #unfiltered
        'daily_wse_filtered': wse_filtered_interp, #filtered
        'daily_gauge': gauge_interp #gauge
    }         

def signed_min_abs_residual(A, B):
    """
    Computes the signed residuals with the smallest absolute value across multiple smoothed estimates.

    Parameters:
        A (ndarray): 2D array of shape (n_models, n_points) containing smoothed values
                     from multiple parameter combinations (e.g., frac/it from LOWESS).
        B (array-like): 1D array of shape (n_points,) with the original observed values
                        to compare against.

    Returns:
        result (np.ndarray): 1D array of shape (n_points,) with the residual from the model
                             that gives the minimum absolute error at each point.
                             Sign is preserved.
    """
    # Convert B to NumPy array and broadcast for subtraction
    B = np.asarray(B)
    E = A - B  # Residuals: each row = one smoothed curve; each col = one time step

    # Compute absolute residuals, masking NaNs by setting them to infinity
    abs_E = np.abs(E)
    abs_E[np.isnan(abs_E)] = np.inf  # So NaNs are ignored when selecting the minimum

    # Find index of minimum absolute residual at each time step (column-wise)
    idx = np.argmin(abs_E, axis=0)

    # Use those indices to extract the original signed residuals from E
    result = E[idx, np.arange(E.shape[1])]

    return result

def filter_lowess(data, 
                  value_col='value', 
                  time_col='datetime',
                  eval_times=None,
                  minfrac=0.1, 
                  maxfrac=0.5, 
                  frac_step=0.1, 
                  it_v=[0, 1], 
                  n_jobs=-1):
    """
    LOWESS-based denoising filter for irregular time series data using multiple
    (frac, it) combinations, with parallel execution and envelope estimation.

    Parameters:
        data (DataFrame): Input time series with [time_col, value_col].
        value_col (str): Column name for signal values.
        time_col (str): Column name for datetime values.
        eval_times (array-like): Times to evaluate the filtered result (default: same as input times).
        minfrac (float): Minimum LOWESS smoothing span (0 < frac < 1).
        maxfrac (float): Maximum LOWESS smoothing span.
        frac_step (float): Step size for generating frac values.
        it_v (list of int): Iteration values to test (for robustness to outliers).
        n_jobs (int): Number of parallel jobs (default: -1 for all cores).

    Returns:
        bottom (np.ndarray): Lower bound (min) of all smoothed estimates.
        top (np.ndarray): Upper bound (max) of all smoothed estimates.
        smoothed_evals (2D np.ndarray): All smoothed curves (rows = frac/it combinations).
    """

    # Prepare and sort data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    if eval_times is None:
        eval_times = data[time_col]
    else:
        eval_times = pd.to_datetime(eval_times)

    x = data[time_col].astype('int64')   # Convert datetime to int64 for compatibility
    y = data[value_col].values
    eval_x = eval_times.astype('int64')

    # Generate (frac, it) parameter combinations
    frac_v = np.linspace(minfrac, maxfrac, int(np.ceil((maxfrac - minfrac) / frac_step) + 1))
    param_list = [(f, it) for it in it_v for f in frac_v]

    #  Define worker function ---
    def smooth_worker(frac, it):
        try:
            # Direct LOWESS evaluation at desired timestamps
            # The xvals argument tells the LOWESS function: 
            #     “do not just smooth at the original x; instead, evaluate the smoothed curve at xvals.”
            # Note: using xvals can fail (NaNs) if extrapolation or sparse data at edges
            result = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, it=it, frac=frac)
            
            # Check for NaNs — indicates failure in direct smoothing
            if np.isnan(result).any():
                raise ValueError("NaN encountered in LOWESS output")

        except:
            # Fallback: LOWESS at original x, then interpolate to eval_x
            raw_result = sm.nonparametric.lowess(exog=x, endog=y, it=it, frac=frac)
            # Use quadratic interpolation (linear is okay too)
            interp_func = interp1d(raw_result[:, 0], raw_result[:, 1], 
                                   kind='quadratic', fill_value='extrapolate')
            #For xvals outside the range of x, it will perform extrapolation using the slope of the nearest segment at the boundary.
            #Caution: Lowess and then interpolation may not give exact results as if xvals could run successfully.
            result = interp_func(eval_x)

        return result

    # Run smoothing in parallel across all (frac, it) parameter combinations
    smoothed_values_list = Parallel(n_jobs=n_jobs)(
        delayed(smooth_worker)(frac, it) for frac, it in param_list
    )
    
    # Combine all smoothed results into a 2D array (one row per (frac, it) pair)
    smoothed_evals = np.vstack(smoothed_values_list)

    # Compute min/max envelope bounds ---
    bottom = np.nanmin(smoothed_evals, axis=0)
    top = np.nanmax(smoothed_evals, axis=0)

    return bottom, top, smoothed_evals

def filter_savgol(
    data,
    value_col='value',
    time_col='datetime',
    eval_times=None,
    window_length_v=[11, 21, 31], # full window widths in days (must be odd integers)
    polyorder_v=[2, 3],
    inter_freq='1D',
    interpolation_method='linear',
    n_jobs=-1
):
    """
    Savitzky-Golay filter over combinations of window_length and polyorder,
    and return empirical bounds and all smoothed results.
    Different from lowess, unequal timestamps need to be interpolated to a regular grid.

    Parameters:
        data (DataFrame): Time series data with columns [time_col, value_col].
        eval_times (array-like): Evaluation points (defaults to original timestamps).
        window_length_v (list): List of window lengths (full width) in days to test (must be odd).
        polyorder_v (list): List of polynomial orders to test.
        inter_freq (str): '1D' (daily) or '1H' (hourly) interpolation grid.
        interpolation_method (str): 'linear' or 'pchip'.
        n_jobs (int): Number of parallel jobs (default: all cores).

    Returns:
        bottom (array): Lower bound of smoothed estimates.
        top (array): Upper bound of smoothed estimates.
        smoothed_evals (2D array): Smoothed values (one row per param combo).
    """

    # Prepare and sort data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    if eval_times is None:
        eval_times = data[time_col]
    else:
        eval_times = pd.to_datetime(eval_times)

    x_orig = data[time_col].astype('int64')
    y_orig = data[value_col].values
    x_eval = eval_times.astype('int64')

    # Build regular interpolation grid
    t_start = data[time_col].min()
    t_end = data[time_col].max()
    regular_time = pd.date_range(start=t_start, end=t_end, freq=inter_freq)
    x_regular = regular_time.astype('int64')

    # Interpolate to regular time grid
    if interpolation_method == 'linear':
        interpolated_values = np.interp(x_regular, x_orig, y_orig)
    elif interpolation_method == 'pchip':
        pchip = PchipInterpolator(x_orig, y_orig, extrapolate=True)
        interpolated_values = pchip(x_regular)
    else:
        raise ValueError("interpolation_method must be 'linear' or 'pchip'")

    # Define (window_length, polyorder) combinations
    param_list = [(wl, po) for po in polyorder_v for wl in window_length_v]

    # Worker function
    def smooth_worker(window_length, polyorder):
        try:
            wl = window_length

            # Convert to hourly window if needed
            if inter_freq == '1H':
                wl = window_length * 24

            # Adjust window to be odd and valid
            if wl >= len(interpolated_values):
                wl = len(interpolated_values) // 2 * 2 + 1  # largest odd number < len
            if wl <= polyorder:
                wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3   
    
            # Apply Savitzky-Golay filter
            smoothed = savgol_filter(interpolated_values, window_length=wl, polyorder=polyorder)

            # Interpolate to eval_x
            if interpolation_method == 'linear':
                return np.interp(x_eval, x_regular, smoothed)
            else:
                interp_func = PchipInterpolator(x_regular, smoothed, extrapolate=True)
                return interp_func(x_eval)

        except Exception as e:
            print(f"Savgol failed for window_length={window_length}, polyorder={polyorder}: {e}")
            return np.full(len(eval_times), np.nan)

    # Run in parallel
    smoothed_values_list = Parallel(n_jobs=n_jobs)(
        delayed(smooth_worker)(wl, po) for wl, po in param_list
    )

    smoothed_evals = np.vstack(smoothed_values_list)

    # Compute bounds
    bottom = np.nanmin(smoothed_evals, axis=0)
    top = np.nanmax(smoothed_evals, axis=0)

    return bottom, top, smoothed_evals

def filter_wavelet(
    data, 
    value_col='value', 
    time_col='datetime',
    eval_times=None,
    wavelet_v=['db4', 'sym2', 'coif1'],  # list of wavelets to test
    level=None, 
    threshold=None,
    inter_freq='1D',
    interpolation_method='linear',
    n_jobs=-1
):
    """
    Wavelet-based denoising over multiple wavelet types with confidence bounds. 
    Different from lowess, unequal timestamps need to be interpolated to a regular grid. 
    
    Parameters:
        data (DataFrame): Input time series with [time_col, value_col].
        eval_times (array-like): Timestamps to evaluate the filtered result.
        wavelet_v (list): List of wavelet names to test (e.g., ['db4', 'sym2']).
        level (int or None): Decomposition level. Auto-adjusted if None.
        threshold (float or None): Threshold for soft denoising.
        inter_freq (str): Regular time grid frequency: '1D' or '1H'.
        interpolation_method (str): 'linear' or 'pchip'.
        n_jobs (int): Number of parallel jobs (default: all cores).
        
    Returns:
        bottom (array): Min bound across all wavelet types.
        top (array): Max bound across all wavelet types.
        smoothed_evals (2D array): Denoised curves (one row per wavelet).
    """

    # Prepare and sort data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    if eval_times is None:
        eval_times = data[time_col]
    else:
        eval_times = pd.to_datetime(eval_times)

    # Build regular time grid
    t_start = data[time_col].min()
    t_end = data[time_col].max()
    regular_time = pd.date_range(start=t_start, end=t_end, freq=inter_freq)

    x_orig = data[time_col].astype('int64')
    x_regular = regular_time.astype('int64')
    x_eval = eval_times.astype('int64')

    # Interpolate to regular grid
    if interpolation_method == 'linear':
        interpolated_values = np.interp(x_regular, x_orig, data[value_col].values)
    elif interpolation_method == 'pchip':
        pchip = PchipInterpolator(x_orig, data[value_col].values, extrapolate=True)
        interpolated_values = pchip(x_regular)
    else:
        raise ValueError("interpolation_method must be 'linear' or 'pchip'")

    # Worker function for each wavelet type
    def smooth_worker(wavelet_name):
        try:
            wavelet = pywt.Wavelet(wavelet_name)
            max_level = pywt.dwt_max_level(len(interpolated_values), wavelet.dec_len)

            # Adjust decomposition level if not specified
            lev = level
            if lev is None:
                lev = min(8, max_level) if inter_freq == '1H' else min(5, max_level)

            # Decompose
            coeffs = pywt.wavedec(interpolated_values, wavelet, level=lev)

            # Estimate noise
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thres = threshold if threshold is not None else sigma * np.sqrt(2 * np.log(len(interpolated_values)))

            # Apply soft thresholding to detail coefficients
            coeffs[1:] = [pywt.threshold(c, thres, mode='soft') for c in coeffs[1:]]

            # Reconstruct and trim
            denoised = pywt.waverec(coeffs, wavelet)[:len(regular_time)]

            # Interpolate denoised back to eval_times
            if interpolation_method == 'linear':
                return np.interp(x_eval, x_regular, denoised)
            else:
                interp_func = PchipInterpolator(x_regular, denoised, extrapolate=True)
                return interp_func(x_eval)

        except Exception as e:
            print(f"Wavelet '{wavelet_name}' failed: {e}")
            return np.full(len(eval_times), np.nan)

    # Parallel execution across wavelet types ---
    smoothed_values_list = Parallel(n_jobs=n_jobs)(
        delayed(smooth_worker)(wavelet_name) for wavelet_name in wavelet_v
    )

    smoothed_evals = np.vstack(smoothed_values_list)

    # Confidence bounds ---
    bottom = np.nanmin(smoothed_evals, axis=0)
    top = np.nanmax(smoothed_evals, axis=0)

    return bottom, top, smoothed_evals
  
def filter_hampel(
    data,
    value_col='value',
    time_col='datetime',
    eval_times=None,
    window_length_v=[11, 21, 31],  # full window widths in days (must be odd integers)
    inter_freq='1D',               # '1D' (daily) or '1H' (hourly) regular grid
    interpolation_method='linear',  # 'linear' or 'pchip'
    n_jobs=-1
):
    """
    Hampel filter over multiple window lengths with confidence bounds.
    Different from lowess, unequal timestamps need to be interpolated to a regular grid.
    
    Parameters:
        data (DataFrame): Input time series with [time_col, value_col].
        eval_times (array-like): Timestamps to evaluate the filtered result.
        window_length_v (list): List of full window lengths (must be odd) in days (will be scaled if for '1H').
        inter_freq (str): '1D' or '1H' interpolation frequency.
        interpolation_method (str): 'linear' or 'pchip'.
        n_jobs (int): Number of parallel jobs.

    Returns:
        bottom (array): Lower bound of filtered estimates.
        top (array): Upper bound of filtered estimates.
        smoothed_evals (2D array): One row per window length.
    """

    # Prepare data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    if eval_times is None:
        eval_times = data[time_col]
    else:
        eval_times = pd.to_datetime(eval_times)

    # Regular time grid (daily or hourly)
    t_start = data[time_col].min().normalize()
    t_end = data[time_col].max().normalize()
    regular_time = pd.date_range(start=t_start, end=t_end, freq=inter_freq)

    x_orig = data[time_col].astype('int64')
    y_orig = data[value_col].values
    x_regular = regular_time.astype('int64')
    x_eval = eval_times.astype('int64')

    # Interpolate to regular grid
    if interpolation_method == 'linear':
        interpolated_values = np.interp(x_regular, x_orig, y_orig)
    elif interpolation_method == 'pchip':
        pchip = PchipInterpolator(x_orig, y_orig, extrapolate=True)
        interpolated_values = pchip(x_regular)
    else:
        raise ValueError("interpolation_method must be 'linear' or 'pchip'")

    # Hampel filter worker
    def smooth_worker(window_length_days):
        try:
            # Scale window length to hours if needed
            wl = window_length_days * 24 if inter_freq == '1H' else window_length_days

            # Ensure odd window length
            if wl % 2 == 0:
                wl += 1
            half_width = wl // 2

            denoised = interpolated_values.copy()
            L = 1.4826  # scale factor for Gaussian distribution
            n_sigmas = 3
            n = len(denoised)

            for i in range(half_width, n - half_width):
                window = denoised[i - half_width:i + half_width + 1]
                median = np.median(window)
                mad = L * np.median(np.abs(window - median))
                if mad == 0:
                    continue
                if np.abs(denoised[i] - median) > n_sigmas * mad:
                    denoised[i] = median

            # Interpolate to eval_times
            if interpolation_method == 'linear':
                return np.interp(x_eval, x_regular, denoised)
            else:
                interp_func = PchipInterpolator(x_regular, denoised, extrapolate=True)
                return interp_func(x_eval)

        except Exception as e:
            print(f"Hampel failed for window_length={window_length_days}: {e}")
            return np.full(len(eval_times), np.nan)

    # Run all permutations in parallel
    smoothed_values_list = Parallel(n_jobs=n_jobs)(
        delayed(smooth_worker)(wl) for wl in window_length_v
    )

    smoothed_evals = np.vstack(smoothed_values_list)

    # Compute envelope bound
    bottom = np.nanmin(smoothed_evals, axis=0)
    top = np.nanmax(smoothed_evals, axis=0)

    return bottom, top, smoothed_evals

def filter_spline(
    data,
    value_col='value',
    time_col='datetime',
    eval_times=None,
    smoothing_factor_v=[1e5, 1e6, 1e7],
    n_jobs=-1
):
    """
    UnivariateSpline filter across multiple smoothing factors
    and return bounds and all smoothed results. Works directly on irregular time steps.
    Spline filter can handle unequal timestamps. 

    Parameters:
        data (DataFrame): Input data with columns [time_col, value_col].
        eval_times (array-like): Times to evaluate the smoothed output. Defaults to input times.
        smoothing_factor_v (list): List of smoothing factor values to test.
        n_jobs (int): Number of parallel jobs to use.

    Returns:
        bottom (np.ndarray): Lower envelope across all smoothed results.
        top (np.ndarray): Upper envelope across all smoothed results.
        smoothed_evals (2D np.ndarray): Each row is a smoothed result for a given `s`.
    """

    # Prepare data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    if eval_times is None:
        eval_times = data[time_col]
    else:
        eval_times = pd.to_datetime(eval_times)

    # Convert datetime to seconds since epoch
    t_numeric = data[time_col].astype('int64') / 1e9
    t_eval_numeric = eval_times.astype('int64') / 1e9
    y = data[value_col].values

    # Spline smoothing worker
    def smooth_worker(s):
        try:
            spline = UnivariateSpline(t_numeric, y, s=s)
            return spline(t_eval_numeric)
        except Exception as e:
            print(f"Spline failed for smoothing factor {s}: {e}")
            return np.full(len(eval_times), np.nan)

    # Run in parallel
    smoothed_values_list = Parallel(n_jobs=n_jobs)(
        delayed(smooth_worker)(s) for s in smoothing_factor_v
    )

    smoothed_evals = np.vstack(smoothed_values_list)

    # Compute envelope bounds
    bottom = np.nanmin(smoothed_evals, axis=0)
    top = np.nanmax(smoothed_evals, axis=0)

    return bottom, top, smoothed_evals

def filter_median(
    data,
    value_col='value',
    time_col='datetime',
    eval_times=None,
    window_length_v=[11, 21, 31],
    inter_freq='1D',
    interpolation_method='linear',
    n_jobs=-1
):
    """
    Median filtering across multiple kernel sizes, optionally adjusting for hourly/daily
    interpolation, and return bounds and all smoothed results.
    Different from lowess, unequal timestamps need to be interpolated to a regular grid.

    Parameters:
        data (DataFrame): Time series with columns [time_col, value_col].
        value_col (str): Column name of the values to be smoothed.
        time_col (str): Column name of the time variable.
        eval_times (array-like): Timestamps to evaluate the filtered result (optional).
        window_length_v (list of int): List of full window lengths (must be odd) to use in median filter.
        inter_freq (str): Frequency of regular interpolation grid, e.g., '1D' or '1H'.
        interpolation_method (str): 'linear' or 'pchip' for interpolation scheme.
        n_jobs (int): Number of parallel jobs (-1 uses all available cores).

    Returns:
        bottom (array): Lower envelope of smoothed estimates.
        top (array): Upper envelope of smoothed estimates.
        smoothed_evals (2D array): Each row is a smoothed result for a given kernel size.
    """

    # Ensure datetime format and sort by time
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    # Set evaluation times to original timestamps if not provided
    if eval_times is None:
        eval_times = data[time_col]
    else:
        eval_times = pd.to_datetime(eval_times)

    # Convert time to numeric (int64 = nanoseconds since epoch)
    x_orig = data[time_col].astype('int64')
    y_orig = data[value_col].values
    x_eval = eval_times.astype('int64')

    # Create a regular time grid based on specified interpolation frequency
    t_start = data[time_col].min()
    t_end = data[time_col].max()
    regular_time = pd.date_range(start=t_start, end=t_end, freq=inter_freq)
    x_regular = regular_time.astype('int64')

    # Interpolate y values to the regular grid
    if interpolation_method == 'linear':
        interpolated_values = np.interp(x_regular, x_orig, y_orig)
    elif interpolation_method == 'pchip':
        pchip = PchipInterpolator(x_orig, y_orig, extrapolate=True)
        interpolated_values = pchip(x_regular)
    else:
        raise ValueError("interpolation_method must be 'linear' or 'pchip'")

    # Worker to apply median filter with a given kernel size
    def smooth_worker(kernel_size_days):
        try:
            # Convert kernel size to hours if interpolation is hourly
            ks = kernel_size_days * 24 if inter_freq == '1H' else kernel_size_days

            # Ensure kernel size is an odd positive integer
            if ks % 2 == 0:
                ks += 1
            if ks < 1:
                raise ValueError("kernel size must be at least 1")

            # Make sure kernel size is not larger than the signal length
            if ks >= len(interpolated_values):
                ks = len(interpolated_values) // 2 * 2 + 1  # max valid odd size

            # Apply median filter
            denoised = medfilt(interpolated_values, kernel_size=ks)

            # Interpolate the denoised result back to the original or evaluation timestamps
            if interpolation_method == 'linear':
                return np.interp(x_eval, x_regular, denoised)
            else:
                interp_func = PchipInterpolator(x_regular, denoised, extrapolate=True)
                return interp_func(x_eval)

        except Exception as e:
            print(f"Median filter failed for kernel size {kernel_size_days}: {e}")
            return np.full(len(eval_times), np.nan)

    # Run filtering for all kernel sizes in parallel
    smoothed_values_list = Parallel(n_jobs=n_jobs)(
        delayed(smooth_worker)(ks) for ks in window_length_v
    )

    # Stack all results into a 2D array
    smoothed_evals = np.vstack(smoothed_values_list)

    # Compute lower and upper bounds across all filtering results
    bottom = np.nanmin(smoothed_evals, axis=0)
    top = np.nanmax(smoothed_evals, axis=0)

    return bottom, top, smoothed_evals

def filter_kalman(
    data,
    value_col='value',
    time_col='datetime',
    eval_times=None,
    inter_freq='1D',                # '1D' or '1H'
    interpolation_method='linear'  # 'linear' or 'pchip'
):
    """
    Kalman filter (with EM-optimized parameters) on a time series.

    Parameters:
        data (DataFrame): Input time series with columns [time_col, value_col].
        value_col (str): Column name for values.
        time_col (str): Column name for timestamps.
        eval_times (array-like): Times to evaluate the smoothed result. Defaults to original timestamps.
        inter_freq (str): Frequency of the regular interpolation grid ('1D' or '1H').
        interpolation_method (str): Method for interpolation to regular grid ('linear' or 'pchip').

    Returns: 
        Note there is no permutation for kalman filter (bottom = top, and smoothed_evals contains only one array), 
        but for consistency, we return the outputs in the same format as the other filters. 
        bottom (array): Lower envelope of smoothed estimates.
        top (array): Upper envelope of smoothed estimates.
        smoothed_evals (2D array): Each row is a smoothed result at eval_times.
    """

    # Prepare and sort data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    if eval_times is None:
        eval_times = data[time_col]
    else:
        eval_times = pd.to_datetime(eval_times)

    # Create regular grid
    t_start = data[time_col].min()
    t_end = data[time_col].max()
    regular_time = pd.date_range(start=t_start, end=t_end, freq=inter_freq)
    x_orig = data[time_col].astype('int64')
    y_orig = data[value_col].values
    x_regular = regular_time.astype('int64')

    # Interpolate to regular grid
    if interpolation_method == 'linear':
        interpolated_values = np.interp(x_regular, x_orig, y_orig)
    elif interpolation_method == 'pchip':
        interp_func = PchipInterpolator(x_orig, y_orig, extrapolate=True)
        interpolated_values = interp_func(x_regular)
    else:
        raise ValueError("interpolation_method must be 'linear' or 'pchip'")

    # Build and fit Kalman filter
    # Adjust EM iterations based on temporal resolution
    n_iter = 10 if inter_freq == '1D' else 20  # finer resolution may require more iterations

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=interpolated_values[0],
        n_dim_obs=1,
        n_dim_state=1
    )

    kf = kf.em(interpolated_values, n_iter=n_iter)

    smoothed_state_means, _ = kf.smooth(interpolated_values)
    smoothed = smoothed_state_means.ravel()

    # Interpolate result back to eval_times
    x_eval = eval_times.astype('int64')

    if interpolation_method == 'linear':
        smoothed_evals_1d = np.interp(x_eval, x_regular, smoothed)
    else:
        interp_func = PchipInterpolator(x_regular, smoothed, extrapolate=True)
        smoothed_evals_1d = interp_func(x_eval)

    # Package output
    smoothed_evals = np.expand_dims(smoothed_evals_1d, axis=0)  # shape: (1, len(eval_times))
    bottom = smoothed_evals_1d
    top = smoothed_evals_1d

    return bottom, top, smoothed_evals




"""
Main script body.

MAIN OUTPUTS:

1. df_lake_time_series (DataFrame)
    Stacked LakeSP time series for all evaluated lakes, with the following attributes:
    • Original LakeSP attributes  
    • wse_adjusted: Cycle-adjusted LakeSP WSE values; equal to the original WSE when no adjustment is applied  
    • gauge_datetime: The closest timestamp in the gauge data within 24 hours of each time in LakeSP, if available
    • gauge_wse: The gauge WSE value corresponding to gauge_datetime, if available  
    • filter_flag: Flag indicating the results of our customized filter. 
        - 1 indicates a retained (good-quality) observation
        - 0 indicates an outlier removed by the filter  

2. df_lake_heuristic_thresholds (DataFrame)
    Heuristic thresholds for SP filtering (for step 1: defining the heuristic baselines),
    calibrated for each lake, pass_id, and data version (crid) grouping.
    
    Each row represents a unique combination of:
    - lake_id
    - crid_scenario (version group)
    - pass_id (SWOT orbit pass)

    Attributes:
    • lake_id: PLD lake_id
    • crid_scenario: Scenario grouping based on data version (CRID).Two possible values:
         - "PIC2_or_PID0": newer versions (e.g., PIC2, PID0)
         - "early_versions": older versions (e.g., PIC0, PGC0)
    • pass_id: Integer ID of the SWOT orbit pass
    • wse_std_threshold: Heuristically calibrated upper threshold for the standard
                         deviation of WSE (wse_std), used to exclude noisy or
                         unstable measurements. Typically computed as a max value
                         for that pass and version grouping, capped between
                         a defined min and max if defined.
    • wse_u_threshold: Heuristically calibrated upper threshold for the uncertainty
                       of WSE (wse_u), used similarly to exclude unreliable
                       observations. Also computed per lake, pass, and version group.

3. df_lake_stats (DataFrame)
    Summary statistics for each lake, including validation metrics and retention rates:

    • lake_id: Lake PLD identifier  

    Comparison of custom-filtered and gauge-based WSE time series:
    • rmse: Root Mean Square Error between filtered LakeSP and gauge WSEs  
    • spearman: Spearman’s rank correlation between filtered LakeSP and gauge WSEs  
    • var_swot: Variability (in standard deviation) of WSE in the filtered LakeSP time series  
    • var_gauge: Variability of gauge WSEs corresponding to timestamps in the filtered LakeSP series 

    Comparison between custom-filtered and gauge-based daily-resolution WSE hydrographs
    • var_swot_daily: Variability of daily WSE interpolated from the filtered LakeSP time series  
    • var_gauge_daily:Variability of daily WSE from the full gauge time series 
    
    Comparison of CNES-baseline1[/2]-filtered and gauge-based WSE time series:
    • rmse_baseline1[/2], following the same definition described above       
    • spearman_baseline1[/2]            
    • var_swot_baseline1[/2]            
    • var_gauge_baseline1[/2]              
    • var_swot_daily_baseline1[/2]
    
    Retention metrics of the customized filter compared to CNES baseline filters:
    • retention_n: Number of LakeSP observations retained after applying the customized filter
    • retention_rate: Proportion (float between 0-1) of original LakeSP observations retained  
    • retention_rate_baseline1: Retention using the stringent CNES baseline filter:
        - xovr_cal_q < 1  
        - ice_clim_f < 1  
        - quality_f < 1  
    • retention_rate_baseline2: Retention using a more lenient CNES baseline filter:
        - xovr_cal_q < 2  
        - ice_clim_f < 2  
        - quality_f < 1 and crid not in ["PIC2", "PID0"], i.e., early SP versions.  
        - quality_f < 3 and crid in ["PIC2", "PID0"], after adopting the bitwise flag. 
    For all retention metrics, NaN means no SWOT observations at all.   
     
    Evaluation of filter-induced improvements using unfiltered (original)) LakeSP data:
    Definition consistent with metrics above
    • rmse_unfiltered, spearman_unfiltered  
    • var_swot_unfiltered, var_gauge_unfiltered  
    • var_swot_daily_unfiltered  
    
    Filter iteration times
    • n_while: Number of round-1 iterations
    • n_while_r2: Number of round-2 interactions. For both numbers: 
        - NaN indicates this lake cannot be run due to no SWOT observations at all
        - -1 indicates this lake is abandoned (not meeting criteria)
        - other integers indicating iteration times (0 if apply_low_pass_filter = 'no')
        
    Ice phenology
    'ice_duration': the number of records where ice_clim_f == 2 as a proportion of the full time series  
"""
# Initialize the three major outputs (described above) as empty dataframe
df_lake_time_series = pd.DataFrame()
df_lake_heuristic_thresholds = pd.DataFrame()
df_lake_stats = pd.DataFrame()

df_Hydrocron = pd.DataFrame() #Optional if we want to save the Hydrocron lake time series to disk. 

# Define fill values depending on variable type. 
fill_text = 'no_data'
fill_float = -999999999999

# Retrieve the LakeSP time series for all lakes if the preferred retrieval method is set to be 'on-premise'
if SP_retrieval_method == 'on-premise': 
    df_Hydrocron = pd.read_csv(work_dir+'/df_Hydrocron.csv')

# Loop through each unique test lake
for feature_id in test_lakeIDs: # test_lakeIDs contain a list of unique PLD lake IDs. 
    print('lake ID: ' + str(feature_id))
    
    # Retrieve metedata for this lake
    if feature_id in test_lakes['lake_id'].values: # Check if feature_id exists in the test_lakes dataframe
        # In case a PLD lake ID is redundant among different gauge sources, we will prefer the first gauge source (see below) 
        gauge_source = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_source'].values[0]
        gauge_id = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_id'].values[0]
        gauge_datum = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_datum'].values[0]
        gauge_dir = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_dir'].values[0]
    else: 
        gauge_source = None #string
        gauge_id = None #string
        gauge_datum = np.nan #numeric 
        gauge_dir = None
    
    # Retrieve LakeSP time series based on the preferred method
    if SP_retrieval_method == 'Hydrocron': # from Hydrocron directly. 
        # Read LakeSP data from Hydrocron
        feature = "PriorLake"
        output =  "csv" #"geojson"    
        fields  = 'lake_id,obs_id,overlap,n_overlap,time,time_str,wse,wse_u,wse_r_u,wse_std,area_total,area_tot_u,area_detct,area_det_u,layovr_val,xtrk_dist,quality_f,dark_frac,ice_clim_f,partial_f,xovr_cal_q,geoid_hght,solid_tide,load_tidef,load_tideg,pole_tide,dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,p_ref_area,crid,cycle_id,pass_id'
        enquiry_input =  "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=" + \
                        feature + "&feature_id=" + str(feature_id) + "&start_time=" + start_time + "&end_time=" + end_time + "&output=" + output + "&fields=" + fields
        hydrocron_response = requests.get(enquiry_input).json()
        extracted_data = hydrocron_response['results'][output]
        df = pd.read_csv(StringIO(extracted_data)) 
        ## Optional: if we want to save the Hydrocron time series. 
        #df_Hydrocron = pd.concat([df_Hydrocron, df], ignore_index=True) #ignore_index=True resets the index.
    else: # on-premise: from the file previously saved in the local disk
        df = df_Hydrocron[df_Hydrocron["lake_id"] == feature_id] # Filter for lake_id based on the current feature_id
    
    # Pre-process the LakeSP data
    df = df.loc[df.time_str != fill_text] # Drop measurements where time_str is no_data.     
    df = df.sort_values('time_str') # Sort df by time, to be on the safe side. 
    df.wse = df.wse.mask(df.wse == fill_float, np.nan) # Replace values in the wse column, where value = fill, with nan.
    df.wse_u = df.wse_u.mask(df.wse_u == fill_float, np.nan) # Replace values in the wse_u column, where value = fill, with nan.
    
    ## Duplicate 'time_str' (e.g., 2023-07-30T21:39:18Z) to a new coliumn datetime in datetime64 format (2023-07-30 21:39:18). 
    df['datetime'] = pd.to_datetime(df['time_str'], format='%Y-%m-%dT%H:%M:%SZ') # Use datetime for computation 
    df['index_col'] = range(0, len(df)) # Add an index column to later label the identified outliers. 
    df['filter_flag'] = 1 # Initialize a filter flag, which will be later updated: 1 = good, 0 = outlier
    
    # Read gauge measurements
    if gauge_source == 'CEHQ': # Quebec lakes
        # Retrieve gauge measurements for each lake from the CEHQ repository.
        url_prefix = "https://www.cehq.gouv.qc.ca/depot/historique_donnees_instantanees/0"+gauge_id
        response = requests.get(url_prefix+"_N_2023.txt", verify=False)  # Warning: disables SSL verification!
        df_2023 = pd.read_csv(StringIO(response.text), skiprows=17, sep='\s+', encoding='iso-8859-2')
        response = requests.get(url_prefix+"_N_2024.txt", verify=False)
        df_2024 = pd.read_csv(StringIO(response.text), skiprows=17, sep='\s+', encoding='iso-8859-2')   
        response = requests.get(url_prefix+"_N_2025.txt", verify=False)
        df_2025 = pd.read_csv(StringIO(response.text), skiprows=17, sep='\s+', encoding='iso-8859-2')
        OBS = []
        OBS.extend(df_2023['(m)'])
        OBS.extend(df_2024['(m)'])
        OBS.extend(df_2025['(m)'])
        OBS = pd.to_numeric(OBS, errors='coerce')
        d_2023 = pd.to_datetime(df_2023['Date'] + ' ' + df_2023['Niveau']) # Niveau means level
        d_2024 = pd.to_datetime(df_2024['Date'] + ' ' + df_2024['Niveau'])
        d_2025 = pd.to_datetime(df_2025['Date'] + ' ' + df_2025['Niveau'])
        d = pd.concat([d_2023, d_2024, d_2025])

        # Adjustment from Eastern Standard Time to UTC: simply subtract 5 hours from the data
        d_utc = d - + pd.Timedelta(hours=5)        
        OBS_2024 =  OBS + gauge_datum
        
        gauge_df = pd.DataFrame({
            'gauge_datetime': pd.to_datetime(d_utc),
            'gauge_wse': OBS_2024
            })  
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
    if gauge_source == 'Tefe':
        gauge_df = pd.read_excel(gauge_dir)
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.      
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
    if gauge_source == 'China':            
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["PLD_Lake_ID"] == feature_id] # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df = pd.DataFrame({
            "gauge_datetime": pd.to_datetime(gauge_df[["Year", "Month", "Day", "Hour", "Minute", "Second"]]),
            "gauge_wse": gauge_df["WSE/m"]
            }) # Convert to gauge_df with required column format
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
    if gauge_source == 'NA':                           
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["lake_id"] == feature_id] # Filter for lake_id based on the current feature_id
        gauge_df["gauge_datetime"] = pd.to_datetime(gauge_df["gage_time"], format='%Y-%m-%dT%H:%M:%SZ') # Convert time
        # Find the gage_id with the longest time span for this lake (there could be multiple gauges for the same lake)
        def time_range(gdf):
            return (gdf["gauge_datetime"].max() - gdf["gauge_datetime"].min()).days
        longest_gage_id = (
            gauge_df.groupby("gage_id")
            .apply(time_range)
            .sort_values(ascending=False)
            .idxmax()
            )
        # Filter for best gage_id and format output
        gauge_df = gauge_df[gauge_df["gage_id"] == longest_gage_id][["gauge_datetime", "gage_stage_m", "gage_id"]]
        gauge_df = gauge_df.rename(columns={"gage_stage_m": "gauge_wse"})
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
    if gauge_source == 'West_Africa':            
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["PLD_Lake_ID"] == feature_id] # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
       
    # Duplicate df to df_eval - which will later become the subset of good LakeSP observations after filtering
    df_eval = df.dropna(subset=['wse']) # Initialize df_eval by dropping nan wse values in df.  
    df_eval = df_eval.drop_duplicates(subset='time', keep='first') # Drop duplicates in time    
    df_eval = df_eval.drop_duplicates(subset='datetime', keep='first') # Drop duplicates in datetime. This may happens as 'datetime' is less precise than 'time'.      
    
    
    """    
    Compute heuristic quality thresholds based on long-term lake statistics
    Logic: 
        - 1. Rather than directly using the LakeSP summary quality flags for filtering, we leverage these flags to identify high-quality observations. 
        - 2. From these selected observations, we calibrate heuristic maximum thresholds for two key metrics:
            • wse_std_threshold: Represents the minimum acceptable surface water level consistency across the lake.
            • wse_u_threshold: Represents the maximum acceptable uncertainty from data processing.
        - 3. Then, these thresholds are used to retrieve a heuristic baseline that consists of the observations satifying the thresholds.  
    Notes:
        - dark_frac is not used here, as it is not directly involved in WSE computation.
        - The bitwise quality flag is not used for now because it is only available for PIC2 and PID0 data (since October 2024).
        - However, we utilize the updated quality_f classification (values 0–3) available in PIC2 and PID0 to define good observations.
        - The heuristic thresholds may need to be independent of both data version and orbit pass. As data quality improves over time, 
            thresholds calibrated for earlier data versions may no longer be suitable for newer versions. 
            Similarly, for lakes that span multiple passes, varying factors such as surrounding terrain can affect the measurements 
            differently across passes, making a single threshold inadequate. 
            To accommodate these variations, we have designed the function "calibrate_heuristic_thresholds" with built-in flexibility. 
            Please refer to the function above for further details.
    """
    # Define df_conservative as the subset of high-quality LakeSP observations, from which wse_std_threshold and wse_u_threshold are calibrated.
    # Selection criteria are broadly aligned with CNES baseline standards (Claire Pottier and Roger Fjørtoft, on behalf of the HR Cal/Val Team, SWOT ST Meeting 2024).
    # quality_f:
    #     - Not used for LakeSP versions prior to PIC2, as only values 0 (good) and 1 (bad) are present and tend to over-reject observations.
    #     - Used for PIC2 and PID0: observations flagged as 2 (degraded) or 3 (bad) are excluded.
    # xovr_cal_q:
    #     - Only good observations (0) are retained. 
    # ice_clim_f:
    #     - Observations with full ice cover (2) are excluded.
    #     - Observations with partial ice cover (1) are retained to avoid under-detection.
    df_conservative = df_eval.query('xovr_cal_q < 1 & ice_clim_f < 2 & quality_f < 2') # Preferred. Use "<" to retain fill values.
    #df_conservative = df_eval.query('xovr_cal_q < 1 & ice_clim_f < 2')
    # Note: Based on our experimentation, including quality_f or not does not seem to affect threshold calibration too much. 
    # In comparison, the standard of xovr_cal_q affects the thresholds more. 
    
    # Calibrate heuristic thresholds: wse_std_threshold and wse_u_threshold
    # [1] Set up default upper caps for the thresholds. 
    # !!! Note: These upper caps may lack strong theoretical justification and may require future revision.!!!
    # For reference, a wse_std_threshold exceeding 3-5 m is unlikely:
    #     - Lake Tefé, a ria lake with strong water-surface gradient during the dry season, shows a maximum 4–5 m WSE range.
    #     - Superelevation at Lake Selingué is typically <3 m.
    wse_std_threshold_max = 3 #3 or 5, but 3 seems more conservative and more accurate, but 5 m may allow more observations in.
    wse_std_threshold_min = 0 # set up to 0 for now. 
    wse_u_threshold_max = 0.5 #0.5 m is probably a bit too conservative. Needs optimization using validation data.
    wse_u_threshold_min = 0.1 #or 0. The value of 0.1 m overall consistent with the science requirement.
    # Note: not capping wse_u_threshold_min may lead to over-filtering. 
    
    # [2] Compute heuristic thresholds, which can be cutomized to be pass- and/or version- independent. 
    # Recommendation: do not apply any of the two groupings (keep both False) to avoid over-rejection. 
    df_heuristic_thresholds = calibrate_heuristic_thresholds(df_conservative, 
                                     by_crid_scenario=False, #Caution: this may lead to over-rejection as new version data is still limited.
                                     by_pass_id=False, #True is probably not necessary, and most lakes only have one pass ID. 
                                     wse_std_threshold_max=wse_std_threshold_max, wse_std_threshold_min=wse_std_threshold_min,
                                     wse_u_threshold_max=wse_u_threshold_max, wse_u_threshold_min=wse_u_threshold_min)
    # Note: output df_heuristic_thresholds contains: ['crid_scenario', 'pass_id', 'wse_std_threshold', 'wse_u_threshold'], and 
    #       crid_scenario has two scenarios: "PIC2_or_PID0" and "early_versions" (e.g., PIC0, PGC0).   
    print(df_heuristic_thresholds)
    
    # Check if we would like to execute both baseline filtering (Step 1) and low-pass filtering (Step 2) or just Step 1
    if apply_low_pass_filter == 'no': # Execute only the heuristic baseline filtering (Step 1)
        # Apply heuristic thresholds to generate the heuristic baseline. 
        df_eval = apply_heuristic_thresholds(df_eval, df_heuristic_thresholds, wse_std_threshold_max = wse_std_threshold_max)
        # Note: in the built-in function, wse_std threshold for freeze-up/ice-covered period is relaxed to increase data availability.
        # Based on our testing, this more lenient condition for ice-covered periods seems necessary. 
    else: # execute both steps     
        # If preferred, first constrain df_eval to the heuristic baseline before executing the low-pass filtering. 
        if evaluating_at_full_data == 'no': # Evaluate outlier removal (z-score clipping) only on the selected heuristic baseline. 
            # Apply heuristic thresholds to generate the heuristic baseline. 
            df_eval = apply_heuristic_thresholds(df_eval, df_heuristic_thresholds, wse_std_threshold_max = wse_std_threshold_max)
        # Otherwise, if evaluating_at_full_data == 'yes', evaluate z-score cliping on the full df_eval data.  
    #plt.plot(df_eval.datetime, df_eval.wse, 'x')
    
    """
    Start round-1 (mandatory) low-pass filtering: results will be stored in df_eval (a selected subset of LakeSP after filtering)
    To avoid confusion: 
        "Filter application" refers to applying the chosen filter method to generate a smoothing curve. This is done on df_apply. 
        "Filter evaluation" refers to using the smoothing curve as a benchmark for z-score clipping to noise removal. This is done on df_eval. 
    This "while" loop:
        - Starts by selecting high-quality LakeSP data (df_apply) from df_eval for filter application (i.e., generating smoothing curve)        
        - Iteratively:
            • Applies the selected filter to df_apply to generate a smoothing curve
            • Evaluates the filter (z-score clipping) on df_eval using the smoothing curve
            • Updates df_eval by removing identified outliers from df_eval
        - Stops when one of the following is met:
            • The maximum residual spread (lim) is sufficiently small (this argument is embedded in the while loop)
            • No additional outliers are removed (i.e., updated_length == initial_length)
            • The loop has run 40 times (empirically sufficient for convergence)
            • The time series for either filter application or evaluation is too short or limited in temporal range. 
    """     
    initial_length = len(df_eval) # In case this is zero, the "while" statement won't run. 
    updated_length = 0 # Initialize the length of the updated df_eval
    n_while = 0  # Initialize the loop/iteration times    
    lowess_QA_check = 'check' # Initialize a QA check for the lowess filter. This is only relevant if filter_type is set to 'lowess'.
    while apply_low_pass_filter == 'yes' and (updated_length < initial_length) and (n_while < 40): # All conditions must be satisfied
        initial_length = len(df_eval) # Note: df_eval is updated per iteration. 
               
        # Apply heuristic thresholds to generate the "heuristic baseline" (i.e., good-quality observations for filter application). 
        df_apply = apply_heuristic_thresholds(df_eval, df_heuristic_thresholds, wse_std_threshold_max = wse_std_threshold_max)
        # Remove bad crossover calibration, although this is redundant for PIC2 and PID0 as quality_f < 3 precludes xovr_cal_q = 2 (see bitwise definition)
        df_apply = df_apply[df_apply['xovr_cal_q'] < 2]  
        # Remove bad observations flagged in PIC2 and PID0: specular_rining_bad, xovr_cal_bad, and low_coh_bad. 
        df_apply = df_apply[df_apply['quality_f'] < 3] 
        # Note: quality_f is not considered prior to PIC2, as only values 0 (good) and 1 (bad) are present and tend to over-reject observations.
        
        # Optional: Further remove positive anomalies during ice-covered period. 
        # Recommendation: do not apply the ice filter yet to avoid under-fitting. 
        # It is highly recommended to set "by_pass" to be True because area_total (which can be pass dependent) is used to detect outliers.
        #df_apply = filter_ice_outliers(df_apply, remove_tukey_outliers, by_pass=True, by_crid_scenario=False,
        #                        multiplier=0.2, lower_q=0, upper_q=1, used_q='upper')
        #plt.plot(df_apply.datetime, df_apply.wse, 'x')
        
        ## Set display options to show full DataFrame
        #pd.set_option('display.max_rows', None)       # Show all rows
        #pd.set_option('display.max_columns', None)    # Show all columns
        #pd.set_option('display.width', None)          # Do not limit line width
        #pd.set_option('display.max_colwidth', None)   # Show full column content
        ## Show the full DataFrame
        #print(df_apply)
        
        # This lake should be abandoned if any of the two scenarios occurs any any time:
        # 1. The time series for filtering application (df_apply) has major temporal gaps (> 3 months, a season).
        # 2. The time range of df_apply is too short (<80% of the original range in df). If so, it would lead to either  
        #    substantial extrapolation during filter evaluation or resulting filter result not representing full time seires            
        exceeds_limit = (df_apply['datetime'].diff()) > pd.Timedelta(days=92) # Check if any time difference exceeds 3 months
        if exceeds_limit.any() or len(df_apply) <=1: # If any gap exceeds 3 months or if there are less than two records, abandon this lake
            df_eval = df_eval.iloc[0:0] # Clear up de_eval  
            n_while = -1 # -1 indicates this lake is abandoned.             
            break # break the while loop
        else:      
            # Check if the time range of df_apply is too short (<80% of the time range of df)
            if (df_apply['time'].max() - df_apply['time'].min()) / (df['time'].max() - df['time'].min()) < 0.8: #If so, abondan this lake. 
                df_eval = df_eval.iloc[0:0] # Clear up de_eval   
                n_while = -1 # -1 indicates this lake is abandoned.
                break # break the while loop      
        
        # Apply the chosen filter (filter_type)       
        if 'lowess' in filter_type:     
            # Determine the minfrac parameter based on the time series (df_apply) length 
            if lowess_QA_check == 'check': # If the time series does not contain too many high wse_std values (possible outliers)
                if len(df_apply) <= 50:
                    minfrac = 0.15
                elif len(df_apply) < 120:
                    minfrac = 0.05
                else: 
                    minfrac = 0.03  
            # Check the proportion of possible outliers in the time series based on wse_std values. 
            # If the porportion is high, having a very small minfrac may lead to overfitting.
            large_wse_std_proportion = len(df_apply[df_apply['wse_std'] > 2])/len(df_apply)
            if lowess_QA_check == 'check':
                if large_wse_std_proportion >= 0.25:
                    minfrac = 0.15
                    lowess_QA_check = 'no more check' # Freeze minfrac from now on, regardless of remaining iteration
            print('minfrac: ' + str(minfrac) + ' ... series length: ' + str(len(df_apply)) + '... large std proportion: ' + str(large_wse_std_proportion))
            
            bottom, top, filter_curves = filter_lowess(
                df_apply, value_col='wse', time_col='datetime', eval_times=df_eval['datetime'],
                minfrac=minfrac, 
                maxfrac=0.2, 
                frac_step=0.02, 
                it_v=[1,2,3,4], 
                n_jobs=-1) #No need to interpolate unequal time
            residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
            
        if 'wavelet' in filter_type:                     
            bottom, top, filter_curves = filter_wavelet(
                df_apply, value_col='wse', time_col='datetime', eval_times=df_eval['datetime'],
                wavelet_v=['db4'], #['db4', 'sym2', 'coif1']. db4 is a general purpose wavelet; sym2 and coif1 returns smoother result
                level=None,
                threshold=None,
                inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                interpolation_method='linear', #'linear' or 'pchip'
                n_jobs=-1
                )
            residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
        
        if 'savgol' in filter_type:
            bottom, top, filter_curves = filter_savgol(
                df_apply, value_col='wse', time_col='datetime', eval_times=df_eval['datetime'],
                window_length_v=[21], #[7, 9, 11, 21, 31, 41, 51], full window widths in days (must be odd integers)
                polyorder_v=[3], #[2,3], 3 outperforms 2. 
                inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                interpolation_method='linear', #'linear' or 'pchip'
                n_jobs=-1
                )
            residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])   
            
        if 'hampel' in filter_type:              
            bottom, top, filter_curves = filter_hampel(
                df_apply, value_col='wse', time_col='datetime', eval_times=df_eval['datetime'],
                window_length_v=[21], #[11, 21, 31], full window widths in days (must be odd integers)
                inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                interpolation_method='linear', #'linear' or 'pchip'
                n_jobs=-1
                )
            residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
            
        if 'spline' in filter_type:
            bottom, top, filter_curves = filter_spline(
                df_apply, value_col='wse', time_col='datetime', eval_times=df_eval['datetime'],
                smoothing_factor_v=[1e6], #[1e5, 1e6, 1e7]
                n_jobs=-1
                ) #No need to interpolate unequal time
            residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
            
        if 'median' in filter_type:
            bottom, top, filter_curves = filter_median(
                df_apply, value_col='wse', time_col='datetime', eval_times=df_eval['datetime'],
                window_length_v=[21], #[11, 21, 31], full window lengths (must be odd integers)
                inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                interpolation_method='linear', #'linear' or 'pchip'
                n_jobs=-1
                )
            residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
        
        if 'kalman' in filter_type:  
            bottom, top, filter_curves = filter_kalman(
                df_apply, value_col='wse', time_col='datetime', eval_times=df_eval['datetime'],
                inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                interpolation_method='linear' #'linear' or 'pchip'
                ) 
            residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
               
        
        # Preserve the evaluated time before df_eval is updated. Time_eval is only used if show_filtering_evolution is set to 'yes'.  
        time_eval = df_eval['datetime'] 
        
        # Compute the z-score
        # Assign residuals to df_eval (the 'residual' attribute will be introduced if for the first time)
        df_eval['residuals'] = residuals        
        if np.nansum(np.abs(residuals)) == 0: # Note sometimes all residuals are 0 due to overfitting.
            z_scores = (residuals - np.nanmean(residuals))/1.0 # Force it to be 0, so there will be no outliers. 
        else: 
            z_scores = (residuals - np.nanmean(residuals))/np.nanstd(residuals)
        
        # Z-score clipping
        # Check whether residuals need to be removed or not based on how spread the residuals are.        
        # This is evaluated by maximum_residual_spread, which computes the maximum residual as a proportion of df_apply range. 
        maximum_residual_spread = np.nanmax(abs(df_eval['residuals']))/ (np.max(df_apply['wse'])  - np.min(df_apply['wse']))
        #print('maximum_residual_spread: ' + str(maximum_residual_spread))
        if maximum_residual_spread > 0.08: #0.1? An emprical threshold; no need to remove data if spread falls below this threshold.     
            df_eval = df_eval[np.abs(z_scores) < z_score_threshold] # Update df_eval by removing outliers for the next iteration. 
                    
        ## -- Abandoned (starts)
        ## Remove positive anomalies due to ice cover. Based on observations, ice cover may lead to high WSE anomalies. 
        #df_eval_ice_free = df_eval[df_eval['ice_clim_f'] <= 1] # Retrieve ice-free observations
        #ice_free_wse_range = np.max(df_eval_ice_free.wse) -  np.min(df_eval_ice_free.wse) # Compute WSE range in ice-free period
        #range_p = 0.2 # An emprical proportion allowing a positive ice anomaly to excced the ice-free range. Lower range_p means more aggressive removal.
        #df_eval = df_eval[ (df_eval.wse - np.max(df_eval_ice_free.wse)) <= ice_free_wse_range*range_p ] # Remove poisitive ice anomalies. 
        ## -- Abondoned (ends)
        
        # To be double-secure, remove positive anomalies during the freeze-up/ice-covered period. 
        # It is highly recommended to set "by_pass" to be True because area_total (which can be pass dependent) is also used to detect outliers.
        df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=True, by_crid_scenario=False,
                                multiplier=0.2, lower_q=0, upper_q=1, used_q='upper')
        
        # Remove remaining isolated extreme outliers using Tukey method (IQR method) 
        # Use 10th and 90th percentile.
        df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse', multiplier=3, lower_q=0.1, upper_q=0.9)
          
        # Plot filter evolution if preferred. Caution: this will generate a series of plot (one per iteration)
        if show_filtering_evolution == 'yes': # Show how the outlier removal evolves through iteraction
            plt.rcParams["font.family"] = "Arial"
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.grid(True, linewidth=0.5, zorder=1)
            
            # Plot gauge measurements if the lake has gauge data
            if gauge_source is not None:
                ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse']-(gauge_df['gauge_wse'].mean()-df['wse'].mean()), \
                        label='gauge', color='green', marker = 'o', markersize=8, linestyle='--') # Include an approximate datum subtraction
            
            # Plot LakeSP observations for smoothing (df_apply)
            ax.errorbar(df_apply['datetime'], df_apply['wse'], yerr=df_apply.wse_u, label='for smoothing', marker='o', \
                        color=(0,1,0), markersize=4, capsize=3, linestyle='') 
            
            # Plot all generated smoothing curves with increasing darkness
            num_lines = filter_curves.shape[0]
            for i in range(num_lines):
                if num_lines == 1:
                    gray_level = 0.2  # fallback gray level when only one line
                else:
                    gray_level = 1.0 - (i / (num_lines - 1)) * 0.8  # from light (0.2) to dark (1.0)
                ax.plot(time_eval, filter_curves[i], linewidth=0.5, color=str(gray_level))         
                
            # Show selected LakeSP observations after filter evaluation (df_eval)
            ax.plot(df_eval['datetime'], df_eval['wse'], label='selected', marker='s', color='orange', linestyle='None') 
            
            # Format x-axis and title
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
            fig.autofmt_xdate()            
            ax.set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))    
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('WSE (m)', fontsize=12)
            ax.set_title('Lake ID ' + str(feature_id) + ' WSE Plot: ' + filter_type)
            ax.legend()            
        
        # Update the length of df_eval (evaluated data after outlier removal)
        updated_length = len(df_eval)
        n_while += 1
    print('n_while used: ' + str(n_while))
    

    
    """
    Optional: data recovery and round-2 filtering
    
    High-quality LakeSP observations may have been unintentionally removed in round 1 above when the filter struggled to eliminate extreme outliers.
    The following section provides an option (i.e., if recovering_observations == "yes"), to reintroduce those removed high-quality observations.
    
    After high-quality observations are reintroduced, it is recommended to run another round (round 2) filtering, which is less aggressive than round 1, 
    to ensure the elimination of very extreme outliers. 
    """
    # Optional: reintroduce/recover high-quality LakeSP observations that were excluded during round-1 filtering. 
    if apply_low_pass_filter == 'yes' and recovering_observations == 'yes':        
        # Initialize df_good_quality as the subset of df containing valid (non-NaN) WSE values
        df_good_quality = df.dropna(subset=['wse'])
        
        # Apply stricter quality control: retain only observations flagged as "good" by CNES baseline quality flags
        df_good_quality = df_good_quality[(df_good_quality['xovr_cal_q'] == 0) & (df_good_quality['quality_f'] == 0) & (df_good_quality['ice_clim_f'] == 0)] 
              
        # Further apply heuristic thresholds (no ice period this time)
        df_good_quality = apply_heuristic_thresholds(df_good_quality, df_heuristic_thresholds, wse_std_threshold_max = wse_std_threshold_max)
   
        # Identify high-quality observations not already present in df_eval based on the index_col
        df_to_recover = df_good_quality[~df_good_quality['index_col'].isin(df_eval['index_col'])]
        
        # Append these recovered observations to df_eval
        df_eval = pd.concat([df_eval, df_to_recover], ignore_index=True)
        
        # Sort df_eval by index_col to maintain chronological order
        df_eval = df_eval.sort_values(by='index_col').reset_index(drop=True)
        # Note the code above is safe even when df_good_quality is empty.
    
    # Optional: run a round-2, less aggressive filtering to eliminate remaining extreme outliers.
    # The logic is consistent with round 1, except that the filter is applied and evaluated on the same data: df_eval.
    n_while_r2 = 0  # Initialize the iteration times for round-2 filtering.
    if apply_low_pass_filter == 'yes' and r2_filter == 'yes':
        initial_length = len(df_eval) # In case this is zero, the "while" statement won't run. 
        updated_length = 0  # Initialize the length of the updated df_eval
            
        while (updated_length < initial_length) and (n_while_r2 < 5): # A max of 5 iteration times to avoid over-rejection in round-2 filtering. 
            initial_length = len(df_eval) # Note: df_eval is updated per iteration.   
            
            # This lake should be abandoned if any of the two scenarios occurs any any time:
            # 1. The time series for filtering application (df_eval) has major temporal gaps (> 3 months, a season).
            # 2. The time range of df_eval is too short (<80% of the original range in df).            
            exceeds_limit = (df_eval['datetime'].diff()) > pd.Timedelta(days=92) # Check if any time difference exceeds 3 months
            if exceeds_limit.any() or len(df_eval) <=1: # If any gap exceeds 3 months or if there are less than two records, abandon this lake
                df_eval = df_eval.iloc[0:0] # Clear up de_eval  
                n_while_r2 = -1 # -1 indicates this lake is abandoned. 
                break # break the while loop
            else:      
                # Check if the time range of df_eval is too short (<80% of the time range of df)
                if (df_eval['time'].max() - df_eval['time'].min()) / (df['time'].max() - df['time'].min()) < 0.8: #If so, abondan this lake. 
                    df_eval = df_eval.iloc[0:0] # Clear up de_eval  
                    n_while_r2 = -1 # -1 indicates this lake is abandoned. 
                    break # break the while loop
                    
            # Apply the chosen filter (filter_type)
            # Note: Different from round 1, eval_times is set to None, meaning that evaluation time is the same as application time.
            if 'lowess' in filter_type:
                bottom, top, filter_curves = filter_lowess(
                    df_eval, value_col='wse', time_col='datetime', eval_times=None,
                    minfrac=0.2, #fixed it to 0.2 for less aggressive filtering
                    maxfrac=0.2, 
                    frac_step=0.02, 
                    it_v=[1,2,3,4], 
                    n_jobs=-1) #No need to interpolate unequal time
                residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
                
            if 'wavelet' in filter_type:                     
                bottom, top, filter_curves = filter_wavelet(
                    df_eval, value_col='wse', time_col='datetime', eval_times=None,
                    wavelet_v=['db4'], #['db4', 'sym2', 'coif1']. db4 is a general purpose wavelet; sym2 and coif1 returns smoother result
                    level=None,
                    threshold=None,
                    inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                    interpolation_method='linear', #'linear' or 'pchip'
                    n_jobs=-1
                    )
                residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
            
            if 'savgol' in filter_type:
                bottom, top, filter_curves = filter_savgol(
                    df_eval, value_col='wse', time_col='datetime', eval_times=None,
                    window_length_v=[21], #[31]?. full window widths in days (must be odd integers)
                    polyorder_v=[3], #[2,3], 3 outperforms 2. 
                    inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                    interpolation_method='linear', #'linear' or 'pchip'
                    n_jobs=-1
                    )
                residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])   
                
            if 'hampel' in filter_type:              
                bottom, top, filter_curves = filter_hampel(
                    df_eval, value_col='wse', time_col='datetime', eval_times=None,
                    window_length_v=[21], #[11, 21, 31], full window widths in days (must be odd integers)
                    inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                    interpolation_method='linear', #'linear' or 'pchip'
                    n_jobs=-1
                    )
                residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
                
            if 'spline' in filter_type:
                bottom, top, filter_curves = filter_spline(
                    df_eval, value_col='wse', time_col='datetime', eval_times=None,
                    smoothing_factor_v=[1e6], #[1e5, 1e6, 1e7]
                    n_jobs=-1
                    ) #No need to interpolate unequal time
                residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
                
            if 'median' in filter_type:
                bottom, top, filter_curves = filter_median(
                    df_eval, value_col='wse', time_col='datetime', eval_times=None,
                    window_length_v=[21], #[11, 21, 31], full window lengths (must be odd integers)
                    inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                    interpolation_method='linear', #'linear' or 'pchip'
                    n_jobs=-1
                    )
                residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])
            
            if 'kalman' in filter_type:  
                bottom, top, filter_curves = filter_kalman(
                    df_eval, value_col='wse', time_col='datetime', eval_times=None,
                    inter_freq='1D', #'1D' (daily) or '1H' (hourly) regular grid
                    interpolation_method='linear' #'linear' or 'pchip'
                    ) 
                residuals = signed_min_abs_residual(filter_curves, df_eval['wse'])

            
            # Preserve the original df_eval before it is modified. This is only used if show_filtering_evolution is set to 'yes'.  
            df_eval_original = df_eval.copy()
            
            # Compute the z-score
            # Assign residuals to df_eval (the 'residual' attribute will be introduced if for the first time)
            df_eval['residuals'] = residuals        
            if np.nansum(np.abs(residuals)) == 0: # Note sometimes all residuals are 0 due to overfitting.
                z_scores = (residuals - np.nanmean(residuals))/1.0 # Force it to be 0, so there will be no outliers. 
            else: 
                z_scores = (residuals - np.nanmean(residuals))/np.nanstd(residuals)
            
            # Z-score clipping
            # Check whether residuals need to be removed or not based on how spread the residuals are.        
            # This is evaluated by maximum_residual_spread, which computes the maximum residual as a proportion of df_eval range. 
            maximum_residual_spread = np.nanmax(abs(df_eval['residuals']))/ (np.max(df_eval['wse'])  - np.min(df_eval['wse']))
            #print('maximum_residual_spread: ' + str(maximum_residual_spread))
            # Note: maximum_residual_spread threshold and z_scores_threshold_r2 are both more lenient than those for round 1. 
            if maximum_residual_spread > 0.08: # An emprical threshold; no need to remove data if spread falls below this threshold.     
                df_eval = df_eval[np.abs(z_scores) < z_scores_threshold_r2] # Update df_eval by removing outliers for the next iteration. 
            
            ## -- Abandoned (starts)
            ## Remove positive anomalies due to ice cover. Based on observations, ice cover may lead to high WSE anomalies. 
            #df_eval_ice_free = df_eval[df_eval['ice_clim_f'] <= 1] # Retrieve ice-free observations
            #ice_free_wse_range = np.max(df_eval_ice_free.wse) -  np.min(df_eval_ice_free.wse) # Compute WSE range in ice-free period
            #range_p = 0.2 # An emprical proportion allowing a positive ice anomaly to excced the ice-free range. Lower range_p means more aggressive removal.
            #df_eval = df_eval[ (df_eval.wse - np.max(df_eval_ice_free.wse)) <= ice_free_wse_range*range_p ] # Remove poisitive ice anomalies. 
            ## -- Abandoned (ends)
            
            # To be double-secure, remove positive anomalies during ice-covered period again. 
            # It is highly recommended to set "by_pass" to be True because area_total (which can be pass dependent) is also used to detect outliers.
            df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=True, by_crid_scenario=False,
                                    multiplier=0.2, lower_q=0, upper_q=1, used_q='upper')
                        
            # Remove remaining isolated outliers using Tukey method (IQR method)    
            # Use 10th and 90th percentile.
            df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse', multiplier=3, lower_q=0.1, upper_q=0.9)
              
            # Plot filter evolution if preferred. Caution: this will generate a series of plot (one per iteration)
            if show_filtering_evolution == 'yes': # Show how the outlier removal evolves through iteraction
                plt.rcParams["font.family"] = "Arial"
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.grid(True, linewidth=0.5, zorder=1)
                
                # Plot gauge measurements if the lake has gauge data
                if gauge_source is not None:
                    ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse']-(gauge_df['gauge_wse'].mean()-df['wse'].mean()), \
                            label='gauge', color='green', marker = 'o', markersize=8, linestyle='--') # Include an approximate datum adjustment
                
                # Plot LakeSP observations for smoothing (df_eval_original)
                ax.errorbar(df_eval_original['datetime'], df_eval_original['wse'], yerr=df_eval_original.wse_u, \
                            label='for smoothing', marker='o', color=(0,1,0),  markersize=4, capsize=3, linestyle='') 
                
                # Plot all generated smoothing curves with increasing darkness
                num_lines = filter_curves.shape[0]
                for i in range(num_lines):
                    if num_lines == 1:
                        gray_level = 0.2  # fallback gray level when only one line
                    else:
                        gray_level = 1.0 - (i / (num_lines - 1)) * 0.8  # from light (0.2) to dark (1.0)
                    ax.plot(df_eval_original.datetime, filter_curves[i], linewidth=0.5, color=str(gray_level))         
                    
                # Show selected LakeSP observations after filter evaluation (df_eval)
                ax.plot(df_eval['datetime'], df_eval['wse'], label='selected', marker='s', color='orange', linestyle='None') 
                
                # Format x-axis and title
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
                fig.autofmt_xdate()            
                ax.set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))    
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('WSE (m)', fontsize=12)
                ax.set_title('Lake ID ' + str(feature_id) + ' WSE Plot: ' + filter_type + ' (round 2)')
                ax.legend()            
            
            # Update the length of df_eval (evaluated data after outlier removal)
            updated_length = len(df_eval)            
            n_while_r2 += 1     
    
    # Post-processing by removing LakeSP observations that are still 100 m higher than the mean WSE
    # Typical range for large reservoirs: 10–60 meters (e.g., the Three Gorges Reservoir ranges in 145-175 m). 
    # Very large reservoirs (e.g., hydropower or multipurpose dams): can exceed 100 meters
    # A few massive reservoirs may approach or even exceed 150–200 meters in water level fluctuation.
    diff_average = np.abs(df_eval['wse'] - np.mean(df_eval['wse']))
    df_eval = df_eval[diff_average <= 100]
    # Note: this works if df_eval is empty.     
    
    """
    Cycle adjustment: To reduce intra-cycle WSE inconsistencies caused by multiple orbit passes
    For large lakes spanning multiple SWOT orbit passes, WSE values within the same orbit cycle may show substantial 
    inconsistencies (e.g., zig-zag patterns) across different passes. 
    
    The following three options are provided to mitigate this issue:
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
    # Duplicate "wse" values to a new column "wse_adjusted" in df_eval (results after filtering). 
    # If cycle-adjustment is needed, wse_adjusted will be updated to be the cycle-adjusted WSEs for option 3.  
    # Otherwise, wse_adjusted will remain a duplicate of wse. 
    df_eval['wse_adjusted'] = df_eval['wse']
    
    # Option 1: Cycle-averaged WSE time series. Note that cycle_id will be sorted in ascending order.   
    df_cycle_avg = df_eval.groupby('cycle_id')['wse'].mean().rename('wse_cycle_avg').reset_index()     
    # Compute the middle observation date per cycle
    cycle_dates = df_eval.groupby('cycle_id')['datetime'].median().rename('mid_date').reset_index()
    # Merge with df_cycle_avg based on cycle_id. Merged dataframe contains mid_date and wse_cycle_avg columns
    df_option1 = pd.merge(df_cycle_avg, cycle_dates, on='cycle_id')
        
    # Compare intra-cycle vs inter-cycle WSE variability
    intra_cycle_std = df_eval.groupby('cycle_id')['wse'].std().median() # Computed as the median of cycle-level WSE standard deviations. 
    inter_cycle_std = df_option1['wse_cycle_avg'].std() # Computed as the standard devaition of cycle-averaged WSEs
    
    # Check if options 2 and 3 cycle adjustment is needed: intra-cycle variability must exceed inter-cycle variability
    cycle_adjustment = 'no' # Initialize an indicator of whether cycle-adjustment is needed or not. By defaut, it is set to "no". 
    if apply_low_pass_filter == 'yes' and intra_cycle_std > inter_cycle_std:
        cycle_adjustment = 'yes' # This will be used to determine if plotting is needed.  
                       
        # Option 2: Retain only observations from the pass that captures the largest observed lake area (area_total).        
        # For each pass_id, compute the median lake area (area_total) observed across all cycles.
        median_pass_areas = df_eval.groupby('pass_id')['area_total'].median().reset_index()
        # Find the row index of the pass that has the largest median lake area, and retrieve the corresponding pass_id for that row.        
        # best_pass_id is the pass that most consistently observes the largest observed portion of the lake. 
        best_pass_id = median_pass_areas.loc[median_pass_areas['area_total'].idxmax(), 'pass_id']

        # Filter the original df_eval to keep only observations associated with best_pass_id
        df_option2 = df_eval[df_eval['pass_id'] == best_pass_id].sort_values('cycle_id')      
                
        # Option 3: Adjust each WSE value by removing its pass-specific bias relative to the overall WSE average.
        # This helps reduce zig-zag patterns caused by systematic offsets between passes.
        # Compute the deviation ("departure") of each WSE value from the overall mean across the time series
        df_eval['departure'] = df_eval['wse'] - df_eval['wse'].mean()
        # For each pass, compute the median departure, which represents the expected bias of that pass
        pass_median_departure = df_eval.groupby('pass_id')['departure'].median()

        # Map each observation to its corresponding pass-level median departure (pass_median_departure)
        df_eval['pass_median_departure'] = df_eval['pass_id'].map(pass_median_departure)

        # Subtract the pass-specific bias from each WSE value to obtain the bias-corrected WSE
        # df_eval.wse_adjusted represents the cycle-adjusted WSEs for Option 3!
        df_eval['wse_adjusted'] = df_eval['wse'] - df_eval['pass_median_departure']
        
        # Run another tukey outlier removal on df_eval.wse_adjusted
        df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse_adjusted', multiplier=3, lower_q=0.1, upper_q=0.9)
    
    #Label survivals (non-outlier LakeSP observations) back to the original df through index_col 
    #     assign df.filter_flag to be 0 when df.index_col goes beyond df_eval.index_col (i.e., outliers)
    #     filter_flag: 1 means good; 0 means outlier
    df.loc[~df['index_col'].isin(df_eval['index_col']), 'filter_flag'] = 0
    # Note: df.query('filter_flag != 0') will be the final original LakeSP observations that survived the filtering!
    # It equals df_eval in size, but df.query('filter_flag != 0') keeps the original attribute structure. 
    
    # Left-join the 'wse_adjusted' column from df_eval into df, based on the unique key index_col.
    df = df.merge(
        df_eval[['index_col', 'wse_adjusted']],  # only bring the column(s) that are shared (retained after filtering)
        on='index_col',
        how='left'  # keep all rows from df, fill unmatched ones with NaN
        ) # Now df will have a new column "wse_adjusted". This also works if df_eval is empty. 
    # Note again:
    #    wse_adjusted is only valid for filtered results (good observations)
    #    If no wse_adjusted was assigned (i.e., outliers, not in df_eval), the value of df.wse_adjusted will be left nan. 
    #    wse_adjusted will equal wse if no cycle adjustment is needed. 
    #    So it is safe to just use wse_adjusted for representing filtered results. 
  
    
               
    """
    Plot filtered time series and compute statistics for this lake
    """        
    # Define CNES baselines
    # Stringent CNES baseline (baseline 1)
    CNES_baseline1 = 'xovr_cal_q < 1 & ice_clim_f < 1 & quality_f < 1'
    # More lenient CNES baseline (baseline 2)
    CNES_baseline2 = ('xovr_cal_q < 2 & ice_clim_f < 2 & '
                  '((quality_f < 1 & (crid != "PIC2") & (crid != "PID0")) '
                  '| (quality_f < 3 & ((crid == "PIC2") | (crid == "PID0"))))')
    
    # For simplicity, add two wse columns (wse_baseline1, and wse_baseline2) to df. 
    # They duplicate the original WSE values at the CNES_baseline1 and CNES_baseline 2 indices, respectively, 
    # and np.nan for other indices.     
    # Evaluate masks
    mask_baseline1 = df.eval(CNES_baseline1)
    mask_baseline2 = df.eval(CNES_baseline2)
    # Create new columns
    df['wse_baseline1'] = np.where(mask_baseline1, df['wse'], np.nan)
    df['wse_baseline2'] = np.where(mask_baseline2, df['wse'], np.nan)
       
    # Set up this final plot. 
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, linewidth=0.5, zorder=1)
    
    # Compute statistics for this lake
    if gauge_source is not None: # If this lake has gauge data (this works if df_eval is empty)
        # Plot the gauge time series
        ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse']-(gauge_df['gauge_wse'].mean()-df['wse'].mean()), \
                label='gauge', color='green', marker = 'o', markersize=8, linestyle='--') # Include an approximate datum adjustment
        
        # Join gauge_df's gauge_datetime and gauge_wse values into SWOT's df (using datetime in df)
        # by finding the closest timestamp in gauge_df within 24 hours of each datetime.
        # If multiple gauge times are within 24 hours, the closest one is used.        
        # Note both gauge_df.gauge_datetime and df.datetime have already been formated to datetime64 format: e.g., using pd.to_datetime(df['datetime'])
        # Perform nearest-time join within 24-hour tolerance
        df = pd.merge_asof(
            df,
            gauge_df[['gauge_datetime', 'gauge_wse']],
            left_on='datetime', #in SWOT
            right_on='gauge_datetime', #in gauge
            direction='nearest',
            tolerance=pd.Timedelta('24h')
            ) # Note: this will generate two extra attributes (gauge_datatime and gauge_wse) in df, if the lake has gauge data.        
  
        # Compute RMSE
        # Select records where gauge_wse is not nan. 
        subset = df[df['gauge_wse'].notna()]
        # Use mean-adjusted anomalies to avoid datum discrepancies
        wse_adjusted_mean = np.nanmean(subset['wse_adjusted'])
        gauge_mean = np.nanmean(subset['gauge_wse'])
        rmse_unfiltered = compute_rmse(df['wse']-wse_adjusted_mean, df['gauge_wse']-gauge_mean)
        # Using only filtered (retained good observation) data with filter_flag == 1:
        rmse = compute_rmse(df['wse_adjusted']-wse_adjusted_mean, df['gauge_wse']-gauge_mean)
        # Using CNES baseline 1:
        rmse_baseline1 = compute_rmse(df['wse_baseline1']-wse_adjusted_mean, df['gauge_wse']-gauge_mean)
        # Using CNES baseline 2:
        rmse_baseline2 = compute_rmse(df['wse_baseline2']-wse_adjusted_mean, df['gauge_wse']-gauge_mean)
        
        # # Compute Spearman'S rho
        # # Using all data points:
        # spearman_unfiltered = compute_spearman(df['wse'], df['gauge_wse']) # Nan will be ignored
        # # Using only filtered data with filter_flag == 1:
        # spearman = compute_spearman(df['wse_adjusted'], df['gauge_wse'])
        # # Using CNES baseline 1: 
        # spearman_baseline1 = compute_spearman(df['wse_baseline1'], df['gauge_wse'])
        # # Using CNES baseline 2: 
        # spearman_baseline2 = compute_spearman(df['wse_baseline2'], df['gauge_wse'])

        # Compute variabilities between df.wse and df.gauge_wse, based on the paired timestamps (discrete timesteps) in df.gauge_datetime. 
        # In other words, this comparison applies to SWOT and gauge duplets only (they must be paired up). 
        # Using all data:
        # Mask for rows with valid gauge_datetime
        valid_mask = df['gauge_datetime'].notna()
        # Standard deviation of SWOT WSE for valid gauge comparisons
        var_swot_unfiltered = df.loc[valid_mask, 'wse'].std() #pandas.Series.std() automatically ignores NaN.
        # Standard deviation of gauge WSE
        var_gauge_unfiltered = df.loc[valid_mask, 'gauge_wse'].std()
        # Using only filtered data: 
        mask_filtered = valid_mask & (df['filter_flag'] == 1) #equals to: valid_mask & (df['wse_adjusted'].notna())
        var_swot = df.loc[mask_filtered, 'wse_adjusted'].std()
        var_gauge = df.loc[mask_filtered, 'gauge_wse'].std()
        # Using CNES baseline 1
        mask_baseline1 = valid_mask & (df['wse_baseline1'].notna())
        var_swot_baseline1 = df.loc[mask_baseline1, 'wse_adjusted'].std()
        var_gauge_baseline1 = df.loc[mask_baseline1, 'gauge_wse'].std()
        # Using CNES baseline 2
        mask_baseline2 = valid_mask & (df['wse_baseline2'].notna())
        var_swot_baseline2 = df.loc[mask_baseline2, 'wse_adjusted'].std()
        var_gauge_baseline2 = df.loc[mask_baseline2, 'gauge_wse'].std()
        
        # Note: var_gauge was computed from the gauge observations paired with SWOT data only,
        # and may not reflect the true variability of the full gauge time series.
        # To better assess how well the filter captures the "actual" lake WSE variability,
        # we identify the overlapping time range between the SWOT and gauge records,
        # generate consecutive daily timestamps within this range,
        # and interpolate both datasets to a consistent daily resolution.
        # We then compare their WSE variability over this shared time window.
        daily_series = convert_to_daily_series(df, gauge_df, 
                                            time_col='datetime', gauge_time_col='gauge_datetime',
                                            wse_col='wse', wse_filtered_col='wse_adjusted', gauge_wse_col='gauge_wse',
                                            interp_method='linear') 
        #filtered daily variability during the full period
        val = daily_series.get('daily_wse_filtered', np.nan)
        var_swot_daily = val.std() if isinstance(val, pd.Series) and not val.isna().all() else (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))
        #gauge daily variability during the full period
        val = daily_series.get('daily_gauge', np.nan)
        var_gauge_daily = val.std() if isinstance(val, pd.Series) and not val.isna().all() else (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))
        #raw daily variability during the full period
        val = daily_series.get('daily_wse', np.nan)
        var_swot_daily_unfiltered = val.std() if isinstance(val, pd.Series) and not val.isna().all() else (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))
        
        # Using CNES baseline 1
        daily_series_baseline1 = convert_to_daily_series(df, gauge_df, 
                                            time_col='datetime', gauge_time_col='gauge_datetime',
                                            wse_col='wse', wse_filtered_col='wse_baseline1', gauge_wse_col='gauge_wse',
                                            interp_method='linear') 
        val = daily_series_baseline1.get('daily_wse_filtered', np.nan)
        var_swot_daily_baseline1 = val.std() if isinstance(val, pd.Series) and not val.isna().all() else (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0)) 
        
        # Using CNES baseline 2
        daily_series_baseline2 = convert_to_daily_series(df, gauge_df, 
                                            time_col='datetime', gauge_time_col='gauge_datetime',
                                            wse_col='wse', wse_filtered_col='wse_baseline2', gauge_wse_col='gauge_wse',
                                            interp_method='linear')  
        val = daily_series_baseline2.get('daily_wse_filtered', np.nan)
        var_swot_daily_baseline2 = val.std() if isinstance(val, pd.Series) and not val.isna().all() else (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))  
    
        # Compute Spearman'S rho (based on daily time series)
        # Using all data points:
        spearman_unfiltered = compute_spearman(daily_series['daily_wse'], daily_series['daily_gauge']) # Nan will be ignored
        # Using only filtered data with filter_flag == 1:
        spearman = compute_spearman(daily_series['daily_wse_filtered'], daily_series['daily_gauge'])
        # Using CNES baseline 1: 
        spearman_baseline1 = compute_spearman(daily_series_baseline1['daily_wse_filtered'], daily_series_baseline1['daily_gauge'])
        # Using CNES baseline 2: 
        spearman_baseline2 = compute_spearman(daily_series_baseline2['daily_wse_filtered'], daily_series_baseline2['daily_gauge'])       
        
    else: # if this lake has no gauge data
        rmse = np.nan,            
        spearman = np.nan,            
        var_swot = np.nan,  #invalid lacking gauge reference          
        var_gauge = np.nan,              
        var_swot_daily = np.nan,            
        var_gauge_daily = np.nan,
            
        rmse_baseline1 = np.nan,            
        spearman_baseline1 = np.nan,            
        var_swot_baseline1 = np.nan,            
        var_gauge_baseline1 = np.nan,              
        var_swot_daily_baseline1 = np.nan,            
            
        rmse_baseline2 = np.nan,            
        spearman_baseline2 = np.nan,            
        var_swot_baseline2 = np.nan,            
        var_gauge_baseline2 = np.nan,              
        var_swot_daily_baseline2 = np.nan,            
                                  
        rmse_unfiltered = np.nan,
        spearman_unfiltered = np.nan,
        var_swot_unfiltered = np.nan,
        var_gauge_unfiltered = np.nan,            
        var_swot_daily_unfiltered = np.nan,
    
    if len(df) > 0: # If this PLD lake has SWOT observations
        retention_n = len(df_eval)
        retention_rate = len(df_eval)/len(df)
        retention_rate_baseline1 = len(df.query(CNES_baseline1))/len(df)
        retention_rate_baseline2 = len(df.query(CNES_baseline2))/len(df)  
        
        # Compute the proportion of fully ice-covered period in the original time series
        ice_duration = (df['ice_clim_f'] == 2).sum() / len(df) # Simple approach for now: just use the record number (not exact time)
        
    else: # Some lakes may not no SWOT observations (e.g., lake_id 4330037643)
        retention_n = np.nan #nan meaning not applicable to this lake as it has no SWOT observations. 
        retention_rate = np.nan
        retention_rate_baseline1 = np.nan
        retention_rate_baseline2 = np.nan
        # Update n_while and n_while_r2 (previously assigned to -1) to disambiguate the difference between:
        # having no SWOT data at all (NaN) and having insufficient data for filter application/evaluation (-1)
        n_while = np.nan
        n_while_r2 = np.nan  
        
        ice_duration = np.nan
    
    # Construct a lake stats dataframe for this lake
    df_this_lake_stats = pd.DataFrame([{
        'lake_id': feature_id,
        'rmse': rmse,            
        'spearman': spearman,            
        'var_swot': var_swot,            
        'var_gauge': var_gauge,              
        'var_swot_daily': var_swot_daily,            
        'var_gauge_daily': var_gauge_daily,
        
        'rmse_baseline1': rmse_baseline1,            
        'spearman_baseline1': spearman_baseline1,            
        'var_swot_baseline1': var_swot_baseline1,            
        'var_gauge_baseline1': var_gauge_baseline1,              
        'var_swot_daily_baseline1': var_swot_daily_baseline1,            
        
        'rmse_baseline2': rmse_baseline2,            
        'spearman_baseline2': spearman_baseline2,            
        'var_swot_baseline2': var_swot_baseline2,            
        'var_gauge_baseline2': var_gauge_baseline2,              
        'var_swot_daily_baseline2': var_swot_daily_baseline2,
        
        'rmse_unfiltered': rmse_unfiltered,
        'spearman_unfiltered': spearman_unfiltered,
        'var_swot_unfiltered': var_swot_unfiltered,
        'var_gauge_unfiltered': var_gauge_unfiltered,            
        'var_swot_daily_unfiltered': var_swot_daily_unfiltered,
        
        'retention_n': retention_n,
        'retention_rate': retention_rate,
        'retention_rate_baseline1': retention_rate_baseline1,
        'retention_rate_baseline2': retention_rate_baseline2,  
        
        'n_while': n_while, 
        'n_while_r2': n_while_r2,
        
        'ice_duration': ice_duration
        }])
    
    # To sum up for dataframe df_lake_stats:
    #       For lakes that have no SWOT data (df is empty): n_while, n_while_r2, and retention metrics are np.nan, 
    #                 and validation metrics are also np.nan.
    #       For lakes that are abandoned: n_while and n_while_r2 are -1, and retention metrics are 0,
    #                 and validation metrics involving filtered SWOT data (e.g., rmse) are all np.nan
    #       For lakes that do not have gauge data: validation metrics such as rmse are np.nan, 
    #                 but n_while, n_while_r2, and retention metrics may have values. 
    
    # Concatenate df_lake_time_series by this df, df_lake_stats by df_this_lake_stats, 
    # and df_lake_heuristic_thresholds by df_heuristic_thresholds
    df_lake_stats = pd.concat([df_lake_stats, df_this_lake_stats], ignore_index=True) #ignore_index=True resets the index.
    df_lake_heuristic_thresholds = pd.concat([df_lake_heuristic_thresholds, df_heuristic_thresholds], ignore_index=True) 
    df_lake_time_series = pd.concat([df_lake_time_series, df], ignore_index=True)     
    # Note: for df_lake_time_series, if the lake has no SWOT data (i.e., df is empty), no record is added. 
    #       for df_lake_stats, if the lake has no SWOT data, lake_id will be kept, but other attributes are nan.  
    #       for df_lake_heuristic_thresholds, if the lake has no SWOT data, lake_id will be kept as well. 
    
    
    # Continue to finish this plot.  
    # Plot raw WSE time series (with error bars)
    ax.errorbar(df.datetime, df.wse, yerr=df.wse_u, label='raw SP', marker='o',
            color=(0.6, 0.6, 0.6), markersize=4, capsize=3, linestyle='', zorder=2)   

    # Flag measurements with summary filters
    ax.plot(df.query('(quality_f == 1 & (crid != "PIC2" & crid != "PID0")) or (quality_f == 3 & (crid == "PIC2" or crid == "PID0"))').datetime, \
            df.query('(quality_f == 1 & (crid != "PIC2" & crid != "PID0")) or (quality_f == 3 & (crid == "PIC2" or crid == "PID0"))').wse,
            label='quality_f = bad', marker='s', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor='red')
    ax.plot(df.query('quality_f == 2 & (crid == "PIC2" or crid == "PID0")').datetime, \
            df.query('quality_f == 2 & (crid == "PIC2" or crid == "PID0")').wse,
            label='quality_f = degraded', marker='s', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor=(0,1,0))
    ax.plot(df.query('quality_f == 1 & (crid == "PIC2" or crid == "PID0")').datetime, \
            df.query('quality_f == 1 & (crid == "PIC2" or crid == "PID0")').wse,
            label='quality_f = suspect', marker='s', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor='m')
    ax.plot(df.query('xovr_cal_q == 2').datetime, df.query('xovr_cal_q == 2').wse,
            label='xovr_cal_q = bad', marker='D', linestyle='', markersize=8,
            markerfacecolor='none', markeredgecolor='brown')
    ax.plot(df.query('xovr_cal_q == 1').datetime, df.query('xovr_cal_q == 1').wse,
            label='xovr_cal_q = suspect', marker='D', linestyle='', markersize=8,
            markerfacecolor='none', markeredgecolor='orange')
    ax.plot(df.query('ice_clim_f == 2').datetime, df.query('ice_clim_f == 2').wse,
            label='ice_clim_f = full', marker='^', linestyle='', markersize=8,
            markerfacecolor='none', markeredgecolor='blue')
    ax.plot(df.query('ice_clim_f == 1').datetime, df.query('ice_clim_f == 1').wse,
            label='ice_clim_f = partial', marker='^', linestyle='', markersize=8,
            markerfacecolor='none', markeredgecolor='cyan')
    ax.plot(df.query('wse_std > 3').datetime, df.query('wse_std > 3').wse,
            label='wse_std > 3 m', marker='o', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor='yellow')
    
    # Plot filtered result (use wse_ajusted to account for possible cycle adjustment)
    ax.errorbar(df_eval.datetime, df_eval.wse_adjusted, df_eval.wse_u,
           label='heuristic filter', color='black', marker='o',
           markersize=4, capsize=3, linestyle='--')    

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    fig.autofmt_xdate()
    ax.set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))  
    if len(df_eval) >= 1:
        range_wse = np.nanmax(df_eval.wse)-np.nanmin(df_eval.wse)
        plt.ylim(np.nanmin(df_eval.wse)-range_wse*2, np.nanmax(df_eval.wse)+range_wse*2)
    elif len(df) >= 1:
        plt.ylim(np.nanmin(df.wse), np.nanmax(df.wse)) 
    #plt.ylim(20, 40)
    
    # Axis labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('WSE (m)', fontsize=12)
    ax.set_title('Lake ID ' + str(feature_id) + ' Time Series Plot ' + filter_type)
          
    # Add statistics as a text box
    if gauge_source is not None: # With gauge data    
        textstr = f'nrmse_unfiltered = {rmse_unfiltered:.3f}\nrmse = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    #ax.legend()  
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.) #legend outside of the plot
    
    #Save the plot
    plt.savefig(work_dir+'\Plots\lakeID_'+str(feature_id)+'_'+filter_type+'.png', bbox_inches='tight') 
    # Do NOT call plt.show(), so nothing opens in Spyder
    #plt.close()  # Optional: frees up memory if many plots are generated

## Optional: if we want to save Hydrocron time series into disk. 
#df_Hydrocron.to_csv(work_dir+'/df_Hydrocron.csv', index=False) 


"""
Present summative statistics for all validated lakes

Recall: major outputs from sections above are:
    df_lake_time_series
    df_lake_stats
"""
# Compute summary validation statistics for the reference lakes, defined as those with gauge measurements AND 
# those that are not abandoned by the customized filter (as well as having original initial SWOT data). 
# The value of rmse not na will work for all. 
unique_lakes = df_lake_stats.loc[df_lake_stats['rmse'].notna(), 'lake_id'].tolist() #unique lake_ids with gauge data

# If there is at least one lake with validation data (i.e., rmse is not NaN)
if unique_lakes: 
    # Create the figure with 3 subplots
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()  # This will make axes[0], axes[1], axes[2], axes[3] each refer to one subplot.
    
    # Scatter plot for gauge WSEs vs SWOT WSEs for each of the lakes
    # Initialize the axis range for the lake WSE plot. 
    min_val = []
    max_val = []
    # Initialize WSE anomalies (w.r.t. each lake WSE mean to avoid datum discrepancy)
    SWOT_wse_anomalies = []
    SWOT_wse_anomalies_unfiltered = []
    SWOT_wse_anomalies_baseline1 = []
    SWOT_wse_anomalies_baseline2 = []
    gauge_wse_anomalies = []
    for lake_id in unique_lakes: #Loop through each lake
        subset = df_lake_time_series[df_lake_time_series['lake_id'] == lake_id]
        # Further select records where gauge_wse is not nan. 
        subset = subset[subset['gauge_wse'].notna()]
        # Compute mean-adjusted anomalies to avoid datum discrepancies
        wse_adjusted_mean = np.nanmean(subset['wse_adjusted'])
        gauge_mean = np.nanmean(subset['gauge_wse'])
        swot_anom = subset['wse_adjusted'] - wse_adjusted_mean
        swot_anom_unfiltered = subset['wse'] - wse_adjusted_mean
        swot_anom_baseline1 = subset['wse_baseline1'] - wse_adjusted_mean
        swot_anom_baseline2 = subset['wse_baseline2'] - wse_adjusted_mean
        gauge_anom = subset['gauge_wse'] - gauge_mean
        # Extend flat sequences
        SWOT_wse_anomalies.extend(swot_anom.tolist())
        SWOT_wse_anomalies_unfiltered.extend(swot_anom_unfiltered.tolist())
        SWOT_wse_anomalies_baseline1.extend(swot_anom_baseline1.tolist())
        SWOT_wse_anomalies_baseline2.extend(swot_anom_baseline2.tolist())
        gauge_wse_anomalies.extend(gauge_anom.tolist())
        # Append min/max from current anomalies only (not the whole list)
        min_val.append(min(np.nanmin(gauge_anom), np.nanmin(swot_anom)))
        max_val.append(max(np.nanmax(gauge_anom), np.nanmax(swot_anom)))
    # Show all lakes in the same color (for simplicity)
    axes[0].scatter(gauge_wse_anomalies, SWOT_wse_anomalies_unfiltered, color='gray', \
                    label='raw SP', s=50, linewidth=0, alpha=0.3) # unfiltered result
    axes[0].scatter(gauge_wse_anomalies, SWOT_wse_anomalies_baseline1, color='blue', \
                    label='stringent CNES baseline', s=50, linewidth=0, alpha=0.3) # baseline 1 result
    axes[0].scatter(gauge_wse_anomalies, SWOT_wse_anomalies_baseline2, color='orange', \
                    label='lenient CNES baseline', s=50, linewidth=0, alpha=0.3) # baseline 2 result
    axes[0].scatter(gauge_wse_anomalies, SWOT_wse_anomalies, color='r', \
                    label='customized filter', s=50, linewidth=0, alpha=0.2) # filtered result
    # Add 1:1 diagonal line
    axes[0].plot([min(min_val), max(max_val)], [min(min_val), max(max_val)], 'k--', linewidth=1)
    # Axis labels and scaling
    axes[0].set_xlim(min(min_val), max(max_val)) 
    axes[0].set_ylim(min(min_val), max(max_val)) 
    axes[0].set_xscale('linear')
    axes[0].set_yscale('linear')
    axes[0].set_xlabel('Gauge WSE (m)')
    axes[0].set_ylabel('SWOT WSE (m)')
    axes[0].set_title('Lake WSE (wrt each lake mean level)')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].legend() 
        
    # Scatter plot for gauge daily WSE variability vs SWOT daily WSE variability
    axes[1].scatter(df_lake_stats.var_gauge_daily, df_lake_stats.var_swot_daily_unfiltered, color='gray', \
                    label='raw SP', s=50, linewidth=0, alpha=0.6) # unfiltered result
    axes[1].scatter(df_lake_stats.var_gauge_daily, df_lake_stats.var_swot_daily_baseline1, color='blue', \
                    label='stringent CNES baseline', s=50, linewidth=0, alpha=0.6) # baseline 1 result
    axes[1].scatter(df_lake_stats.var_gauge_daily, df_lake_stats.var_swot_daily_baseline2, color='orange', \
                    label='lenient CNES baseline', s=50, linewidth=0, alpha=0.6) # baseline 2 result
    axes[1].scatter(df_lake_stats.var_gauge_daily, df_lake_stats.var_swot_daily, color='r', \
                    label='customized filter', s=50, linewidth=0, alpha=0.8) # filtered result
    # Add 1:1 diagonal line
    min_val = min(np.nanmin(df_lake_stats.var_gauge_daily), np.nanmin(df_lake_stats.var_swot_daily_unfiltered), \
                  np.nanmin(df_lake_stats.var_swot_daily))
    max_val = max(np.nanmax(df_lake_stats.var_gauge_daily), np.nanmax(df_lake_stats.var_swot_daily_unfiltered), \
                  np.nanmax(df_lake_stats.var_swot_daily))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    # Axis labels and scaling
    axes[1].set_xlim(min_val, max_val) 
    axes[1].set_ylim(min_val, max_val) 
    #axes[1].set_xlim(0,10) 
    #axes[1].set_ylim(0,10) 
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Gauge WSE std. (m)')
    axes[1].set_ylabel('SWOT WSE std. (m)')
    axes[1].set_title('Seasonl lake variability (in std.)')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[1].legend() 
    
    # Print relative errors in variability (0-1)
    print(np.nanpercentile(abs(df_lake_stats.var_swot_daily_unfiltered - df_lake_stats.var_gauge_daily)/df_lake_stats.var_gauge_daily, 68))
    print(np.nanpercentile(abs(df_lake_stats.var_swot_daily_baseline1 - df_lake_stats.var_gauge_daily)/df_lake_stats.var_gauge_daily, 68))
    print(np.nanpercentile(abs(df_lake_stats.var_swot_daily_baseline2 - df_lake_stats.var_gauge_daily)/df_lake_stats.var_gauge_daily, 68))
    print(np.nanpercentile(abs(df_lake_stats.var_swot_daily - df_lake_stats.var_gauge_daily)/df_lake_stats.var_gauge_daily, 68))
    
    # axes[1].scatter(df_lake_stats.var_gauge_unfiltered, df_lake_stats.var_swot_unfiltered, color='gray', \
    #                 label='raw SP', s=50, linewidth=0, alpha=0.6) # unfiltered result
    # axes[1].scatter(df_lake_stats.var_gauge_baseline1, df_lake_stats.var_swot_baseline1, color='blue', \
    #                 label='stringent CNES baseline', s=50, linewidth=0, alpha=0.6) # baseline 1 result
    # axes[1].scatter(df_lake_stats.var_gauge_baseline2, df_lake_stats.var_swot_baseline2, color='orange', \
    #                 label='lenient CNES baseline', s=50, linewidth=0, alpha=0.6) # baseline 2 result
    # axes[1].scatter(df_lake_stats.var_gauge, df_lake_stats.var_swot, color='r', \
    #                 label='customized filter', s=50, linewidth=0, alpha=0.8) # filtered result
    # # Add 1:1 diagonal line
    # min_val = min(np.nanmin(df_lake_stats.var_gauge_unfiltered), np.nanmin(df_lake_stats.var_swot_unfiltered), \
    #               np.nanmin(df_lake_stats.var_swot))
    # max_val = max(np.nanmax(df_lake_stats.var_gauge_unfiltered), np.nanmax(df_lake_stats.var_swot_unfiltered), \
    #               np.nanmax(df_lake_stats.var_swot))
    # axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
    # # Axis labels and scaling
    # axes[1].set_xlim(min_val, max_val) 
    # axes[1].set_ylim(min_val, max_val) 
    # #axes[1].set_xlim(0,10) 
    # #axes[1].set_ylim(0,10) 
    # axes[1].set_xscale('log')
    # axes[1].set_yscale('log')
    # axes[1].set_xlabel('Gauge WSE std. (m)')
    # axes[1].set_ylabel('SWOT WSE std. (m)')
    # axes[1].set_title('Seasonl lake variability (in std.)')
    # axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    # axes[1].legend() 
    
    # # Print relative errors in variability (0-1)
    # print(np.nanpercentile(abs(df_lake_stats.var_swot_unfiltered - df_lake_stats.var_gauge_unfiltered)/df_lake_stats.var_gauge_unfiltered, 68))
    # print(np.nanpercentile(abs(df_lake_stats.var_swot_baseline1 - df_lake_stats.var_gauge_baseline1)/df_lake_stats.var_gauge_baseline1, 68))
    # print(np.nanpercentile(abs(df_lake_stats.var_swot_baseline2 - df_lake_stats.var_gauge_baseline2)/df_lake_stats.var_gauge_baseline2, 68))
    # print(np.nanpercentile(abs(df_lake_stats.var_swot - df_lake_stats.var_gauge)/df_lake_stats.var_gauge, 68))
    
    # Plot CDF of spearman rhos
    # Sort values for CDF
    spearman_unfiltered_sorted = np.sort(df_lake_stats['spearman_unfiltered'].dropna().values)
    spearman_baseline1_sorted = np.sort(df_lake_stats['spearman_baseline1'].dropna().values)
    spearman_baseline2_sorted = np.sort(df_lake_stats['spearman_baseline2'].dropna().values)
    spearman_filtered_sorted = np.sort(df_lake_stats['spearman'].dropna().values)
    # Compute empirical CDF values
    cdf_spearman_unfiltered = np.linspace(0, 1, len(spearman_unfiltered_sorted), endpoint=False)
    cdf_spearman_baseline1 = np.linspace(0, 1, len(spearman_baseline1_sorted), endpoint=False)
    cdf_spearman_baseline2 = np.linspace(0, 1, len(spearman_baseline2_sorted), endpoint=False)
    cdf_spearman_filtered = np.linspace(0, 1, len(spearman_filtered_sorted), endpoint=False)
    # Plot
    axes[2].plot(spearman_unfiltered_sorted, cdf_spearman_unfiltered, label='raw SP', color='gray', alpha=0.6)
    axes[2].plot(spearman_baseline1_sorted, cdf_spearman_baseline1, label='stringent CNES baseline', color='blue', alpha=0.6)
    axes[2].plot(spearman_baseline2_sorted, cdf_spearman_baseline2, label='lenient CNES baseline', color='orange', alpha=0.6)
    axes[2].plot(spearman_filtered_sorted, cdf_spearman_filtered, label='customized filter', color='red', alpha=0.8)
    # Axis labels and scaling
    axes[2].set_xlim(-1, 1) 
    axes[2].set_ylim(0, 1) 
    axes[2].set_xscale('linear')
    axes[2].set_yscale('linear')
    axes[2].set_xlabel('Spearman rho')
    axes[2].set_ylabel('CDF')
    axes[2].set_title('CDF curves of Spearman rho')
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[2].legend() 
    
    # Plot CDF of retention rates
    # Sort values for CDF
    subset = df_lake_stats[(df_lake_stats['n_while'] != -1) & (df_lake_stats['n_while_r2'] != -1)] #drop abandoned lake
    retention_rate_sorted = np.sort(subset['retention_rate'].dropna().values)
    retention_rate_baseline1_sorted = np.sort(subset['retention_rate_baseline1'].dropna().values)
    retention_rate_baseline2_sorted = np.sort(subset['retention_rate_baseline2'].dropna().values)
    # Compute empirical CDF values
    cdf_retention_rate = np.linspace(0, 1, len(retention_rate_sorted), endpoint=False)
    cdf_retention_rate_baseline1 = np.linspace(0, 1, len(retention_rate_baseline1_sorted), endpoint=False)
    cdf_retention_rate_baseline2 = np.linspace(0, 1, len(retention_rate_baseline2_sorted), endpoint=False)
    # Plot
    axes[3].plot(retention_rate_baseline1_sorted, cdf_retention_rate_baseline1, \
                 label='stringent CNES baseline', color='blue', alpha=0.8)
    axes[3].plot(retention_rate_baseline2_sorted, cdf_retention_rate_baseline2, \
                 label='lenient CNES baseline', color='orange', alpha=0.8)
    axes[3].plot(retention_rate_sorted, cdf_retention_rate, label='customized filter', color='red', alpha=0.8)
    # Axis labels and scaling
    axes[3].set_xlim(0, 1) 
    axes[3].set_ylim(0, 1) 
    axes[3].set_xscale('linear')
    axes[3].set_yscale('linear')
    axes[3].set_xlabel('Retention rates')
    axes[3].set_ylabel('CDF')
    axes[3].set_title('CDF curves of retention rates')
    axes[3].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[3].legend()
    
    #Save the plot
    plt.tight_layout()
    plt.savefig(work_dir+'\Plots\lake_stats.png')    
       
    
    # Exploring the relationship between ice duration and seasonal variability error. 
    # Plot the scatter plot between df_lake_stats['ice_duration'] and relative error in seasonal variability
    df_lake_stats['rel_err_var'] = abs(df_lake_stats.var_swot_daily - df_lake_stats.var_gauge_daily)/df_lake_stats.var_gauge_daily
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, linewidth=0.5, zorder=1)
    axes = axes.flatten()  # This will make axes[0], axes[1], axes[2], axes[3] each refer to one subplot.
    ax.plot(df_lake_stats['ice_duration'], df_lake_stats['rel_err_var'], 'o')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel('Proportion of ice observations')
    ax.set_ylabel('Relative error of seasonal variability (m)')
    ax.set_title('Seasonl lake variability (in std.)')
    plt.tight_layout()
    #plt.show()
    plt.savefig(work_dir+'\Plots\ice_error_scatter.png') 
    
    # Box plot
    bins = np.arange(0, 1.01, 0.1)
    labels = [f'{round(bins[i], 1)}–{round(bins[i+1], 1)}' for i in range(len(bins)-1)]
    # Add binned ice_duration column
    df_lake_stats['ice_duration_bin'] = pd.cut(
        df_lake_stats['ice_duration'],
        bins=bins,
        labels=labels,
        include_lowest=True
        )
    sns.boxplot(x='ice_duration_bin', y='rel_err_var', data=df_lake_stats)
    plt.xlabel('Binned Ice Duration')
    plt.ylabel('Relative Error in Variability')
    plt.title('Boxplot of Relative Error vs Ice Duration')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()
    plt.savefig(work_dir+'\Plots\ice_error_box.png') 




'''
Sensitivity test of SWOT passes and SP versions for heuristic thresholds 

The output is df_heuristic_threshold_sensitivity, which contains heuristic threshold sensitivity statistics for each
unique lake_id derived from df_lake_heuristic_thresholds.

Each row corresponds to a single lake_id and summarizes:
   - The threshold differences between two CRID scenarios:
     ("PIC2_or_PID0" as newer versions vs "early_versions" as older ones)
   - The variability in thresholds across different SWOT pass_ids.

 Columns:
   - lake_id:                PLD lake id.

   - wse_std_versionnew:     Maximum wse_std_threshold for the lake under
                             crid_scenario == "PIC2_or_PID0".

   - wse_std_versionold:     Maximum wse_std_threshold for the lake under
                             crid_scenario == "early_versions".

   - wse_u_versionnew:       Maximum wse_u_threshold for the lake under
                             crid_scenario == "PIC2_or_PID0".

   - wse_u_versionold:       Maximum wse_u_threshold for the lake under
                             crid_scenario == "early_versions".

   - version_count:          Number of unique crid_scenario values observed
                             for this lake (should be 1 or 2).

   - wse_std_passvar:        Standard deviation of the maximum wse_std_threshold 
                             values computed per unique pass_id
                             (regardless of crid_scenario).

   - wse_u_passvar:          Standard deviation of the maximum wse_u_threshold 
                             values computed per unique pass_id
                             (regardless of crid_scenario).

   - pass_count:             Number of unique pass_id values used in threshold 
                             calibration for this lake.
'''
results = [] # Initialize a class list
for lake_id, group in df_lake_heuristic_thresholds.groupby('lake_id'):
    row = {'lake_id': lake_id}

    # Split by crid_scenario
    group_new = group[group['crid_scenario'] == 'PIC2_or_PID0']
    group_old = group[group['crid_scenario'] == 'early_versions']

    # Max thresholds for each version (ignore NaNs if any valid)
    row['wse_std_versionnew'] = group_new['wse_std_threshold'].dropna().max() if not group_new['wse_std_threshold'].dropna().empty else np.nan
    row['wse_std_versionold'] = group_old['wse_std_threshold'].dropna().max() if not group_old['wse_std_threshold'].dropna().empty else np.nan
    row['wse_u_versionnew'] = group_new['wse_u_threshold'].dropna().max() if not group_new['wse_u_threshold'].dropna().empty else np.nan
    row['wse_u_versionold'] = group_old['wse_u_threshold'].dropna().max() if not group_old['wse_u_threshold'].dropna().empty else np.nan

    # Number of unique crid_scenario values
    row['version_count'] = group['crid_scenario'].nunique()

    # Standard deviation of max thresholds per unique pass_id
    std_per_pass = group.groupby('pass_id')['wse_std_threshold'].max()
    u_per_pass = group.groupby('pass_id')['wse_u_threshold'].max()

    row['wse_std_passvar'] = std_per_pass.std(ddof=0) if len(std_per_pass) > 1 else 0.0
    row['wse_u_passvar'] = u_per_pass.std(ddof=0) if len(u_per_pass) > 1 else 0.0

    # Number of unique pass_id values
    row['pass_count'] = group['pass_id'].nunique()

    results.append(row)
df_heuristic_threshold_sensitivity = pd.DataFrame(results)

# Plot the sensitivity test
plt.rcParams["font.family"] = "Arial"
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.flatten()  # This will make axes[0], axes[1], axes[2], axes[3] each refer to one subplot.

# Scatter plot of wse_std_versionold vs wse_std_versionnew
x = df_heuristic_threshold_sensitivity['wse_std_versionold']
y = df_heuristic_threshold_sensitivity['wse_std_versionnew']
axes[0].scatter(x, y, alpha=0.7)
# Add 1:1 diagonal line
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[0].set_xscale('linear')
axes[0].set_yscale('linear')
axes[0].set_xlabel('PIC0 & PGC0')
axes[0].set_ylabel('PIC2 & PID0')
axes[0].set_title('wse_std threshold')
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Scatter plot of wse_u_versionold vs wse_u_versionnew
x = df_heuristic_threshold_sensitivity['wse_u_versionold']
y = df_heuristic_threshold_sensitivity['wse_u_versionnew']
axes[1].scatter(x, y, alpha=0.7)
# Add 1:1 diagonal line
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
axes[1].set_xscale('linear')
axes[1].set_yscale('linear')
axes[1].set_xlabel('PIC0 & PGC0')
axes[1].set_ylabel('PIC2 & PID0')
axes[1].set_title('wse_u threshold')
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Histogram of wse_std_passvar (excluding pass_count = 0)
filtered_df = df_heuristic_threshold_sensitivity[df_heuristic_threshold_sensitivity['pass_count'] > 0]
axes[2].hist(filtered_df['wse_std_passvar'], bins=20, edgecolor='black', alpha=0.75)
axes[2].set_xlabel('Variation among passes')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Histogram of wse_std threshold variation among passes')
axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

# Histogram of wse_u_passvar (excluding pass_count = 0)
axes[3].hist(filtered_df['wse_u_passvar'], bins=20, edgecolor='black', alpha=0.75)
axes[3].set_xlabel('Variation among passes')
axes[3].set_ylabel('Frequency')
axes[3].set_title('Histogram of wse_u threshold variation among passes')
axes[3].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig(work_dir+'\Plots\heuristic_sensitivity_test.png') 
