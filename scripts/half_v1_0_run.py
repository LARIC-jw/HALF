"""
Heuristic Adaptive Lake Filter (HALF) execution workflow for SWOT LakeSP.

HALF is designed to remove outliers from SWOT LakeSP water surface elevation
(WSE) time series while preserving hydrologically meaningful temporal dynamics.

The workflow implemented here follows three primary steps:

    1. Calibrate lake-specific heuristic thresholds from a conservative,
       high-quality LakeSP subset.
    2. Build a heuristic baseline and iteratively remove residual outliers using
       a configurable low-pass filter.
    3. Optionally correct cross-pass WSE offsets for lakes observed by multiple
       SWOT passes within the same cycle.

This script queries the Hydrocron API for one or more Prior Lake Database (PLD)
lake IDs supplied by the user. It (1) applies HALF, (2) creates a raw-versus-filtered
diagnostic plot for each lake with usable observations, and (3) writes the
combined observation-level results and final-attempt threshold summary to the
configured work directory. Observation-level threshold traceability fields can
optionally be appended to the main LakeSP output table.

Script name and version
-------
half_v1_0_run.py: 
    Main script to run the filter workflow
HALF v1.0
Last updated: 2026-06-19

Script by:
-------
Jida Wang (jidaw@illinois.edu)
Mélanie Trudel (melanie.trudel@usherbrooke.ca)

Parts of this script were developed with assistance from ChatGPT (OpenAI) for 
brainstorming, debugging, drafting, documentation, and related editorial suggestions.

The authors are responsible for the conceptual design, methodological integrity, 
maintenance, review, testing, and validation of the final implementation.

Citation
--------
Trudel, M., Wang, J., Biancamaria, S., Harlan, M.E., Shah, D., Gao, H.,
Collins, E., Getirana, A., Song, C., Reis Alencar Oliveira, R., Gosset, M.,
Rodrigues Martins, E.S., Fleischmann, A., Hymans, D., Grippa, M., Girard, F.,
Kergoat, L., Pottier, C., Fjørtoft, R., Oubanas, H., & Pavelsky, T.M. (2026).
A Heuristic Adaptive Filter for SWOT Lake Vector Data Products.
Geophysical Research Letters, in review.
"""

from __future__ import annotations

import os
from io import StringIO

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Import reusable HALF functions from half_v1_0_functions.py.
# Keep that module in the same directory as this script, or otherwise make it
# available on the Python import path.
from half_v1_0_functions import (
    calibrate_heuristic_thresholds,
    apply_customized_filter,
    apply_baseline_tukey_filter,
    sp_cycle_adjustment,
)


# =============================================================================
# User configuration and schema documentation
# =============================================================================
# This section contains three parts:
#
#   1. Documentation dictionaries:
#        INPUT_SCHEMA and OUTPUT_SCHEMA describe the expected inputs and outputs.
#        They are provided primarily for user guidance. The required LakeSP
#        columns are also checked after each Hydrocron query.
#
#   2. Editable runtime configuration:
#        RUN_CONFIG and FILTER_CONFIG control how the filtering workflow runs.
#        Review these dictionaries before executing the script.
#
#   3. Derived variables:
#        Scalar variables below the configuration dictionaries are generated for
#        compatibility with the existing script structure. These should normally
#        not be edited directly.
#
# The default filtering settings follow the configuration used in
# Trudel et al. (2026), subject to the selected LakeSP collection and time range.
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Documentation dictionaries: inputs and outputs
# -----------------------------------------------------------------------------

INPUT_SCHEMA = {
    "Lake IDs": {
        "source": "RUN_CONFIG['lake_ids']",
        "description": (
            "One or more PLD lake IDs to query from the Hydrocron PriorLake "
            "time-series endpoint. Enter the IDs as a Python list."
        ),
        "example": [2510236842, 6220321573],
    },
    "Observation-level threshold-detail option": {
        "source": "RUN_CONFIG['append_threshold_details_to_output']",
        "allowed_values": ["yes", "no"],
        "description": (
            "Controls whether threshold source, calibrated, bounded, and "
            "applied values are appended to each LakeSP observation in "
            "df_Hydrocron."
        ),
    },
    "LakeSP time series": {
        "source": "Hydrocron API",
        "required_columns": [
            "lake_id", "time", "time_str", "wse", "wse_u", "wse_std",
            "xtrk_dist", "quality_f", "xovr_cal_q", "ice_clim_f",
            "cycle_id", "pass_id", "crid", "collection_shortname",
            "p_ref_area", "area_total",
        ],
        "notes": (
            "LakeSP time is measured in seconds since 2000-01-01 00:00:00 UTC "
            "and is used for chronological sorting because it can be more "
            "precise than time_str. Internet access is required for Hydrocron."
        ),
    },
    "HALF function module": {
        "source": "half_v1_0_functions.py",
        "required_functions": [
            "calibrate_heuristic_thresholds",
            "apply_customized_filter",
            "apply_baseline_tukey_filter",
            "sp_cycle_adjustment",
        ],
    },
}

OUTPUT_SCHEMA = {
    "df_Hydrocron": (
        "Combined observation-level LakeSP table for all successfully queried "
        "and processed lakes. It retains the requested Hydrocron fields and "
        "appends index_col, datetime, filter_flag, wse_adjusted, n_while, "
        "n_while_r2, filter_attempt, and intra_cycle_flag. When "
        "RUN_CONFIG['append_threshold_details_to_output'] is 'yes', the table "
        "also includes observation-level threshold source, calibrated, bounded, "
        "and applied values for wse_std, wse_u, and xtrk_dist."
    ),
    "df_threshold_summary": (
        "Combined threshold-row table for all processed lakes. It contains the "
        "calibrated thresholds returned by calibrate_heuristic_thresholds(), "
        "bounded thresholds associated with the final strict or lenient HALF "
        "attempt, grouping provenance, and filter_attempt. Bounded values are "
        "left blank when the final result uses the Tukey fallback or no filter."
    ),
    "df_Hydrocron_output_csv": (
        "CSV export of df_Hydrocron written to work_dir. This file contains the "
        "original Hydrocron/LakeSP observations augmented with HALF outputs, "
        "including filter_flag, wse_adjusted, filter_attempt, and optional "
        "observation-level threshold details."
    ),    
    "threshold_summary_csv": (
        "CSV export of df_threshold_summary written to work_dir using the "
        "filename df_HALF_threshold_summary_<script_version>_"
        "<collection_version>.csv."
    ),
    "diagnostic_plots": (
        "One raw-versus-HALF PNG diagnostic plot per lake with usable LakeSP "
        "observations, written directly to work_dir."
    ),
}


# -----------------------------------------------------------------------------
# 2. Editable runtime configuration
# -----------------------------------------------------------------------------
# Edit RUN_CONFIG and FILTER_CONFIG before running the workflow as needed.
# RUN_CONFIG controls lake IDs, the LakeSP collection, query period, paths,
# output naming, and optional threshold-detail fields. FILTER_CONFIG controls
# HALF filtering behavior.

RUN_CONFIG = {
    # One or more PLD lake IDs. Duplicate IDs are removed while preserving the
    # order of their first occurrence.
    "lake_ids": [7240054132, 2510236842, 4340980733],

    # LakeSP collection short name. Common values:
    #   - "SWOT_L2_HR_LakeSP_2.0" for Version C
    #   - "SWOT_L2_HR_LakeSP_D"   for Version D
    "collection_shortname": "SWOT_L2_HR_LakeSP_2.0",

    # UTC query window passed directly to Hydrocron.
    "start_time": "2023-07-21T05:33:45Z",
    "end_time": "2025-05-18T21:36:19Z",

    # Repository/work directory. Keep "." for a repository-relative run, or
    # replace it with an absolute path for a local run.
    # By default, work_dir is the output directory, too. 
    "work_dir": ".", 

    # Script label used in output filenames.
    "script_version": "V1_0",

    # Optional intra-cycle cross-pass WSE adjustment. Recommended default: "yes".
    "execute_intra_cycle_adjustment": "yes",

    # If "yes", append observation-level threshold traceability columns to
    # df_Hydrocron: *_thr_source, *_thr_cal, *_thr_bound, and *_thr_app.
    # If "no", these details are omitted from df_Hydrocron; the separate
    # threshold-summary CSV is still written.
    "append_threshold_details_to_output": "yes",
}

# Filter parameters:
# The parameter values below and other related parameters passed to HALF functions
# (e.g., threshold-calibration grouping options such as by_crid_scenario,
# by_pass_id, and by_ice; threshold-application settings such as threshold
# bounds, ice overrides, and ice-condition rules; and filtering settings such
# as temporal-gap criteria, low-pass filter type, z-score thresholds, and
# residual-spread tolerances) are based on experimentation in Trudel et al.
# (2026, in review). These settings were initially tuned with visual inspection
# aided by gauge data for LakeSP Version C. They may therefore not be optimal
# for every lake, region, hydrologic condition, or newer LakeSP version. Further
# lake-specific, region-specific, or version-specific tuning may be needed.
FILTER_CONFIG = {
    # If "yes" (recommended), run both heuristic-baseline filtering and
    # iterative low-pass residual filtering. If "no", return only the
    # heuristic-baseline subset.
    "apply_low_pass_filter": "yes",

    # If "yes", evaluate residual outliers against the full LakeSP record.
    # If "no" (recommended), evaluate only the current candidate retained set.
    "evaluating_at_full_data": "no",

    # Optional second filtering round. Round 1 is more aggressive; round 2 is
    # more permissive and may recover valid observations.
    "r2_filter": "yes",

    # Supported low-pass filters: lowess, wavelet, savgol, kalman, spline,
    # median, and hampel.
    "filter_type": "savgol",

    # Residual z-score thresholds for [round 1, round 2]. Examples:
    # 2.576 ≈ 99% two-tailed, 2.807 ≈ 99.5%, 2.967 ≈ 99.7%,
    # 3.291 ≈ 99.9%, and 3.5 ≈ 99.95%.
    "z_score_thresholds": [2.576, 3.5],

    # Maximum relative residual-spread tolerances for [round 1, round 2].
    # Smaller values are stricter; larger values preserve more observations.
    "maximum_residual_spreads": [0.07, 0.05],

    # Diagnostic plotting switch for iteration evolution. Use "yes" only when
    # inspecting individual cases because a batch run can generate many figures.
    "show_filtering_evolution": "no",

    # Temporal-coverage criteria used by apply_customized_filter().     
    # Strict mode fails when the heuristic baseline contains any gap longer than
    # the specified maximum temporal gap, or when it does not meet the minimum
    # temporal range/span requirement.    
    # Lenient mode permits major baseline gaps, but it avoids residual evaluation
    # inside those unsupported gaps, where the smoothing curve is weakly constrained.
    "strict_max_temporal_gap_days": 90,
    "strict_min_temporal_range_days": 365,
    "lenient_max_temporal_gap_days": 90,
    "lenient_min_temporal_range_days": 365,
}


# -----------------------------------------------------------------------------
# 3. Derived variables used by the remainder of the script
# -----------------------------------------------------------------------------
# Edit RUN_CONFIG or FILTER_CONFIG rather than modifying this block directly.

lake_ids = RUN_CONFIG["lake_ids"]
collection_shortname = RUN_CONFIG["collection_shortname"]
start_time = RUN_CONFIG["start_time"]
end_time = RUN_CONFIG["end_time"]
work_dir = os.path.abspath(RUN_CONFIG["work_dir"])
script_version = RUN_CONFIG["script_version"]
execute_intra_cycle_adjustment = RUN_CONFIG["execute_intra_cycle_adjustment"]
append_threshold_details_to_output = RUN_CONFIG["append_threshold_details_to_output"]

apply_low_pass_filter = FILTER_CONFIG["apply_low_pass_filter"]
evaluating_at_full_data = FILTER_CONFIG["evaluating_at_full_data"]
r2_filter = FILTER_CONFIG["r2_filter"]
filter_type = FILTER_CONFIG["filter_type"]
z_score_thresholds = FILTER_CONFIG["z_score_thresholds"]
maximum_residual_spreads = FILTER_CONFIG["maximum_residual_spreads"]
show_filtering_evolution = FILTER_CONFIG["show_filtering_evolution"]
strict_max_temporal_gap_days = FILTER_CONFIG["strict_max_temporal_gap_days"]
strict_min_temporal_range_days = FILTER_CONFIG["strict_min_temporal_range_days"]
lenient_max_temporal_gap_days = FILTER_CONFIG["lenient_max_temporal_gap_days"]
lenient_min_temporal_range_days = FILTER_CONFIG["lenient_min_temporal_range_days"]

# Accept a single scalar ID for convenience, but normalize all input to a list.
if isinstance(lake_ids, (str, int, np.integer)):
    lake_ids = [lake_ids]
else:
    lake_ids = list(lake_ids)

lake_ids = list(dict.fromkeys(lake_ids))
if not lake_ids:
    raise ValueError("RUN_CONFIG['lake_ids'] must contain at least one PLD lake ID.")

if append_threshold_details_to_output not in {"yes", "no"}:
    raise ValueError(
        "RUN_CONFIG['append_threshold_details_to_output'] must be 'yes' or 'no'."
    )

VERSION_FILENAME_BY_COLLECTION = {
    "SWOT_L2_HR_LakeSP_2.0": "vC",
    "SWOT_L2_HR_LakeSP_D":   "vD",
}

if collection_shortname not in VERSION_FILENAME_BY_COLLECTION:
    valid = ", ".join(VERSION_FILENAME_BY_COLLECTION)
    raise ValueError(
        f"Unsupported collection_shortname={collection_shortname!r}. "
        f"Expected one of: {valid}."
    )

version_filename = VERSION_FILENAME_BY_COLLECTION[collection_shortname]

os.makedirs(work_dir, exist_ok=True)

# Combined observation-level Hydrocron table with appended HALF outputs.
df_Hydrocron_output_csv = os.path.join(
    work_dir,
    f"df_Hydrocron_with_HALF_{script_version}_{version_filename}.csv",
)

# Combined final-attempt threshold-summary output file.
threshold_summary_csv = os.path.join(
    work_dir,
    f"df_HALF_threshold_summary_{script_version}_{version_filename}.csv",
)


"""
-----------------------------
---------MAIN OUTPUTS--------
-----------------------------
The filtering workflow accumulates two primary DataFrames:

1. df_Hydrocron
   Each row represents one original LakeSP observation from a successfully
   queried lake. In addition to the requested Hydrocron attributes, the
   workflow appends HALF filtering outputs. 
   See apply_customized_filter() for the main filtering workflow, 
   sp_cycle_adjustment() for wse_adjusted, and apply_heuristic_thresholds() 
   for the optional observation-level threshold traceability fields.

   - index_col:
       stable within-lake unique ID used to map retained observations back
       to the original LakeSP record.

   - datetime:
       high-precision UTC timestamp derived from LakeSP time.

   - filter_flag:
       1 for an observation retained by HALF; 0 for a removed observation.

   - wse_adjusted:
       retained WSE after optional cross-pass bias correction; NaN for removed
       observations.

   - n_while and n_while_r2:
       lake-level iteration counts/status codes for filtering rounds 1 and 2.
       These values are repeated for all observation rows from the same lake.

   - filter_attempt:
       lake-level filtering hierarchy outcome: 1_strict, 2_lenient, 3_tukey,
       or 4_none. True no-data cases are skipped and therefore are not written
       to df_Hydrocron.

   - intra_cycle_flag:
       lake-level flag indicating whether cross-pass adjustment changed at least
       one retained WSE value. A value of 1 means at least one value was
       adjusted; 0 means no retained value was changed. This value is repeated
       for all observation rows from the same lake.
 
    If append_threshold_details_to_output == "yes", df_Hydrocron also includes
    observation-level threshold traceability fields for each metric. 
    See apply_heuristic_thresholds() in half_v1_0_functions.py for the full 
    definitions and implementation details of these fields:
        
    - row_ice_condition:        
        Row-level ice category derived from ice_clim_f using the same convention
        used for threshold calibration and rule-based threshold selection:
        observations with ice_clim_f < 2 are treated as "ice-free", while
        observations with ice_clim_f >= 2 are treated as "ice-covered". 

    - *_thr_source:
        Selected threshold source for the observation: ice-free, ice-covered,
        both, or not apply.
    
    - *_thr_cal:
        Calibrated threshold selected for the observation from
        df_heuristic_thresholds, before missing-value fallback, safeguard bounds,
        or ice-condition overrides.
    
    - *_thr_bound:
        Selected threshold after missing-value fallback and safeguard bounds, but
        before observation-level ice-condition overrides.
    
    - *_thr_app:
        Final threshold applied to the observation. For wse_std and wse_u, this
        includes the ice-condition override when ice_clim_f >= 1. For xtrk_dist,
        no ice override is applied, so *_thr_app equals *_thr_bound whenever
        xtrk_dist is active.
    
    Observations removed before threshold evaluation, such as duplicate
    high-precision timestamps excluded from df_eval, retain NaN in these optional
    threshold-detail columns.

2. df_threshold_summary
   Threshold-row table stacked across processed lakes. It contains one row per
   calibrated threshold group from df_heuristic_thresholds, together with
   grouping_scheme, final-attempt bounded thresholds, and filter_attempt. 
   See calibrate_heuristic_thresholds() for how calibrated threshold groups and
   grouping_scheme are defined, and apply_heuristic_thresholds() for how
   final-attempt bounded thresholds are generated.

   Main threshold fields are:

   - *_thr_cal:
       Calibrated threshold from the conservative high-quality LakeSP subset.

   - *_thr_bound:
       Final-attempt threshold after missing-value fallback and safeguard
       bounds. These values are attempt-specific because strict and lenient
       HALF attempts use different bounds.

   - grouping_scheme:
       Provenance flag describing how the threshold row was represented in the
       full input time series and conservative calibration subset. See
       calibrate_heuristic_thresholds() for the detailed definition.

   - filter_attempt:
       Final filtering hierarchy outcome for the lake: 1_strict, 2_lenient,
       3_tukey, or 4_none.

   Note: Interpreting blank threshold fields:
       Blank *_thr_cal values mean that the calibrated threshold could not be
       estimated for that threshold group from the conservative calibration subset
       and could not be filled by the fallback rules in
       calibrate_heuristic_thresholds(). This can occur for sparse groups,
       especially for pass-dependent xtrk_dist thresholds.
       
       *_thr_bound is populated only for threshold rows selected by the final 
       HALF threshold-application rules and only for metrics that are active. 
       For example, if xtrk_dist is set to "not apply", xtrk_dist_thr_bound remains
       blank even when xtrk_dist_thr_cal exists. 
       Similarly, if the rules select only the ice-free threshold row, 
       the ice-covered and both rows may retain calibrated values but have blank 
       bounded values.
       
       For final 3_tukey or 4_none outcomes, *_thr_cal values are retained for
       traceability, but *_thr_bound values are left blank because heuristic
       thresholds were not the final filtering method.


Iteration-status conventions
----------------------------
- -9: no valid LakeSP input; filtering was not applicable.
- -2: the filtering round was disabled or not applicable.
- -1: the filtering attempt started but was abandoned.
-  0: no regular iteration was completed.
- >0: number of completed filtering iterations.
"""

# Define the LakeSP numeric fill value used by this workflow.
fill_float = -999999999999

# Collect per-lake observation tables in a list and concatenate them once after
# the loop. This avoids repeatedly reallocating a growing DataFrame.
df_Hydrocron_parts = []

# Collect per-lake calibrated/bounded threshold tables in the same way.
df_threshold_summary_parts = []

# Loop through each requested PLD lake ID.
for feature_id in lake_ids:
    print(f"Lake ID: {feature_id}")

    # Retrieve the LakeSP time series from Hydrocron.
    feature = "PriorLake"
    output = "csv"
    fields = (
        "lake_id,reach_id,obs_id,overlap,n_overlap,time,time_tai,time_str,wse,"
        "wse_u,wse_r_u,wse_std,area_total,area_tot_u,area_detct,area_det_u,"
        "layovr_val,xtrk_dist,ds1_l,ds1_l_u,ds1_q,ds1_q_u,ds2_l,ds2_l_u,"
        "ds2_q,ds2_q_u,quality_f,dark_frac,ice_clim_f,ice_dyn_f,partial_f,"
        "xovr_cal_q,geoid_hght,solid_tide,load_tidef,load_tideg,pole_tide,"
        "dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,lake_name,p_res_id,p_lon,"
        "p_lat,p_ref_wse,p_ref_area,p_date_t0,p_ds_t0,p_storage,cycle_id,"
        "pass_id,continent_id,range_start_time,range_end_time,crid,geometry,"
        "PLD_version,collection_shortname"
    ) #LakeSP built-in fields. 

    enquiry_input = (
        "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        + "feature=" + feature
        + "&feature_id=" + str(feature_id)
        + "&start_time=" + start_time
        + "&end_time=" + end_time
        + "&output=" + output
        + "&fields=" + fields
        + "&collection_name=" + collection_shortname
    )

    # Robust handling of possible request/response problems:
    #   - request failure or invalid JSON response
    #   - missing or empty Hydrocron result table
    #   - unreadable CSV output
    #   - missing LakeSP columns required by the HALF workflow
    # If any of these checks fail, the lake is skipped and the loop continues.
    try:
        response = requests.get(enquiry_input, timeout=120)
        response.raise_for_status()
        hydrocron_response = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"Hydrocron request failed for lake {feature_id}: {exc}. Skipping.")
        continue

    if not isinstance(hydrocron_response, dict):
        print(f"Hydrocron returned an unexpected response for lake {feature_id}; skipping.")
        continue

    results = hydrocron_response.get("results")
    if not isinstance(results, dict):
        print(f"Hydrocron returned no result table for lake {feature_id}; skipping.")
        continue

    extracted_data = results.get(output, "")
    if not isinstance(extracted_data, str) or not extracted_data.strip():
        print(f"Hydrocron returned no LakeSP records for lake {feature_id}; skipping.")
        continue

    try:
        df = pd.read_csv(StringIO(extracted_data))
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Hydrocron CSV could not be read for lake {feature_id}: {exc}. Skipping.")
        continue

    if df.empty:
        print(f"Hydrocron returned an empty LakeSP table for lake {feature_id}; skipping.")
        continue

    missing_columns = [
        column
        for column in INPUT_SCHEMA["LakeSP time series"]["required_columns"]
        if column not in df.columns
    ]
    if missing_columns:
        print(
            f"Hydrocron response for lake {feature_id} is missing required "
            f"columns {missing_columns}; skipping."
        )
        continue

    
    # ==============Apply the HALF filter chain below==================
    """
    LakeSP time series preprocessing
    """
    # Add index_col to preserve the original df ordering/reproducibility and later label outliers.
    df = df.copy() # Make df an independent DataFrame and eliminate the SettingWithCopyWarning
    df['index_col'] = range(len(df))
    
    # Drop records that cannot be used by HALF: time and WSE must both be valid.
    # Invalid WSE fill values are removed rather than converted to NaN because the
    # smoothing filters and outlier tests require finite WSE values.
    df = df.loc[
        (df["time"] != fill_float) &
        (df["wse"] != fill_float) &
        df["time"].notna() &
        df["wse"].notna()
    ].copy()

    # Skip lakes for which Hydrocron returned no usable time/WSE records.
    if df.empty:
        print(f"No valid LakeSP time/WSE records remain for lake {feature_id}; skipping.")
        continue

    # Mask out invalid values for filtering metrics
    df.wse_u    = df.wse_u.mask(df.wse_u  == fill_float, np.nan)
    df.wse_std  = df.wse_std.mask(df.wse_std == fill_float, np.nan)
    df.xtrk_dist= df.xtrk_dist.mask(df.xtrk_dist == fill_float, np.nan)

    # Initialize the observation-level filter flag: 
    # 1 = retained; 0 = removed.
    df['filter_flag'] = 1

    # Ensure chronological order
    df = df.sort_values('time', kind='mergesort')  # Sort df by time to be safe (mergesort keeps relative order for ties).
    # - "time" is measurement in seconds in the UTC time scale since 1 Jan 2000 00:00:00 UTC.
    # - "time_str" is a string giving UTC time, in YYYY-MM-DDThh:mm:ssZ, where the Z suffix indicates UTC time.
    # Caution: "time" can have higher precision, and is therefore more precise than time_str.
    #           This leads to sometimes time_str values being duplicate. So always use "time" to sort the data.

    # Convert 'time' to yyyy-mm-dd hh:mm:ss (datetime64[ns], timezone-naive for simplicity). All SP time is based on UTC consistently.
    df['datetime'] = pd.to_datetime(df['time'], unit='s', origin=pd.Timestamp('2000-01-01 00:00:00')) # Timezone-naive UTC for consistency with the rest of the workflow.
    #Note: this result will be equivalent to time_str in LakeSP, but more precise (e.g., nanosecond precision) than time_str.
    #      Avoid using time_str for sorting because it can be less precise.

    # Initialize df_eval by dropping duplicate timestamps, if any
    df_eval = df.drop_duplicates(subset='time', keep='first').copy()
    # Duplicate high-precision timestamps are removed to avoid repeated evaluations.
    # df_eval is the candidate observation set used by the filtering workflow.
    



    """
    Step 1 — calibrate lake-specific heuristic quality thresholds.

    The built-in LakeSP quality flags are not used here as the final filter.
    Instead, they identify a conservative, high-confidence training subset 
    (based on user-defined conservative_SQL) from which lake-specific thresholds 
    are calibrated.  The calibrated thresholds are then applied to the full 
    LakeSP record to build the heuristic baseline.   

    Calibrated diagnostic variables
    -------------------------------
    wse_std_thr_cal
        Calibrated upper threshold for within-polygon WSE variability ('wse_std', metres).
        Larger values may indicate mixed water/land pixels, false water
        detections, or spatially inconsistent WSE retrievals.
    wse_u_thr_cal
        Calibrated upper threshold for LakeSP WSE uncertainty ('wse_u', metres), which
        summarizes random and systematic uncertainty contributions from the
        interferogram and LakeSP processing chain.
    xtrk_dist_thr_cal
        Calibrated lower threshold for absolute cross-track distance ('abs(xtrk_dist)',
        metres). It is calibrated but is not applied in the current release
        because small absolute cross-track distance is not a reliable outlier
        indicator for every lake.

    Notes
    -----
    - 'dark_frac' is not applied directly because it is not part of the WSE
      computation itself.
    - The bitwise quality flag is not applied directly in this release; the
      updated 'quality_f' classes available in newer CRIDs are used to define
      the conservative calibration subset.
    - 'calibrate_heuristic_thresholds' supports stratification by CRID scenario,
      orbit pass, and ice condition to accommodate evolving LakeSP versions and
      pass-dependent lake-observation geometry.
    """
    # Calibrate lake-specific heuristic thresholds from the conservative high-quality subset
    # (function calibrate_heuristic_thresholds)
    # Inputs:
    #   df_eval: one-lake LakeSP observation table after preprocessing and duplicate-time removal.
    #   conservative_SQL: query expression defining the high-quality subset used for threshold calibration.
    #   by_crid_scenario: whether to calibrate thresholds separately by CRID scenario for [wse_std, wse_u, xtrk_dist].
    #   by_pass_id: whether to calibrate thresholds separately by SWOT pass for [wse_std, wse_u, xtrk_dist].
    #   by_ice: whether to calibrate thresholds separately by ice condition for [wse_std, wse_u, xtrk_dist].
    # Output:
    #   df_heuristic_thresholds: calibrated threshold table with *_thr_cal fields and grouping_scheme.
    # See detailed parameter definitions in calibrate_heuristic_thresholds() in half_v1_0_functions.py.
    
    # Define conservative_SQL as the criteria of high-quality LakeSP subset, from which thresholds are calibrated.
    # Selection criteria are broadly aligned with CNES baseline standards (Claire Pottier and Roger Fjørtoft,
    # on behalf of the HR Cal/Val Team, SWOT ST Meeting 2024).
    # quality_f:
    #     - Prior to version C2, observations flagged as 1 (bad) are excluded.
    #     - Versions C2 and D0: observations flagged as 2 (degraded) or 3 (bad) are excluded.
    #     - Caution: Using quality_f = 0 prior to PIC2 may over-reject observations.
    # xovr_cal_q:
    #     - Only good observations (0) are retained.
    # ice_clim_f:
    #     - Observations with full ice cover (2) are considered ice-covered.
    #     - Observations with partial/possible ice cover (1) are retained in ice-free to avoid under-detection.
    # Ice coverage conditions will be considered in a structured way in function "calibrate_heuristic_thresholds".
    # Use:
    # (1) xovr_cal_q < 1, and
    # (2) either
    #     quality_f < 1 (for any crid), or
    #     quality_f == 1 and crid ends with "C2" or "Dx" (PID0, PGD0).
    new_version_suffix = '(crid.str.endswith("C2") | crid.str.match(".*D.$"))'
    conservative_SQL = (
        f'((xovr_cal_q < 1) & ('
        f'   (quality_f < 1)'
        f' | ((quality_f == 1) & {new_version_suffix})'
        f'))'
    )

    # Diagnostic variables for threshold calibration: 
    #  - wse_std (in m): Lake surface flatness, assuming errors lead to large wse_std
    #  - wse_u (in m): Algorithm processing uncertainty, assuming errors lead to large wse_u
    #  - xtrk_dist (in m, abs. 0-75000 m): Distance of lake polygon centroid from the nadir track, assuming errors lead to small |xtrk_dist|
    # The calibration of each variable can be stratified to pass, version, and ice-condition.
    #    
    # Note for pre-Version-D2 LakeSP products:
    #    A lake split by the [-10 to 10] km central track may also have an |xtrk_dist value| < 10 km,
    #    Yet its WSE value can be useful, as shown in many cases, such as 4340980733 (TGD) and 4530143033.
    #    So, an observation whose |xtrk_dist| < 10 km may not mean poor WSE,
    #    and we cannot just exclude observations whose |xtrk_dist| < 10 km, either.
    #    In other words, while wse_std represents water consistency and wse_u represents algorithm uncertainty, both indicating quality,
    #    small |xtrk_dist| may not always represent poor quality, so it is not used by default here.
    #    Note: This issue of xtrk_dist has been addressed in LakeSP VD2.
    #
    # Caution: Applying pass and version groupings/stratification for wse_std or wse_u can help avoid over-rejection,
    #          but overly small groups may also make thresholds unstable.
    #          Pass grouping is theoretically needed for xtrk_dist as lake position varies in different passes.
    df_heuristic_thresholds = calibrate_heuristic_thresholds(df_eval, conservative_SQL,
                                       by_crid_scenario = [False, False, False], # boolean sequence for [wse_std, wse_u, xtrk_dist]
                                       by_pass_id = [False, False, True],        # boolean sequence for [wse_std, wse_u, xtrk_dist]
                                       by_ice = [True, True, True])              # boolean sequence for [wse_std, wse_u, xtrk_dist]
    # Note: output df_heuristic_thresholds contains:
    #       ['lake_id', 'crid_scenario', 'pass_id', 'ice_condition',
    #        'wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal',
    #        'grouping_scheme']
    #       - crid_scenario depends on the last two digits in crid, e.g., C0, C2, and D0.
    #       - ice_condition has three scenarios: "ice-free", "ice-covered", and "both".
    # Prefer using by_crid_scenario = False, e.g., lake_id 7720028943,
    # because PIC2 data was still limited at the time of initial analysis.
    
    

    """
    Step 2 — apply the customized filter to produce the retained LakeSP WSE time series.

    The calibrated heuristic thresholds first extract a preliminary baseline
    subset from the LakeSP record.
    If low-pass filtering is enabled, a smoothing model is fit to that baseline and
    residuals are evaluated iteratively to remove remaining outliers.

    Round 1 uses the stricter residual threshold;
    optional Round 2 uses the more permissive threshold to reduce over-rejection.

    The following calls implement the filtering hierarchy used in the manuscript:
    strict HALF → lenient HALF → Tukey fallback → no filter.
    """
    # Apply HALF filtering (function apply_customized_filter).
    # Inputs:
    #   df_eval: preprocessed one-lake LakeSP observation table to be filtered.
    #   df_heuristic_thresholds: calibrated threshold table from calibrate_heuristic_thresholds().
    #   *_threshold_bounds: lower/upper bounds for applied wse_std, wse_u, and abs(xtrk_dist) thresholds.
    #   *_ice_min: minimum applied wse_std/wse_u thresholds for ice-affected observations.
    #   allow_major_gap: whether major temporal gaps are permitted in the heuristic baseline.
    #   max_temporal_gap, min_temporal_range: temporal-support criteria for this attempt.
    #   rules_for_ice_*_data: per-variable threshold-source rules for [wse_std, wse_u, xtrk_dist].
    #   gauge_df, plot_period: optional inputs for diagnostic plotting only.
    #   low-pass settings: controls for smoothing type, residual tests, round-2 filtering, and plotting.
    # Outputs:
    #   df_filtered: retained LakeSP observations from this HALF attempt.
    #   n_while_filtered: [round-1, round-2] iteration counts/status codes.
    #   filter_status: attempt outcome: success, fail, heuristic baseline, or no data.
    #   threshold_summary: calibrated and bounded threshold table for this attempt.
    #   observation_thresholds: observation-level threshold source, calibrated, bounded, and applied values.
    # See detailed parameter definitions in apply_customized_filter() in half_v1_0_functions.py.
    
    # Reasoning for threshold bounds:
    # Calibrated thresholds are data-adaptive, but they can occasionally be too strict, too permissive, 
    # or undefined when the conservative calibration subset is sparse or unreliable. 
    # So, empirical bounds below offer safeguard limits: they cap unrealistically permissive thresholds, 
    # relax overly strict thresholds, and supply fallback values when calibrated thresholds are missing.  
    
    # Some empirical notes: 
    # - max bound for wse_std_threshold = 3 or 5. 3 seems more conservative and accurate, but 5 m may allow more observations in.
    # - min bound for wse_std_threshold = 0
    # - max bound for wse_u_threshold = 0.5. This conservative value may benefit from future sensitivity testing.
    # - min bound for wse_u_threshold = 0 or 0.1 The value of 0.1 m is overall consistent with the science requirement.
    # Note: Not capping minimum wse_u_threshold may lead to over-rejection.
    #       The upper bounds may lack strong theoretical justification and may require future revision.
    #       For reference, a wse_std_threshold exceeding 3-5 m is unlikely:
    #          - Lake Tefé, a ria lake with strong water-surface gradient during the dry season, shows a maximum 4–5 m WSE range.
    #          - Backwater superelevations for Lake Selingué and Shardara Reservoir are typically <3 m.
    
    # - maximum bound for xtrk_dist_threshold = 75000, which is the valid max.
    # - minimum bound for xtrk_dist_threshold = 0, which is the valid min.
    # Note: Do not enforce xtrk_dist for LakeSP versions before Version D2.

    # More filtering rules:
    # Based on experimentation, it seems by_crid_scenario = True (stratification by LakeSP version) does lead to some degree
    #    of over-rejection, but they are also often reasonable.
    # Not setting wse_u_threshold min bound seems to improve noise: e.g., good observations in PIC2 often have wse_u < 0.1.
    #    Overall, ice leads to larger wse_u due to increased interferometric noise.
    #    So, use 0 min bound for wse_u_threshold, but set up a higher min bound for freeze-up condition only (wse_u_ice_min).
    # wse_std_ice_min = 3 # Elevate min wse_std threshold for ice/freeze-up condition to allow for data
    # wse_u_ice_min = 0.1 # Elevate min wse_u threshold for ice/freeze-up conditions to allow for data. 0.1 m is also set to include valid extremes.

    # Select which calibrated threshold is applied to each observation type.
    # Each list has three entries corresponding to:
    #     [wse_std, wse_u, xtrk_dist]
    #
    # Allowed rule values:
    #     "ice-free"    : apply the threshold calibrated from ice-free observations
    #     "ice-covered" : apply the threshold calibrated from ice-covered observations
    #     "both"        : apply the threshold calibrated from all observations pooled together
    #     "not apply"   : do not use this variable as a filtering criterion
    #
    # The first list is used for observations whose SP ice condition is ice_clim_f < 2. 
    # The second list is used for observations whose SP ice condition is ice_clim_f == 2.
    #
    # Current default:
    #     - wse_std and wse_u are filtered using ice-free calibrated thresholds for
    #       both ice-free and ice-covered observations.
    #     - xtrk_dist is not used for filtering because small |xtrk_dist| (before version D2)
    #       does not consistently indicate poor WSE quality for lakes spanning or near nadir.
    rules_for_ice_free_data    = ['ice-free', 'ice-free', 'not apply']
    rules_for_ice_covered_data = ['ice-free', 'ice-free', 'not apply']
    # Note: These rules also determine which threshold-summary rows receive *_thr_bound.
    # Rows that are calibrated but not selected by these rules keep *_thr_cal but
    # have blank *_thr_bound in the threshold-summary output.

    # Execute the filter function
    # Attempt 1: strict.
    # allow_major_gap = 'no' recommended, which enables constant full control throughout the time series.
    filter_attempt = '1_strict' # the most strict filtering        
    (
         df_filtered,             # Filtered LakeSP observations returned by this HALF attempt.
         n_while_filtered,        # [round-1, round-2] iteration counts/status codes.
         filter_status,           # Filtering status for this attempt: success, fail, heuristic baseline, or no data.
         threshold_summary,       # Attempt-level calibrated and bounded threshold table.
         observation_thresholds,  # Observation-level threshold sources, calibrated, bounded, and applied values.
    ) = apply_customized_filter(df_eval, df_heuristic_thresholds,
                                         # Bound overrides for applied df_heuristic_thresholds
                                         wse_std_threshold_bounds = [0, 3],
                                         wse_u_threshold_bounds = [0, 0.5],
                                         xtrk_dist_threshold_bounds = [0, 75000], # retained for traceability; xtrk_dist is not applied by default

                                         # Ice overrides for applied thresholds on ice-affected rows
                                         wse_std_ice_min = 3, #use 0 if not applying the override
                                         wse_u_ice_min = 0.1, #use 0 if not applying the override

                                         allow_major_gap = 'no', # 'yes/no', to indicate if gap in the filtered time series is allowed.
                                         max_temporal_gap = strict_max_temporal_gap_days, # Maximum temporal gap (days) for strict filtering
                                         min_temporal_range = strict_min_temporal_range_days, # Minimum temporal range/span (days) for strict filtering

                                         # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
                                         # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
                                         rules_for_ice_free_data = rules_for_ice_free_data,
                                         rules_for_ice_covered_data = rules_for_ice_covered_data,

                                         gauge_df = None, # Gauge data are not used in this filtering-only script.
                                         plot_period = [start_time, end_time], # Start and end time for diagnostic plotting.

                                         apply_low_pass_filter = apply_low_pass_filter,
                                         evaluating_at_full_data = evaluating_at_full_data,
                                         r2_filter = r2_filter,
                                         filter_type = filter_type,
                                         z_score_thresholds = z_score_thresholds,
                                         maximum_residual_spreads = maximum_residual_spreads,
                                         show_filtering_evolution = show_filtering_evolution)
    
    # In case of no valid SWOT data for this lake (df_eval/df is empty)
    if filter_status == 'no data': # n_while_r2 is always -9 when n_while is -9.
        filter_attempt = 'no data' # Overwrite 'strict' by 'no data'

    # Attempt 2: lenient
    if filter_status == 'fail':
        # In the strict attempt above, allow_major_gap = "no" means that the filtering attempt fails
        # if the heuristic baseline (df_apply) contains any temporal gap longer than max_temporal_gap.
        #
        # df_eval = the current candidate LakeSP observations being evaluated.
        # df_apply = the heuristic-baseline subset generated inside the function after applying
        # calibrated thresholds.

        # Here, allow_major_gap = "yes" is more lenient. A large gap in df_apply no longer causes
        # immediate failure. Instead, observations in df_eval that fall inside those large df_apply gaps
        # are removed before residual evaluation.
        #
        # Rationale:
        #   df_apply is used to fit the low-pass smooth curve. If df_apply has a long temporal gap,
        #   the smooth curve inside that gap is mostly an interpolation across unsupported time.
        #   Residuals computed there may be misleading and should NOT be used to decide whether
        #   observations are valid or outliers.
        #
        # Tradeoff:
        #   This mode can recover useful local continuous segments for lakes that fail the strict
        #   temporal-continuity requirement, but the resulting filtered time series may be more fragmented.
        #   Users can apply an additional post-filtering temporal-coverage criterion, such as a
        #   maximum allowed output gap (e.g., 120 days as in Trudel et al. (2026)), to decide whether the result
        #   is suitable for their application.

        # Execute the customized filter with more lenient parameters
        filter_attempt = '2_lenient'  # More lenient filtering
        (
            df_filtered,             # Filtered LakeSP observations returned by this HALF attempt.
            n_while_filtered,        # [round-1, round-2] iteration counts/status codes.
            filter_status,           # Filtering status for this attempt: success, fail, heuristic baseline, or no data.
            threshold_summary,       # Attempt-level calibrated and bounded threshold table.
            observation_thresholds,  # Observation-level threshold sources, calibrated, bounded, and applied values.
        ) = apply_customized_filter(df_eval, df_heuristic_thresholds,
                                             # Bound overrides for applied df_heuristic_thresholds
                                             wse_std_threshold_bounds = [0.5, 5],
                                             wse_u_threshold_bounds = [0.1, 0.5],
                                             xtrk_dist_threshold_bounds = [0, 75000], # retained for traceability; xtrk_dist is not applied by default

                                             # Ice overrides for applied thresholds on ice-affected rows
                                             wse_std_ice_min = 5, #use 0 if not applying the override
                                             wse_u_ice_min = 0.1, #use 0 if not applying the override

                                             allow_major_gap = 'yes', # 'yes/no', to indicate if gap in the filtered time series is allowed.
                                             max_temporal_gap = lenient_max_temporal_gap_days, # Maximum temporal gap (days) for lenient filtering
                                             min_temporal_range = lenient_min_temporal_range_days, # Minimum temporal range/span (days) for lenient filtering

                                             # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
                                             # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
                                             rules_for_ice_free_data = rules_for_ice_free_data,
                                             rules_for_ice_covered_data = rules_for_ice_covered_data,

                                             gauge_df = None, # Gauge data are not used in this filtering-only script.
                                             plot_period = [start_time, end_time], # Start and end time for diagnostic plotting.

                                             apply_low_pass_filter = apply_low_pass_filter,
                                             evaluating_at_full_data = evaluating_at_full_data,
                                             r2_filter = r2_filter,
                                             filter_type = filter_type,
                                             z_score_thresholds = z_score_thresholds,
                                             maximum_residual_spreads = maximum_residual_spreads,
                                             show_filtering_evolution = show_filtering_evolution)

    # Attempt 3: baseline Tukey (IQR)
    if filter_status == 'fail':

        filter_attempt = '3_tukey' # Simple baseline Tukey filter
        # This fallback attempt is intended only as a conservative outlier screen and is not recommended for phenology characterization.
        # Here, the previous customized filter does not apply, and instead we use a simple baseline_tukey method

        # Define the baseline condition as a boolean mask
        baseline_SQL = '(quality_f == 0) & (xovr_cal_q == 0) & (ice_clim_f < 2)'

        # Remove remaining isolated extreme outliers using Tukey method (IQR method)
        df_filtered, n_while_filtered, filter_status = apply_baseline_tukey_filter(df_eval, baseline_SQL,
                                                                       multiplier = 3,
                                                                       lower_q = 0.25,
                                                                       upper_q = 0.75,
                                                                       iteration_n=5)
        n_while_filtered = [n_while_filtered, -2] # Match the [round 1, round 2] iteration-count convention used by other attempts.

    # No filtering attempt succeeded for this lake; df_filtered remains empty.
    if filter_status == 'fail':
        filter_attempt = '4_none'
        n_while_filtered = [0, 0]

    
    # Finalize threshold outputs only after the strict -> lenient -> Tukey -> no-filter hierarchy 
    # has determined the final attempt for this lake.
    #
    # For final strict or lenient HALF results, retain the bounded and applied
    # thresholds which are already returned by the corresponding apply_customized_filter() call.
    #
    # For final Tukey or no-filter results, retain calibrated thresholds for traceability 
    # but blank bounded/applied fields because heuristic thresholds were not the final filtering method.
    threshold_bound_columns = [
        'wse_std_thr_bound',
        'wse_u_thr_bound',
        'xtrk_dist_thr_bound',
    ]
    threshold_app_columns = [
        'wse_std_thr_app',
        'wse_u_thr_app',
        'xtrk_dist_thr_app',
    ]

    if filter_attempt in {'3_tukey', '4_none'}:
        # Rebuild the summary from the original calibrated threshold table.
        threshold_summary = df_heuristic_thresholds.copy()
        for column in threshold_bound_columns:
            threshold_summary[column] = np.nan

        # Preserve source and calibrated values at the observation level, but
        # blank bounded/applied values because they were not used by the final
        # Tukey or no-filter result.
        for column in threshold_bound_columns + threshold_app_columns:
            if column in observation_thresholds.columns:
                observation_thresholds[column] = np.nan

    # No threshold or observation output is appended when no usable LakeSP input
    # was available for the lake.
    if filter_attempt == 'no data':
        print(f"No threshold outputs were created for lake {feature_id}; skipping.")
        continue

    # Record the final hierarchy outcome on every threshold-summary and
    # observation-threshold row. In df_Hydrocron, filter_attempt is also stored
    # directly as lake-level metadata on every original observation row.
    threshold_summary = threshold_summary.copy()
    observation_thresholds = observation_thresholds.copy()
    threshold_summary['filter_attempt'] = filter_attempt
    observation_thresholds['filter_attempt'] = filter_attempt

    n_while = n_while_filtered[0]
    n_while_r2 = n_while_filtered[1]




    """
    Step 3 — Cross-pass bias correction (intra-cycle cross-pass WSE adjustment)

    Logic: For lakes spanning multiple SWOT orbit passes, WSE values within the same orbit cycle may show substantial
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
    # Initialize the cross-pass adjustment flag.
    #   0 = no intra-cycle cross-pass WSE adjustment was applied
    #   1 = at least one retained WSE value was modified by the adjustment
    intra_cycle_flag = 0

    # Apply optional intra-cycle cross-pass WSE adjustment.
    # This step is intended for lakes observed by multiple SWOT passes within the
    # same cycle. When pass-specific WSE offsets are detected, sp_cycle_adjustment()
    # estimates and removes those offsets while preserving the retained observations
    # whenever possible.
    if execute_intra_cycle_adjustment == 'no':
        # No cross-pass adjustment: use the retained original WSE values directly.
        df_filtered["wse_adjusted"] = df_filtered["wse"]

    else:        
        # Apply optional intra-cycle cross-pass WSE adjustment (function sp_cycle_adjustment).
        # Input:
        #   df_filtered: retained LakeSP observations after HALF filtering.
        # Outputs:
        #   df_option1: cycle-averaged WSE series, not used here.
        #   df_option2: best-pass-only WSE series, not used here.
        #   df_filtered: retained observations with wse_adjusted added or updated.
        # See detailed parameter definitions in sp_cycle_adjustment() in half_v1_0_functions.py.
        _, _, df_filtered = sp_cycle_adjustment(df_filtered)
        # Note: depending on the internal checks in sp_cycle_adjustment(), this step may
        # leave WSE unchanged if cross-pass inconsistency is not substantial.
        # Applying cross-pass bias correction may also remove additional outliers.

        # Record whether the adjustment changed any retained WSE value for this lake.
        if (
            df_filtered["wse"].notna()
            & df_filtered["wse_adjusted"].notna()
            & (df_filtered["wse"] != df_filtered["wse_adjusted"])
        ).any():
            intra_cycle_flag = 1



    """
    Label filtered result back to the original df
    """
    # Label survivals (non-outlier LakeSP observations) back to the original df through index_col.
    # filter_flag:
    #   1 means good or retained observation
    #   0 means outlier
    df.loc[~df['index_col'].isin(df_filtered['index_col']), 'filter_flag'] = 0
    # Note: df.query('filter_flag != 0') will be the final LakeSP observations that survived the filtering.
    # It equals df_filtered in size, but df.query('filter_flag != 0') keeps the original attribute structure.

    # Left-join the 'wse_adjusted' column from df_filtered into df, based on the unique key index_col.
    df = df.merge(
        df_filtered[['index_col', 'wse_adjusted']],  # only bring the column(s) that are shared (retained after filtering)
        on='index_col',
        how='left'  # keep all rows from df, fill unmatched ones with NaN
        ) # Now df will have a new column "wse_adjusted".
    # Note:
    #    After joining, wse_adjusted is valid only for retained observations; removed observations are filled with NaN.
    #    wse_adjusted will equal wse if no cycle adjustment is needed.
    #    Therefore, wse_adjusted is the preferred column for final HALF-filtered WSE.

    # Optionally append observation-level threshold traceability fields to the main LakeSP output. 
    # index_col is unique within this lake and is therefore used as the per-lake merge key. 
    # Note: Rows excluded before threshold evaluation, including duplicate 
    # high-precision timestamps removed from df_eval (see section "LakeSP time series preprocessing"), 
    # remain in df but with blank threshold-detail fields.
    if append_threshold_details_to_output == 'yes':
        threshold_detail_columns = [
            'index_col',
            'row_ice_condition',
            'wse_std_thr_source',
            'wse_std_thr_cal',
            'wse_std_thr_bound',
            'wse_std_thr_app',
            'wse_u_thr_source',
            'wse_u_thr_cal',
            'wse_u_thr_bound',
            'wse_u_thr_app',
            'xtrk_dist_thr_source',
            'xtrk_dist_thr_cal',
            'xtrk_dist_thr_bound',
            'xtrk_dist_thr_app',
        ]
        available_threshold_columns = [
            column
            for column in threshold_detail_columns
            if column in observation_thresholds.columns
        ]

        if 'index_col' in available_threshold_columns:
            df = df.merge(
                observation_thresholds[available_threshold_columns],
                on='index_col',
                how='left',
                validate='one_to_one',
            )

    # Store lake-level filtering metadata on every observation row for convenient export.
    df[['n_while', 'n_while_r2', 'filter_attempt', 'intra_cycle_flag']] = [n_while, n_while_r2, filter_attempt, intra_cycle_flag]
    
    # Append the processed observations and final-attempt threshold summary to
    # their per-lake output lists. The lists are concatenated once after the loop.
    df_Hydrocron_parts.append(df)
    df_threshold_summary_parts.append(threshold_summary)
    







    """
    At this point, the filtering process is complete.    
    The following section plots the raw and filtered LakeSP time series.
    """







   
    # Set up the time series plot.
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, linewidth=0.5, zorder=1)

    # Plot the raw and HALF-filtered time series.
    # Plot raw WSE time series (with error bars)
    ax.errorbar(df.datetime, df.wse, yerr=df.wse_u, label='raw SP', marker='o',
            color=(0.6, 0.6, 0.6), markersize=4, capsize=3, linestyle='', zorder=2)

    # Quality flag measurements
    bad_q      = f'((quality_f == 1) & ~{new_version_suffix}) | ((quality_f == 3) & {new_version_suffix})'
    degrad_q   = f'(quality_f == 2) & {new_version_suffix}'
    suspect_q  = f'(quality_f == 1) & {new_version_suffix}'
    d_bad     = df.query(bad_q, engine="python")
    d_degrad  = df.query(degrad_q, engine="python")
    d_suspect = df.query(suspect_q, engine="python")
    ax.plot(d_bad.datetime, d_bad.wse,
        label='quality_f = bad', marker='s', linestyle='', markersize=7,
        markerfacecolor='none', markeredgecolor='red')
    ax.plot(d_degrad.datetime, d_degrad.wse,
        label='quality_f = degraded', marker='s', linestyle='', markersize=7,
        markerfacecolor='none', markeredgecolor=(0, 1, 0))
    ax.plot(d_suspect.datetime, d_suspect.wse,
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
    
    # Mark observations whose wse_std exceeds the observation-level calibrated threshold.
    # This is diagnostic only; final applied thresholds may differ after safeguard bounds 
    # and ice-condition overrides.
    if "wse_std_thr_cal" in df.columns:
        mask_wse_std_exceed_cal = (
            df["wse_std"].notna()
            & df["wse_std_thr_cal"].notna()
            & (df["wse_std"] > df["wse_std_thr_cal"])
        )    
        ax.plot(
            df.loc[mask_wse_std_exceed_cal, "datetime"],
            df.loc[mask_wse_std_exceed_cal, "wse"],
            label="wse_std > cal. thr.",
            marker="o", linestyle="", markersize=7, 
            markerfacecolor="none", markeredgecolor="yellow",
        )

    # Only for visual reference.
    mask = (df["xtrk_dist"].abs() < 10000) | (df["xtrk_dist"].abs() > 60000)
    ax.plot(df[mask].datetime, df[mask].wse,
            label='xtrk_dist out [10, 60]km', marker='s', linestyle='', markersize=15,
            markerfacecolor='none', markeredgecolor='black')

    # Plot filtered result (use wse_adjusted to account for possible intra-cycle adjustment)
    ax.errorbar(df_filtered.datetime, df_filtered.wse_adjusted, df_filtered.wse_u,
           label='HALF', color='black', marker='o',
           markersize=4, capsize=3, linestyle='--')

    # Format axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    fig.autofmt_xdate()
    ax.set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))
    if len(df_filtered) >= 1 and df_filtered['wse'].notna().any():
        wse_min = np.nanmin(df_filtered['wse'])
        wse_max = np.nanmax(df_filtered['wse'])
        range_wse = wse_max - wse_min
        padding = range_wse * 2 if range_wse > 0 else 0.5
        plt.ylim(wse_min - padding, wse_max + padding)
    elif len(df) >= 1 and df['wse'].notna().any():
        wse_min = np.nanmin(df['wse'])
        wse_max = np.nanmax(df['wse'])
        padding = (wse_max - wse_min) * 0.05 if wse_max > wse_min else 0.5
        plt.ylim(wse_min - padding, wse_max + padding)

    # Axis labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('WSE (m)', fontsize=12)
    ax.set_title('Lake ID ' + str(feature_id) + ' Time Series Plot. Filter: ' + filter_type)

    # Display the marker definitions and save the diagnostic plot.
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.savefig(
        os.path.join(
            work_dir,
            f'Attempt_{filter_attempt}_lakeID_{feature_id}_{filter_type}_{script_version}_{version_filename}.png'
        ),
        bbox_inches='tight'
    )
    if show_filtering_evolution == 'no':
        plt.close()  # free up memory



# Concatenate the per-lake output tables once after all lakes have been processed.
if df_Hydrocron_parts:
    df_Hydrocron = pd.concat(df_Hydrocron_parts, ignore_index=True)
else:
    df_Hydrocron = pd.DataFrame()

if df_threshold_summary_parts:
    df_threshold_summary = pd.concat(
        df_threshold_summary_parts,
        ignore_index=True,
    )
else:
    df_threshold_summary = pd.DataFrame(
        columns=[
            'lake_id', 'crid_scenario', 'pass_id', 'ice_condition',
            'wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal',
            'grouping_scheme', 'wse_std_thr_bound', 'wse_u_thr_bound',
            'xtrk_dist_thr_bound', 'filter_attempt',
        ]
    )


"""
Save final table outputs:
    df_Hydrocron
    df_threshold_summary    
"""
# Save the combined observation-level Hydrocron table with appended HALF outputs.
df_Hydrocron.to_csv(df_Hydrocron_output_csv, index=False)

# Save calibrated thresholds and final-attempt bounded thresholds.
df_threshold_summary.to_csv(threshold_summary_csv, index=False)

print(f"Saved HALF observation-level table: {df_Hydrocron_output_csv}")
print(f"Saved threshold-summary table: {threshold_summary_csv}")

if append_threshold_details_to_output == "yes":
    print("Observation-level threshold details were appended to the HALF observation-level table.")
else:
    print("Observation-level threshold details were not appended to the HALF observation-level table.")

print(f"Saved diagnostic plots to: {work_dir}")
print(
    "Processed lakes with output rows: "
    f"{df_Hydrocron['lake_id'].nunique() if not df_Hydrocron.empty else 0}"
)