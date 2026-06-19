"""
Heuristic Adaptive Lake Filter (HALF) validation workflow for SWOT LakeSP.

HALF is designed to remove outliers from SWOT LakeSP water surface elevation (WSE)
time series while preserving hydrologically meaningful temporal dynamics.

The workflow implemented here follows three primary steps:

    1. Calibrate lake-specific heuristic thresholds from a conservative
       high-quality LakeSP subset.
    2. Build a heuristic baseline and iteratively remove residual outliers using
       a configurable low-pass filter.
    3. Optionally correct cross-pass WSE offsets for lakes observed by multiple
       SWOT passes within the same cycle.

This script integrates both LakeSP filtering and gauge validation results
as reported in Trudel et al. (2026, in review).

It retrieves or loads LakeSP time
series, reads paired gauge records, applies HALF, computes validation metrics,
and writes lake-level and observation-level outputs.

Script name and version
-------
half_v1_0_validation.py: 
    Main script for running HALF with gauge validation
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

# Import customized functions from half_v1_0_functions.py
# See half_v1_0_functions.py for definitions and functionality.
from half_v1_0_functions import (
    calibrate_heuristic_thresholds,
    apply_customized_filter,
    apply_baseline_tukey_filter,
    compute_correlation,
    compute_mae,
    compute_median_residual,
    compute_variability_amplitude,
    compute_variability_p10_p90_range,
    compute_variability_std,
    convert_to_daily_series,
    sp_cycle_adjustment,
)

# =============================================================================
# User configuration and schema documentation
# =============================================================================
# This section contains three parts:
#
#   1. Documentation dictionaries:
#        INPUT_SCHEMA and OUTPUT_SCHEMA describe the expected inputs and outputs.
#        They are provided for user guidance and do not affect the script execution.
#
#   2. Editable runtime configuration:
#        RUN_CONFIG, FILTER_CONFIG, and VALIDATION_CONFIG control how the
#        validation workflow is executed.
#        Please edit these dictionaries before running the script if necessary.
#
#   3. Derived variables:
#        Scalar variables below the configuration dictionaries are created for
#        backward compatibility with the rest of the script.
#        These should generally not be edited directly.
#
# The default settings are configured to reproduce the validation results
# reported in Trudel et al. (2026), assuming the same input data are provided.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Documentation dictionaries: inputs and outputs
# -----------------------------------------------------------------------------
# These dictionaries document the required input tables, expected columns, and
# major output products. They are intended as in-script documentation for users.
# They do not affect the script execution.

INPUT_SCHEMA = {
    "LakeSP time series": {
        "source": "Hydrocron API or a local df_Hydrocron_<version>.csv file",
        "required_columns": [
            "lake_id", "time", "time_str", "wse", "wse_u", "wse_std",
            "xtrk_dist", "quality_f", "xovr_cal_q", "ice_clim_f",
            "cycle_id", "pass_id", "crid", "collection_shortname",
            "p_ref_area", "area_total",
        ],
        "notes": (
            "LakeSP time is seconds since 2000-01-01 00:00:00 UTC and is "
            "used for chronological sorting because it can be more precise "
            "than time_str."
        ),
    },
    "Gauge data": {
        "source": "Regional CSV/XLSX files listed in the gauge metadata setup",
        "required_output_columns_after_loading": ["gauge_datetime", "gauge_wse"],
        "notes": (
            "Gauge records are bias-aligned to LakeSP before validation because "
            "many source gauges use local or unknown vertical datums."
        ),
    },
    "Cycle lookup table": {
        "source": "CSV table mapping SWOT science-orbit cycles to UTC times",
        "default_relative_path": "SWOT_cycle_lookup_table.csv",
    },
}

OUTPUT_SCHEMA = {
    "df_lake_time_series": (
        "Stacked observation-level LakeSP table for all evaluated lakes. "
        "Includes original LakeSP fields, filter_flag, wse_adjusted, gauge "
        "matches where available, benchmark-filtered WSE columns, and iteration "
        "metadata."
    ),
    "df_lake_heuristic_thresholds": (
        "Lake-specific calibrated heuristic thresholds returned by "
        "calibrate_heuristic_thresholds() for wse_std, wse_u, and xtrk_dist. "
        "The table uses wse_std_thr_cal, wse_u_thr_cal, and "
        "xtrk_dist_thr_cal and may be stratified by CRID scenario, pass_id, "
        "and ice condition. This validation script does not append bounded or "
        "observation-level applied threshold details."
    ),
    "df_lake_stats": (
        "Lake-level validation metrics, retention rates, filtering attempt "
        "status, matched-observation counts, and variability diagnostics for "
        "raw LakeSP, HALF, and benchmark filters."
    ),
    "plots_dir": "Directory containing per-lake diagnostics and summary figures.",
}


# -----------------------------------------------------------------------------
# 2. Editable runtime configuration
# -----------------------------------------------------------------------------
# Please edit the three dictionaries below before running the validation workflow.
#
# RUN_CONFIG controls input paths, LakeSP version, time window, and output naming.
# FILTER_CONFIG controls the HALF filtering behavior.
# VALIDATION_CONFIG controls gauge comparison and variability metric settings.

RUN_CONFIG = {
    # LakeSP collection short name. Common values:
    #   - "SWOT_L2_HR_LakeSP_2.0" for Version C
    #   - "SWOT_L2_HR_LakeSP_D"   for Version D
    "collection_shortname": "SWOT_L2_HR_LakeSP_D",

    # Science-orbit cycle and UTC time window. The cycle-to-UTC lookup table is
    # available from the SWOT events/cycle table distributed by PO.DAAC/JPL:
    # https://podaac.jpl.nasa.gov/SWOT-events/SWOT_events.html#cycletabletop
    "start_cycle": 1,
    "start_time": "2023-07-21T05:33:45Z",
    "end_cycle": 32,
    "end_time": "2025-05-18T21:36:19Z",  # Start time of end_cycle + 1

    # Repository/work directory. Keep this as "." for a repository-relative run,
    # or replace it with an absolute path for a local run.
    "work_dir": ".",

    # Relative paths are interpreted from work_dir. Absolute paths are also valid.
    "cycle_lookup_csv": "SWOT_cycle_lookup_table.csv",

    # LakeSP retrieval mode:
    #   - "Hydrocron": query the Hydrocron API lake by lake.
    #   - "on-premise": read a previously saved df_Hydrocron_<version>.csv file.
    "SP_retrieval_method": "on-premise",

    # Validation script label used in output filenames.
    "script_version": "V1_0",

    # Cross-pass bias correction switch. Recommended default is "yes".
    "execute_intra_cycle_adjustment": "yes",
}

FILTER_CONFIG = {
    # If "yes" (recommended default), run both heuristic-baseline filtering and
    # iterative low-pass residual filtering.
    # If "no", only the heuristic-baseline subset is used.
    "apply_low_pass_filter": "yes",

    # If "yes", residual outlier tests are evaluated against the full LakeSP
    # record. If "no", they are evaluated only against the current candidate
    # retained set.
    "evaluating_at_full_data": "no",

    # Optional second-pass residual filtering. Round 1 is more aggressive;
    # round 2 is more lenient and may recover valid observations.
    "r2_filter": "yes",

    # Supported low-pass filters: lowess, wavelet, savgol, kalman, spline,
    # median, and hampel.
    "filter_type": "savgol",

    # Residual z-score thresholds for [round 1, round 2]. Examples:
    # 2.576 ≈ 99% two-tailed, 2.807 ≈ 99.5%, 2.967 ≈ 99.7%,
    # 3.291 ≈ 99.9%, 3.5 ≈ 99.95%.
    "z_score_thresholds": [2.576, 3.5],

    # Maximum relative residual-spread tolerances for [round 1, round 2].
    # Smaller values force stricter convergence; larger values preserve more data.
    "maximum_residual_spreads": [0.07, 0.05],

    # Diagnostic plotting switch for iteration evolution. Use with caution for
    # large validation batches because it can generate many figures.
    "show_filtering_evolution": "no",

    # Temporal-coverage criteria used by apply_customized_filter().
    # Strict mode fails when the heuristic baseline has gaps longer than the
    # maximum gap below. Lenient mode allows such gaps but avoids evaluating
    # residuals inside those unsupported periods.
    "strict_max_temporal_gap_days": 90,
    "strict_min_temporal_range_days": 365,
    "lenient_max_temporal_gap_days": 90,
    "lenient_min_temporal_range_days": 365,
}

VALIDATION_CONFIG = {
    # Residuals with an absolute value <= zero_resid_tol are excluded from
    # mean absolute error (MAE) and signed median residual calculations after
    # gauge datum alignment. This avoids counting self-calibrated zero residuals
    # as validation skill.
    "zero_resid_tol": 1e-7,
    "min_matched_obs_for_error_metrics": 1,

    # Minimum number of raw LakeSP-gauge matched pairs required before a lake is
    # considered usable for gauge validation. This check is applied before
    # zero-residual exclusion.
    "min_raw_gauge_matches": 2,

    # Daily variability metrics are computed only if the shared raw LakeSP-gauge
    # daily overlap window spans at least this many days.
    "min_daily_span_days": 365,

    # Gauge interpolation excludes the interior of major gaps of this size or
    # larger, so daily metrics do not bridge long missing gauge periods.
    "major_gauge_gap_days": 90,

    # Reference-lake selection used for validation summaries and figures.
    # The default keeps strict Attempt-1 lakes plus lenient Attempt-2 lakes that
    # still satisfy the post-filtering temporal coverage criteria below.
    "reference_lake_selection": "strict_plus_gap_checked_lenient",
    "reference_min_retained_span_days": 365,
    "reference_max_retained_gap_days": 120,
}


# -----------------------------------------------------------------------------
# 3. Derived variables used by the remainder of the script
# -----------------------------------------------------------------------------
# The variables below are generated from the configuration dictionaries above.
# They are kept as scalar variables for compatibility with the original script
# structure. Please edit RUN_CONFIG, FILTER_CONFIG, or VALIDATION_CONFIG
# instead of modifying this block directly.

collection_shortname = RUN_CONFIG["collection_shortname"]
start_cycle = RUN_CONFIG["start_cycle"]
start_time = RUN_CONFIG["start_time"]
end_cycle = RUN_CONFIG["end_cycle"]
end_time = RUN_CONFIG["end_time"]
work_dir = os.path.abspath(RUN_CONFIG["work_dir"])
cycle_lookup_csv = RUN_CONFIG["cycle_lookup_csv"]
SP_retrieval_method = RUN_CONFIG["SP_retrieval_method"]
script_version = RUN_CONFIG["script_version"]
execute_intra_cycle_adjustment = RUN_CONFIG["execute_intra_cycle_adjustment"]

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

zero_resid_tol = VALIDATION_CONFIG["zero_resid_tol"]
min_matched_obs_for_error_metrics = VALIDATION_CONFIG[
    "min_matched_obs_for_error_metrics"
]
min_raw_gauge_matches = VALIDATION_CONFIG["min_raw_gauge_matches"]
min_daily_span_days = VALIDATION_CONFIG["min_daily_span_days"]
major_gauge_gap_days = VALIDATION_CONFIG["major_gauge_gap_days"]
reference_lake_selection = VALIDATION_CONFIG["reference_lake_selection"]
reference_min_retained_span_days = VALIDATION_CONFIG[
    "reference_min_retained_span_days"
]
reference_max_retained_gap_days = VALIDATION_CONFIG[
    "reference_max_retained_gap_days"
]

VERSION_FILENAME_BY_COLLECTION = {
    "SWOT_L2_HR_LakeSP_2.0": "vC",
    "SWOT_L2_HR_LakeSP_D": "vD",
}

if collection_shortname not in VERSION_FILENAME_BY_COLLECTION:
    valid = ", ".join(VERSION_FILENAME_BY_COLLECTION)
    raise ValueError(
        f"Unsupported collection_shortname={collection_shortname!r}. "
        f"Expected one of: {valid}."
    )

version_filename = VERSION_FILENAME_BY_COLLECTION[collection_shortname]

# Directory for per-lake diagnostics and summary figures.
plots_dir = os.path.join(work_dir, f"plots_{script_version}_{version_filename}")
os.makedirs(plots_dir, exist_ok=True)

if not os.path.isabs(cycle_lookup_csv):
    cycle_lookup_csv = os.path.join(work_dir, cycle_lookup_csv)

cyc = pd.read_csv(cycle_lookup_csv)


"""
Validation metadata setup

Summary of the validation gauge data:
As of 06/17/2026, we have collected the following gauge data.

Region	        lake_count	Frequency	            Data sources
Canada	        282	        Hourly	                ECCC, CEHQ, HQ, University of Sherbrooke
US	            284	        Hourly	                USGS, USBR, Harlan et al. (2026)
Norway	        232	        Hourly	                NVE
Switzerland 	29	        Hourly	                BAFU
China	        38	        Daily (inconsecutive)	China Water and Rain Information website (http://xxfb.mwr.cn/sq_dxsk.html)
Burkina Faso	1	        Sub-hourly to hourly	Specifically in situ set-up for SWOT cal/val (Girard et al. 2025)
Niger	        1	        Sub-hourly to hourly	AMMA-CATCH observatory (Girard et al. 2025)
Amazonia/Brazil	6	        Daily to hourly	        Instituto Mamirauá, Brazil
Ceará/Brazil	8	        Sub-hourly	Ceará       Funceme
Other Brazil    63	        Daily	                Operador Nacional do Sistema Elétrico (ONS; https://www.ons.org.br)
India           304	        Daily (inconsecutive)	Water Resources Information System of India (https://indiawris.gov.in/wris/#/).
Total	        1,248

See Trudel et al. (2026) for more details.

Note: Lakes may overlap across gauge data sources.
      Unique PLD IDs are deduplicated and cleaned through the sanity checks below.
"""
# Initialize a dataframe for tested lakes (i.e., potential validation lakes)
test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum']) # 'gauge_datum' is retained for backward compatibility but is not used below.

# Reservoirs in India
# Load the metadata CSV file
df_India_metadata = pd.read_csv(work_dir + "/gauge_data/India/Metadata_PLD_ID_with_WRIS_merged_after_manual_check.csv")
unique_pld_ids = df_India_metadata["lake_id (PLD_SWOT)"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
gauge_data_path = work_dir + "/gauge_data/India/Indian_reservoirs_daily_05272026-corrected.csv"
df_India = pd.read_csv(gauge_data_path)
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'reservoir_name': str(df_India_metadata[df_India_metadata["lake_id (PLD_SWOT)"] == unique_pld_ids[n]].iloc[0]["Reservoir Name"]),
        'reservoir_state': str(df_India_metadata[df_India_metadata["lake_id (PLD_SWOT)"] == unique_pld_ids[n]].iloc[0]["State"]),
        'gauge_source': 'India',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'India', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('India: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Reservoirs in China
# Load the original CSV file
gauge_data_path = work_dir + "/gauge_data/China/Daily_water_level_for_Chinese_reservoirs-corrected.csv"
df_China = pd.read_csv(gauge_data_path)
unique_pld_ids = df_China["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'China',
        'gauge_dir': gauge_data_path,
        'gauge_id': str(df_China[df_China["PLD_Lake_ID"] == unique_pld_ids[n]].iloc[0]["Name"]), #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('China: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Reservoirs in West Africa
# Load the original CSV file
gauge_data_path = work_dir + "/gauge_data/West-Africa/West_Africa_water_level_meters.csv"
df_wf = pd.read_csv(gauge_data_path)
unique_pld_ids = df_wf["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'West_Africa',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'West_Africa', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('West_Africa: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Lake Tefe in the Amazon
new_row = pd.DataFrame([{
    'lake_id': 6220321573, #PLD lake ID
    'gauge_source': 'Tefe',
    'gauge_dir': work_dir + '/gauge_data/Amazon/LakeTefe_WaterLevel_6220321573.xlsx',
    'gauge_id': 'Tefe',
    'gauge_datum': np.nan
}])
test_lakes = pd.concat([test_lakes, new_row], ignore_index=True) # Reindex the resulting DataFrame with a fresh, sequential index
print('Tefe: total number of unique PLD lakes with gauge data: 1')

# Other Floodplain lakes in the Amazon
# Load the original CSV file
gauge_data_path = work_dir + '/gauge_data/Amazon/Amazon_Floodplain_Lakes.csv'
df_amazon = pd.read_csv(gauge_data_path)
unique_pld_ids = df_amazon["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Amazon_floodplain',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'Amazon_floodplain', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('Amazon_floodplain: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Reservoirs in Ceara State, Brazil
# Load the original CSV file
gauge_data_path = work_dir + "/gauge_data/Ceara-Brazil/Ceara_reservoirs_in_situ.csv"
df_ceara = pd.read_csv(gauge_data_path)
unique_pld_ids = df_ceara["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Ceara_Brazil',
        'gauge_dir': gauge_data_path,
        'gauge_id': str(df_ceara[df_ceara["PLD_Lake_ID"] == unique_pld_ids[n]].iloc[0]["Name"]), #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('Ceara_Brazil: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Other reservoirs in Brazil
# Load the original CSV file
gauge_data_path = work_dir + "/gauge_data/Other-Brazil/Brazil_reservoirs_in_situ.csv"
df_Brazil = pd.read_csv(gauge_data_path)
unique_pld_ids = df_Brazil["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Other_Brazil',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'Other_Brazil', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field○
    }])
    #if new_row['lake_id'].iloc[0] != 6420530103: #gauge values are all the same
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('Other_Brazil: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Bafu gauge (Switzerland)
gauge_data_path = work_dir+'/gauge_data/Europe/bafu_in_situ.csv'
df_Bafu = pd.read_csv(gauge_data_path, sep=',', encoding='iso-8859-2')
unique_pld_ids = df_Bafu["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Bafu',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'Bafu', #String
        'gauge_datum': np.nan
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('Bafu (Switzerland): total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# NVE gauge (Norway)
gauge_data_path = work_dir+'/gauge_data/Europe/nve_in_situ.csv'
df_nve = pd.read_csv(gauge_data_path, sep=',', encoding='iso-8859-2')
unique_pld_ids = df_nve["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'nve',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'nve', #String
        'gauge_datum': np.nan
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('NVE (norway): total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# US
gauge_data_path = work_dir+'/gauge_data/US/US_in_situ.csv'
df_US = pd.read_csv(gauge_data_path, sep=',', encoding='iso-8859-2')
unique_pld_ids = df_US["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'US',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'US', #String
        'gauge_datum': np.nan
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('US: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Canada_GNSS
gauge_data_path = work_dir+'/gauge_data/Canada/Canada_in_situ_gnss.csv'
df_Canada = pd.read_csv(gauge_data_path, sep=',', encoding='iso-8859-2')
unique_pld_ids = df_Canada["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Canada_GNSS',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'Canada_GNSS', #String
        'gauge_datum': np.nan
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('Canada_GNSS: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))

# Canada_NO GNSS
gauge_data_path = work_dir+'/gauge_data/Canada/Canada_in_situ.csv'
df_Canada_UN = pd.read_csv(gauge_data_path, sep=',', encoding='iso-8859-2')
unique_pld_ids = df_Canada_UN["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs.
this_test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum'])
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Canada_UN',
        'gauge_dir': gauge_data_path,
        'gauge_id': 'Canada_UN', #String
        'gauge_datum': np.nan
    }])
    this_test_lakes = pd.concat([this_test_lakes, new_row], ignore_index=True)
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
print('Canada_No_GNSS: total number of unique PLD lakes with gauge data: ' + str(len(pd.unique(np.array(this_test_lakes['lake_id']) ))))


# Retrieve candidate validation lake IDs.
test_lakeIDs = np.array(test_lakes['lake_id'])

# Lakes (PLD IDs) to remove, whose gauge data seem problematic
lakes_to_exclude = [6420530103, 7421035423, 7520006182, 7520021732, 7120849393,2320145123,2320148843,2140016762,2320145652, \
                    7120837533, 7230294612, 7250511622, 8222590692, 7121168993, \
                    7320035782, 7320137783, 7320345753, 7320361003, 7320057002, 7250202733,\
                    7420084453, 7420932263, 7421075502, 7520007003, 7520021453, 7720003573,\
                    7720028943, 7730001053, 7830181563, 7310114012, 7830198863, \
                    4530124232,4530746652,4530045882,4530182732,4530360803,4530368712]
# Daily gauge data for a few Indian reservoirs are less complete, so monthly records are used instead:
Indian_lakes_only_for_monthly_gauge_data = [4530317302,4530373612,4530377962,4530746533,4530291552,4530322432,4530366332]
Indian_gauge_monthly = work_dir + "/gauge_data/India/Monthly Reservoir Level & Storage Timeseries data-corrected.csv"

test_lakeIDs =  [x for x in test_lakeIDs if x not in list(set(lakes_to_exclude))] # Remove lakes with known problematic gauge records.

# Retrieve unique values in test_lakeIDs while preserving their original order.
test_lakeIDs = pd.unique(test_lakeIDs)
# Note: In case a PLD lake IDs is redundant among different gauge sources, we will prefer
# listed above. The gauge-source blocks are ordered by preference.
print('total number of unique PLD lakes with gauge data: ' + str(len(test_lakeIDs)))


# ======== Additional Helpers for Computing MAE and Median Residuals =============================
def _finite_pair_mask(swot, gauge):
    swot = np.asarray(swot, dtype=float)
    gauge = np.asarray(gauge, dtype=float)
    return np.isfinite(swot) & np.isfinite(gauge)

def _drop_zero_residual_pairs(swot, gauge, zero_tol=zero_resid_tol):
    """
    Keep finite paired observations, but remove pairs whose residual is
    effectively zero after bias correction.

    Returns
    -------
    swot_keep, gauge_keep, n_before, n_after
    """
    swot = np.asarray(swot, dtype=float)
    gauge = np.asarray(gauge, dtype=float)

    valid = _finite_pair_mask(swot, gauge)
    residual = swot - gauge
    keep = valid & (np.abs(residual) > zero_tol)

    return swot[keep], gauge[keep], int(valid.sum()), int(keep.sum())

def _compute_mae_medianE_drop_zero_residuals(
    swot,
    gauge,
    min_n=min_matched_obs_for_error_metrics,
    zero_tol=zero_resid_tol
):
    """
    Compute both MAE and signed median residual after dropping effectively
    zero residuals.
    """
    swot_keep, gauge_keep, n_before, n_after = _drop_zero_residual_pairs(
        swot, gauge, zero_tol=zero_tol
    )

    if n_after >= min_n:
        mae = compute_mae(swot_keep, gauge_keep)
        medianE = compute_median_residual(swot_keep, gauge_keep)
    else:
        mae = np.nan
        medianE = np.nan

    return mae, medianE, n_before, n_after
# =========================================================================================


"""
Main processing outputs
-----------------------
The validation workflow accumulates three primary DataFrames.

1. df_lake_time_series
   Observation-level LakeSP table stacked across all evaluated lakes.

   Key columns in addition to LakeSP attributes:
   - datetime: UTC timestamp derived from LakeSP 'time'.
   - filter_flag: 1 for retained HALF observations, 0 for removed outliers.
   - wse_adjusted: final retained WSE after optional cross-pass adjustment;
     NaN for removed observations.
   - gauge_datetime, gauge_wse: nearest matched gauge observation, when gauge
     data are available within the regional time tolerance.
   - gauge_wse_bias_corrected: gauge WSE shifted to the LakeSP datum using the
     median LakeSP-gauge offset.
   - wse_benchmark_stringent: original WSE retained by the stringent benchmark;
     NaN otherwise.
   - wse_benchmark_permissive: original WSE retained by the permissive
     benchmark; NaN otherwise.
   - n_while, n_while_r2, filter_attempt: iteration and fallback status.

2. df_lake_heuristic_thresholds
   Lake-specific heuristic thresholds returned by
   'calibrate_heuristic_thresholds'.

   Core columns:
   - lake_id: PLD lake ID.
   - crid_scenario: two-character CRID suffix such as C0, C2, or D0.
   - pass_id: SWOT orbit pass ID.
   - ice_condition: ice-free, ice-covered, or both.
   - wse_std_thr_cal: calibrated upper threshold for 'wse_std' in metres.
   - wse_u_thr_cal: calibrated upper threshold for 'wse_u' in metres.
   - xtrk_dist_thr_cal: calibrated lower threshold for 'abs(xtrk_dist)' in
     metres; calibrated for traceability but not applied in the default HALF
     settings.
   - grouping_scheme: provenance flag describing how the threshold row is
     represented in the full input time series and conservative calibration
     subset. See calibrate_heuristic_thresholds() for details.

   This validation workflow stores the calibrated threshold table only. 
   For simplicity, the bounded threshold summary and observation-level 
   threshold details returned by apply_customized_filter() are intentionally 
   not appended to the validation outputs.

3. df_lake_stats
   Lake-level validation and retention metrics.

   Metric groups:
   - Raw LakeSP: *_raw columns.
   - HALF: columns without a method suffix, such as mae, correlation,
     retention_rate, and var_*_swot_daily.
   - Stringent benchmark: *_benchmark_stringent columns.
   - Permissive benchmark: *_benchmark_permissive columns.

   Iteration-status conventions:
   - n_while or n_while_r2 = -9: no valid LakeSP observations.
   - n_while or n_while_r2 = -2: the round was disabled or not applicable.
   - n_while or n_while_r2 = -1: filtering attempt was abandoned.
   - n_while or n_while_r2 >= 0: number of executed iterations.

   filter_attempt values:
   - 1_strict: strict HALF criteria were met.
   - 2_lenient: lenient HALF criteria were required.
   - 3_tukey: Tukey fallback was used.
   - 4_none: no filtering attempt succeeded.
   - no data: no valid LakeSP observations were available.

Notes
-----
- Retention rates are fractions in [0, 1]. NaN means no LakeSP observations
  were available for that lake.
- 'ice_duration' is the fraction of LakeSP observations with ice_clim_f == 2.
- 'medianE' is computed with 'compute_median_residual(swot, gauge)', which
  follows: residual = second argument - first argument.
"""
# Initialize the three major outputs described above.
df_lake_time_series = pd.DataFrame()
df_lake_heuristic_thresholds = pd.DataFrame()
df_lake_stats = pd.DataFrame()

# Define fill values depending on variable type.
fill_text = 'no_data'
fill_float = -999999999999

# Retrieve LakeSP time series
if SP_retrieval_method == 'on-premise': # retrieve all lakes from the previously saved 'on-premise' file
    df_Hydrocron = pd.read_csv(os.path.join(work_dir, f"df_Hydrocron_{version_filename}.csv"))
else: # otherwise, read from Hydrocron.
    df_Hydrocron = pd.DataFrame() # Initialize an empty dataframe for Hydrocron retrieval

# Loop through each unique test lake
lake_id_cycle_adjusted = [] # Initiate an array identifying lakes with intra-cycle adjustment.
for feature_id in test_lakeIDs: # test_lakeIDs contain a list of unique PLD lake IDs.
    print('lake ID: ' + str(feature_id))

    # Retrieve gauge metadata for this lake
    if feature_id in test_lakes['lake_id'].values: # Check if feature_id exists in the test_lakes dataframe
        # In case a PLD lake ID is redundant among different gauge sources, we will favor the first gauge source
        gauge_source = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_source'].values[0]
        gauge_id = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_id'].values[0]
        gauge_datum = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_datum'].values[0]
        gauge_dir = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'gauge_dir'].values[0]
    else:
        gauge_source = None #string
        gauge_id = None #string
        gauge_datum = np.nan #numeric
        gauge_dir = None #string

    # Read gauge measurements
    if gauge_source == 'China': # Reservoirs in China
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df.loc[gauge_df["PLD_Lake_ID"] == feature_id].copy() # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df = pd.DataFrame({
            "gauge_datetime": pd.to_datetime(gauge_df[["Year", "Month", "Day", "Hour", "Minute", "Second"]]),
            "gauge_wse": gauge_df["WSE/m"]
            }) # Convert to gauge_df with required column format
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
        time_tolerance = '24h'

    if gauge_source == 'West_Africa': # Reservoirs in west Africa
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df.loc[gauge_df["PLD_Lake_ID"] == feature_id].copy() # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
        time_tolerance = '24h'

    if gauge_source == 'Tefe': # Lake Tefe in the Amazon
        gauge_df = pd.read_excel(gauge_dir)
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
        time_tolerance = '24h'

    if gauge_source == 'Amazon_floodplain': # Other floodplain lakes in the Amazon
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df.loc[gauge_df["PLD_Lake_ID"] == feature_id].copy() # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
        time_tolerance = '24h'

    if gauge_source == 'Ceara_Brazil': # Small reservoirs in Ceara State, Brazil
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df.loc[gauge_df["PLD_Lake_ID"] == feature_id].copy() # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
        time_tolerance = '24h'

    if gauge_source == 'India': # Indian reservoirs
        reservoir_name = (test_lakes.loc[test_lakes['lake_id'] == feature_id, 'reservoir_name'].values[0])

        if feature_id not in Indian_lakes_only_for_monthly_gauge_data: #use daily gauge records
            # Avoid re-reading the regional file inside the loop; use the preloaded table instead.
            gauge_df = df_India.loc[
                df_India["Reservoir_name"].astype(str).str.strip().str.lower()
                == str(reservoir_name).strip().lower()
            ].copy() #case insensitive
            gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['Date']) # Ensure datetime is in datetime64 format.
            gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
            gauge_df['gauge_wse'] = gauge_df['Level']
            time_tolerance = '16d' # Indian reservoir records may be irregular; use a wider matching tolerance.
        else: #use monthly gauge records
            reservoir_state = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'reservoir_state'].values[0]
            gauge_df = pd.read_csv(Indian_gauge_monthly)
            gauge_df = gauge_df.loc[
                (gauge_df["Reservoir Name"] == reservoir_name) &
                (gauge_df["State"] == reservoir_state)
            ].copy()
            gauge_df["gauge_datetime"] = pd.to_datetime(gauge_df["Date"] + "-15 12:00:00", format='%Y-%m-%d %H:%M:%S') #Assuming 15th of each month for now.
            gauge_df = gauge_df.rename(columns={"Level": "gauge_wse"})
            # Find the "District" with the longest time span for this lake (there could be multiple districts for the same lake)
            def time_range(gdf):
                s = gdf['gauge_datetime'].dropna()
                return (s.max() - s.min()).days if not s.empty else float('nan')
            longest_gage_id = (
                gauge_df.groupby('District', dropna=False)
                .apply(time_range, include_groups=False)  # silence the warning
                .idxmax()
                )
            # Filter for best gage_id and format output
            gauge_df = gauge_df.loc[gauge_df["District"] == longest_gage_id].copy()
            gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
            time_tolerance = '16d'

    if gauge_source == 'Other_Brazil': # Other reservoirs in Brazil
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df.loc[gauge_df["PLD_Lake_ID"] == feature_id].copy() # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
        time_tolerance = '24h'

    if gauge_source == 'Bafu': # Lakes in Switzerland
        gauge_df = df_Bafu.loc[df_Bafu["PLD_Lake_ID"] == feature_id].copy()
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime'])
        gauge_df = gauge_df.sort_values('gauge_datetime')
        time_tolerance = '24h'

    if gauge_source == 'nve': # Lakes in Norway
        gauge_df = df_nve.loc[df_nve["PLD_Lake_ID"] == feature_id].copy()
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime'])
        gauge_df = gauge_df.sort_values('gauge_datetime')
        time_tolerance = '24h'

    if gauge_source == 'Canada_GNSS': # Lakes in Canada
        gauge_df = df_Canada.loc[df_Canada["PLD_Lake_ID"] == feature_id].copy()
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime'])
        gauge_df = gauge_df.sort_values('gauge_datetime')
        time_tolerance = '24h'

    if gauge_source == 'Canada_UN': # Lakes in Canada
        gauge_df = df_Canada_UN.loc[df_Canada_UN["PLD_Lake_ID"] == feature_id].copy()
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime'])
        gauge_df = gauge_df.sort_values('gauge_datetime')
        time_tolerance = '24h'

    if gauge_source == 'US': # Lakes in the US
        gauge_df = df_US.loc[df_US["PLD_Lake_ID"] == feature_id].copy()
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime'])
        gauge_df = gauge_df.sort_values('gauge_datetime')
        time_tolerance = '24h'

    # Additional gauge data clean-up:
    # Some individual gauge observations appear erroneous. They are removed from the record.
    if gauge_source is None:
        gauge_df = None
    else: # Manual removal of spotted gauge errors
        if feature_id == 4520010163:
            gauge_df.loc[(gauge_df['gauge_wse'] > 320), 'gauge_wse'] = None #310
        if feature_id == 4520053003:
            gauge_df.loc[(gauge_df['gauge_wse'] > 317), 'gauge_wse'] = None
        if feature_id == 4530121723:
            gauge_df.loc[(gauge_df['gauge_wse'] < 286.298), 'gauge_wse'] = None
        if feature_id == 4530143033:
            gauge_df.loc[(gauge_df['gauge_wse'] > 510), 'gauge_wse'] = None
        if feature_id == 4530176703:
            gauge_df.loc[(gauge_df["gauge_wse"] > 750) | (gauge_df["gauge_wse"] < 700), 'gauge_wse'] = None
        if feature_id == 4530179073:
            gauge_df.loc[(gauge_df["gauge_wse"] < 640), 'gauge_wse'] = None
        if feature_id == 4530181833:
            gauge_df.loc[(gauge_df["gauge_wse"] > 570), 'gauge_wse'] = None
        if feature_id == 4530181863:
            gauge_df.loc[(gauge_df["gauge_wse"] > 720) | (gauge_df["gauge_wse"] < 680), 'gauge_wse'] = None
        if feature_id == 4530247173:
            gauge_df.loc[(gauge_df["gauge_wse"] > 350), 'gauge_wse'] = None
        if feature_id == 4530267193:
            gauge_df.loc[(gauge_df["gauge_wse"] < 550), 'gauge_wse'] = None
        if feature_id == 4530268033:
            gauge_df.loc[(gauge_df["gauge_wse"] > 470), 'gauge_wse'] = None
        if feature_id == 4530313823:
            gauge_df.loc[(gauge_df["gauge_wse"] < 130), 'gauge_wse'] = None
        if feature_id == 4530662203:
            gauge_df.loc[(gauge_df["gauge_wse"] < 130), 'gauge_wse'] = None
        if feature_id == 4530670823:
            gauge_df.loc[(gauge_df["gauge_wse"] < 250), 'gauge_wse'] = None
        if feature_id == 4530746562:
            gauge_df.loc[(gauge_df["gauge_wse"] < 79), 'gauge_wse'] = None
        if feature_id == 4540003332:
            gauge_df.loc[(gauge_df["gauge_wse"] > 200), 'gauge_wse'] = None
        if feature_id == 4540003873:
            gauge_df.loc[(gauge_df["gauge_wse"] < 520), 'gauge_wse'] = None
        if feature_id == 4540010073:
            gauge_df.loc[(gauge_df["gauge_wse"] > 560), 'gauge_wse'] = None
        if feature_id == 4540028623:
            gauge_df.loc[(gauge_df["gauge_wse"] > 150), 'gauge_wse'] = None
        if feature_id == 4540066493:
            gauge_df.loc[(gauge_df["gauge_wse"] > 620) | (gauge_df["gauge_wse"] < 580), 'gauge_wse'] = None
        if feature_id == 4540095363:
            gauge_df.loc[(gauge_df["gauge_wse"] > 150), 'gauge_wse'] = None
        if feature_id == 4550007133:
            gauge_df.loc[(gauge_df["gauge_wse"] > 120), 'gauge_wse'] = None
        if feature_id == 4550022642:
            gauge_df.loc[(gauge_df["gauge_wse"] < 45), 'gauge_wse'] = None
        if feature_id == 4550050162:
            gauge_df.loc[(gauge_df["gauge_wse"] > 84), 'gauge_wse'] = None
        if feature_id == 4550070222:
            gauge_df.loc[(gauge_df["gauge_wse"] > 75), 'gauge_wse'] = None
        if feature_id == 4530241183:
            gauge_df.loc[(gauge_df["gauge_wse"] < 515), 'gauge_wse'] = None
        if feature_id == 4530377462:
            gauge_df.loc[(gauge_df["gauge_wse"] < 170), 'gauge_wse'] = None
        if feature_id == 4540009913:
            gauge_df.loc[(gauge_df["gauge_wse"] < 570), 'gauge_wse'] = None
        if feature_id == 4530388743:
            gauge_df.loc[(gauge_df["gauge_wse"] < 95), 'gauge_wse'] = None
        if feature_id == 4530389213:
            gauge_df.loc[(gauge_df["gauge_wse"] < 30), 'gauge_wse'] = None
        if feature_id == 4530322143:
            gauge_df.loc[(gauge_df["gauge_wse"] < 80), 'gauge_wse'] = None
        if feature_id == 2510066302:
            gauge_df.loc[(gauge_df['gauge_wse'] < 150), 'gauge_wse'] = None
        if feature_id == 2510165942:
            mask = (gauge_df['gauge_datetime'] >= pd.Timestamp('2023-12-31')) & (gauge_df['gauge_datetime'] <= pd.Timestamp('2025-01-01'))
            gauge_df.loc[mask, 'gauge_wse'] = None
        if feature_id == 2510207473:
            gauge_df.loc[(gauge_df['gauge_wse'] < 15), 'gauge_wse'] = None
        if feature_id == 2510219872:
            gauge_df.loc[(gauge_df['gauge_wse'] < -200000), 'gauge_wse'] = None
        if feature_id == 2510280033:
            gauge_df.loc[(gauge_df['gauge_wse'] < 100), 'gauge_wse'] = None
        if feature_id == 2510280312:
            gauge_df.loc[(gauge_df['gauge_wse'] < -2000), 'gauge_wse'] = None
        if feature_id == 7421019143:
            mask = (gauge_df['gauge_datetime'] <= pd.Timestamp('2025-02-10')) & (gauge_df['gauge_wse'] > 39)
            gauge_df.loc[mask, 'gauge_wse'] = None
        if feature_id == 2510163762:
            gauge_df.loc[(gauge_df['gauge_wse'] < 600), 'gauge_wse'] = None
        if feature_id == 2510164362:
            gauge_df.loc[(gauge_df['gauge_wse'] < 500), 'gauge_wse'] = None
        if feature_id == 2510165803:
            gauge_df.loc[(gauge_df['gauge_wse'] < 800), 'gauge_wse'] = None
        if feature_id == 2510193482:
            gauge_df.loc[(gauge_df['gauge_wse'] > 2000000), 'gauge_wse'] = None
        if feature_id == 2510270043:
            gauge_df.loc[(gauge_df['gauge_wse'] < 0), 'gauge_wse'] = None
        if feature_id == 4320068453:
            gauge_df.loc[(gauge_df['gauge_wse'] < 300), 'gauge_wse'] = None
        if feature_id == 7121134183:
            gauge_df.loc[(gauge_df['gauge_wse'] < 233.5), 'gauge_wse'] = None
        if feature_id == 7121329183:
            gauge_df.loc[(gauge_df['gauge_wse'] < 215), 'gauge_wse'] = None
        if feature_id == 7130096352:
            gauge_df.loc[(gauge_df['gauge_wse'] < 27), 'gauge_wse'] = None
        if feature_id == 7250051493:
            gauge_df.loc[(gauge_df['gauge_wse'] > 2), 'gauge_wse'] = None
        if feature_id == 7250860123:
            gauge_df.loc[(gauge_df['gauge_wse'] < 376), 'gauge_wse'] = None
        if feature_id == 7250891072:
            gauge_df.loc[(gauge_df['gauge_wse'] < 155), 'gauge_wse'] = None
        if feature_id == 7420176273:
            gauge_df.loc[(gauge_df['gauge_wse'] < 362), 'gauge_wse'] = None
        if feature_id == 7421019143:
            gauge_df.loc[(gauge_df['gauge_wse'] > 60), 'gauge_wse'] = None
        if feature_id == 7421075772:
            gauge_df.loc[(gauge_df['gauge_wse'] > 2), 'gauge_wse'] = None
        if feature_id == 7421099602:
            gauge_df.loc[(gauge_df['gauge_wse'] > 29), 'gauge_wse'] = None
        if feature_id == 7720027272:
            gauge_df.loc[(gauge_df['gauge_wse'] > 2550.5), 'gauge_wse'] = None
        if feature_id == 7730019732:
            gauge_df.loc[(gauge_df['gauge_wse'] > 3048), 'gauge_wse'] = None
        if feature_id == 7740037982:
            gauge_df.loc[(gauge_df['gauge_wse'] > 4), 'gauge_wse'] = None
        if feature_id == 7720027743:
            gauge_df.loc[(gauge_df['gauge_wse'] < 2400), 'gauge_wse'] = None
        if feature_id == 7740035873:
            gauge_df.loc[(gauge_df['gauge_wse'] < 226), 'gauge_wse'] = None
        if feature_id == 8320311912:
            gauge_df.loc[(gauge_df['gauge_wse'] < 300), 'gauge_wse'] = None
        if feature_id == 4340457112:
            mask = gauge_df['gauge_datetime'] < pd.Timestamp('2024-04-01')
            gauge_df.loc[mask, 'gauge_wse'] = None
        if feature_id == 2510260562:
            mask = ( (gauge_df['gauge_datetime'] > pd.Timestamp('2023-07-01')) & (gauge_df['gauge_datetime'] < pd.Timestamp('2023-09-01')) ) \
                & (gauge_df['gauge_wse'] < 241)
            gauge_df.loc[mask, 'gauge_wse'] = None
            mask = ( (gauge_df['gauge_datetime'] > pd.Timestamp('2024-03-27')) & (gauge_df['gauge_datetime'] < pd.Timestamp('2024-04-04')) ) \
                & (gauge_df['gauge_wse'] > 239.5)
            gauge_df.loc[mask, 'gauge_wse'] = None
        #Drop all None values
        gauge_df = gauge_df.dropna(subset=['gauge_wse'])

    # Retrieve LakeSP time series based on the preferred method
    if SP_retrieval_method == 'Hydrocron': # from Hydrocron directly.
        # Read LakeSP data from Hydrocron
        feature = "PriorLake"
        output =  "csv" #"geojson"
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
        )
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
        hydrocron_response = requests.get(enquiry_input, timeout=120).json()

        try: # If this Hydrocron response is not empty.
            extracted_data = hydrocron_response['results'][output]
            df = pd.read_csv(StringIO(extracted_data))
        except: # Return an empty DataFrame with the LakeSP column structure when available.
            # Assume the first lake does not return an empty df so SP attributes already exists in df_Hydrocron.
            df = df_Hydrocron[df_Hydrocron["lake_id"] == -999]

        # Append the result to df_Hydrocron. Nothing is added if df is empty.
        df_Hydrocron = pd.concat([df_Hydrocron, df], ignore_index=True)

    # If on-premise, df is retrieved from the previously saved local file;
    # otherwise (if Hydrocron), retrieve from the incrementally appended df_Hydrocron
    df = df_Hydrocron[df_Hydrocron["lake_id"] == feature_id] # Filter for lake_id based on the current feature_id
    if df.empty:
        p_ref_area = np.nan
    else:
        p_ref_area = df["p_ref_area"].iloc[0]

    df = df[df["collection_shortname"] == collection_shortname]



    # ==============Apply the filter chain below==================

    """
    LakeSP time series preprocessing
    """
    # Add index_col to preserve the original df ordering/reproducibility and later label outliers.
    df = df.copy() # Make df an independent DataFrame and eliminate the SettingWithCopyWarning
    df['index_col'] = range(len(df))

    # Drop invalid records for simplicity: time and WSE must both exist.
    df = df.loc[(df.time != fill_float) & (df.wse != fill_float)]

    # Mask out invalid values for filtering metrics
    df.wse_u    = df.wse_u.mask(df.wse_u  == fill_float, np.nan)
    df.wse_std  = df.wse_std.mask(df.wse_std == fill_float, np.nan)
    df.xtrk_dist= df.xtrk_dist.mask(df.xtrk_dist == fill_float, np.nan)

    # Initialize an outlier label: # 1 = good; 0 = outlier
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
    # Note: df_eval.empty, i.e., no valid SWOT data available for this lake,
    #       is possible but will be handled by the following scripts.
    #       When df_eval is empty, downstream filtering returns an empty result.



    """
    Step 1 — calibrate lake-specific heuristic quality thresholds.

    The LakeSP summary quality flags are not used here as the final filter.
    Instead, they identify a conservative, high-confidence training subset from
    which lake-specific thresholds are calibrated.  The calibrated thresholds are
    then applied to the full LakeSP record to build the heuristic baseline.

    Calibrated diagnostic variables
    -------------------------------
    wse_std_thr_cal
        Calibrated upper threshold for within-polygon WSE variability
        ('wse_std', metres).
        Larger values may indicate mixed water/land pixels, false water
        detections, or spatially inconsistent WSE retrievals.
    wse_u_thr_cal
        Calibrated upper threshold for LakeSP WSE uncertainty ('wse_u',
        metres), which
        summarizes random and systematic uncertainty contributions from the
        interferogram and LakeSP processing chain.
    xtrk_dist_thr_cal
        Calibrated lower threshold for absolute cross-track distance
        ('abs(xtrk_dist)', metres). It is not applied in the current release
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

    # Calibrate heuristic thresholds, which can be stratified to pass, version, and ice-condition.
    #  - wse_std (in m): Lake surface flatness, assuming errors lead to large wse_std
    #  - wse_u (in m): Algorithm processing uncertainty, assuming errors lead to large wse_u
    #  - xtrk_dist (in m, abs. 0-75000 m): Distance of lake polygon centroid from the nadir track, assuming errors lead to small |xtrk_dist|

    # Note for pre-Version-D2 LakeSP products:
    #    A lake split by the [-10 to 10] km central track may also have an |xtrk_dist value| < 10 km,
    #    Yet its WSE value can be useful, as shown in many cases, such as 4340980733 (TGD) and 4530143033.
    #    So, an observation whose |xtrk_dist| < 10 km may not mean poor WSE,
    #    and we cannot just exclude observations whose |xtrk_dist| < 10 km, either.
    #    In other words, while wse_std represents water consistency and wse_u represents algorithm uncertainty, both indicating quality,
    #    small |xtrk_dist| may not always represent poor quality, so it is not used by default here.
    #    Note: This issue of xtrk_dist has been addressed in LakeSP VD2.

    # Caution: Applying pass and version groupings/stratification for wse_std or wse_u can help avoid over-rejection,
    #          but overly small groups may also make thresholds unstable.
    #          Pass grouping is theoretically needed for xtrk_dist as lake position varies in different passes.
    df_heuristic_thresholds = calibrate_heuristic_thresholds(df_eval, conservative_SQL,
                                       by_crid_scenario = [False, False, False], #boolean sequence for wse_std, wse_u, xtrk_dist
                                       by_pass_id = [False, False, True], #wse_std, wse_u, xtrk_dist
                                       by_ice = [True, True, True]) #fixed here; wse_std, wse_u, xtrk_dist
    # Note: output df_heuristic_thresholds contains:
    #       ['lake_id', 'crid_scenario', 'pass_id', 'ice_condition',
    #        'wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal',
    #        'grouping_scheme']
    #       - crid_scenario depends on the last two digits in crid, e.g., C0, C2, and D0.
    #       - ice_condition has three scenarios: "ice-free", "ice-covered", and "both".
    #       See function calibrate_heuristic_thresholds for details.
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
    # Reasoning for threshold bounds:
    # - max bound for wse_std_threshold = 3 or 5. 3 seems more conservative and accurate, but 5 m may allow more observations in.
    # - min bound for wse_std_threshold = 0
    # - max bound for wse_u_threshold = 0.5. 0.5 m may be conservative and may benefit from future optimization using validation data.
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
    # The first list is used for observations treated as ice-free for threshold
    # selection (LakeSP ice_clim_f < 2). The second list is used for observations
    # treated as ice-covered for threshold selection (LakeSP ice_clim_f >= 2).
    #
    # Current default:
    #     - wse_std and wse_u are filtered using ice-free calibrated thresholds for
    #       both ice-free and ice-covered observations.
    #     - xtrk_dist is not used for filtering because small |xtrk_dist| (before version D2)
    #       does not consistently indicate poor WSE quality for lakes spanning or near nadir.
    rules_for_ice_free_data    = ['ice-free', 'ice-free', 'not apply']
    rules_for_ice_covered_data = ['ice-free', 'ice-free', 'not apply']


    # Execute the filter function
    # Attempt 1: strict.
    # allow_major_gap = 'no' recommended, which enables constant full control throughout the time series.
    filter_attempt = '1_strict' # the most strict filtering
    (
        df_filtered,                    # Filtered LakeSP observations returned by this HALF attempt.
        n_while_filtered,               # [round-1, round-2] iteration counts/status codes.
        filter_status,                  # Filtering status: success, fail, heuristic baseline, or no data.
        _threshold_summary_unused,      # Returned by the current API; not stored in this validation script.
        _observation_thresholds_unused, # Returned by the current API; not stored in this validation script.
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

                                         gauge_df = gauge_df, # enter gauge_df; None if no gauge data is available.
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
            df_filtered,                    # Filtered LakeSP observations returned by this HALF attempt.
            n_while_filtered,               # [round-1, round-2] iteration counts/status codes.
            filter_status,                  # Filtering status: success, fail, heuristic baseline, or no data.
            _threshold_summary_unused,      # Returned by the current API; not stored in this validation script.
            _observation_thresholds_unused, # Returned by the current API; not stored in this validation script.
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

                                             gauge_df = gauge_df, # enter gauge_df; None if no gauge data is available.
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
        # Apply the recommended pass-bias adjustment option from sp_cycle_adjustment().
        # The function returns multiple outputs; only the adjusted filtered time series
        # is needed in this validation workflow.
        #
        # Note: depending on the internal checks in sp_cycle_adjustment(), this step may
        # leave WSE unchanged if cross-pass inconsistency is not substantial.
        # Applying cross-pass bias correction may also remove additional outliers.
        # See sp_cycle_adjustment() for details.
        _, _, df_filtered = sp_cycle_adjustment(df_filtered)

        # Record whether the adjustment changed any retained WSE value for this lake.
        if (
            df_filtered["wse"].notna()
            & df_filtered["wse_adjusted"].notna()
            & (df_filtered["wse"] != df_filtered["wse_adjusted"])
        ).any():
            lake_id_cycle_adjusted.append(df_filtered["lake_id"].iloc[0])
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

    # Assign filter_attempt, n_while, and n_while_r2 values into new columns
    df[['n_while', 'n_while_r2', 'filter_attempt']] = [n_while, n_while_r2, filter_attempt]







    """
    By now, filtering process has been completed.
    The following sections are for computing validation statistics and plotting outputs.
    """







    """
    Plot filtered time series and compute validation statistics for this lake.

    Terminology
    -----------
    - "HALF" refers to the heuristic adaptive filter output.
    - "Stringent benchmark" is the conservative benchmark filter.
    - "Permissive benchmark" is the more inclusive/permissive benchmark filter.

    The term "benchmark" is used for comparison purposes.
    Both benchmark filters are based on built-in LakeSP quality flags.
    """
    # Define comparison benchmarks based on LakeSP native quality flags.
    # These are not used by HALF itself; they are retained only for validation.
    # Stringent benchmark: maximizes quality confidence but often reduces temporal coverage.
    STRINGENT_BENCHMARK_SQL = 'xovr_cal_q < 1 & ice_clim_f < 1 & quality_f < 1'
    # Permissive benchmark: preserves more observations while still removing
    # low-confidence cases according to native LakeSP flags.
    PERMISSIVE_BENCHMARK_SQL = (
        f'(xovr_cal_q < 2) & (ice_clim_f < 2) & (' # ice_clim_f<=2 was also tested. Conclusions are similar.
        f'   ((quality_f < 1) & ~{new_version_suffix})'
        f' | ((quality_f < 3) &  {new_version_suffix})'
        f')'
    )

    # Add benchmark WSE columns. Rows that do not pass the corresponding
    # benchmark receive NaN so downstream metrics can ignore them.
    # Evaluate masks
    mask_benchmark_stringent = df.eval(STRINGENT_BENCHMARK_SQL)
    mask_benchmark_permissive = df.index.isin(df.query(PERMISSIVE_BENCHMARK_SQL, engine="python").index)
    # Create benchmark-specific WSE columns.
    df['wse_benchmark_stringent'] = np.where(mask_benchmark_stringent, df['wse'], np.nan)
    df['wse_benchmark_permissive'] = np.where(mask_benchmark_permissive, df['wse'], np.nan)

    # Set up the time series plot.
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, linewidth=0.5, zorder=1)

    # Compute statistics for this lake
    if gauge_source is not None: # If this lake has gauge data (this works if df_filtered is empty)
        # Perform nearest-time join
        # Join gauge_df's gauge_datetime and gauge_wse values into SWOT's df (using datetime in df)
        # by finding the closest timestamp in gauge_df within time_tolerance (e.g., 24 HR) of each datetime.
        # If multiple gauge times are within time_tolerance, the closest one is used.
        # Note both gauge_df.gauge_datetime and df.datetime have already been formatted to datetime64 format:
        #    e.g., using pd.to_datetime(df['datetime'])
        gauge_df = gauge_df[gauge_df['gauge_datetime'].notna()]
        df = pd.merge_asof(
            df,
            gauge_df[['gauge_datetime', 'gauge_wse']],
            left_on='datetime', #in SWOT
            right_on='gauge_datetime', #in gauge
            direction='nearest',
            tolerance=pd.Timedelta(time_tolerance)
            ) # Note: this will generate two extra attributes (gauge_datatime and gauge_wse) in df.

        # Estimate the gauge-to-LakeSP datum offset with robust fallbacks.
        # Prefer the median gauge-minus-LakeSP offset from ice-free matched observations.
        # The bias can be caused by unknown levelling of the gauge data, or
        # by the difference between the average geoid of the lake and the geoid at the gauge station
        # Use wse_adjusted as the preferred SWOT wse field if available; otherwise, use wse (original LakeSP wse)
        if df['wse_adjusted'].isna().all(): # If 'wse_adjusted' (filtered wse) is entirely NaN or unavailable
            bias_correction_field = 'wse'
        else:
            bias_correction_field = 'wse_adjusted'
        # Use ice-free observations if possible; otherwise, use all observations
        if (df['ice_clim_f'] < 1).any(): # if there are ice-free observations in df
            mask = df['ice_clim_f'] < 1
            bias_swot_gauge = np.nanmedian(df.loc[mask, 'gauge_wse'].values - df.loc[mask, bias_correction_field].values)

            # In case gauge and valid SP wse values during ice-free period do not overlap:
            if np.isnan(bias_swot_gauge):
                bias_swot_gauge = np.nanmedian(df['gauge_wse'] - df[bias_correction_field]) # Use full period
        else: # Use full period
            bias_swot_gauge = np.nanmedian(df['gauge_wse'] - df[bias_correction_field])
        # In case gauge values and valid SP wse values (e.g., if wse_adjusted is used) still do not overlap:
        if np.isnan(bias_swot_gauge):
            bias_swot_gauge = np.nanmedian(df['gauge_wse'] - df['wse'])  # Use original wse to increase the chance
        # Rare: in case no valid wse values during gauge available period
        # fallback: approximate datum alignment using unpaired medians.
        # This is mainly for visualization when no valid paired differences are available.
        if np.isnan(bias_swot_gauge):
            bias_swot_gauge = np.nanmedian(df['gauge_wse']) - np.nanmedian(df[bias_correction_field])
        # In case df is empty.
        if np.isnan(bias_swot_gauge):
            bias_swot_gauge = 0
        # Store the bias-corrected gauge WSE for validation metrics and plotting.
        df['gauge_wse_bias_corrected'] = df['gauge_wse']-bias_swot_gauge

        # Plot the gauge time series
        ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse']-bias_swot_gauge, \
               label='gauge', color='green', marker = 'o', markersize=6, linestyle='--')


        # ========================================================================================
        # Compute MAE and signed median residual after excluding effectively zero residuals
        # ========================================================================================
        """
        Lakes with only one raw LakeSP–gauge matched pair should be excluded from being
        considered valid gauge-validation lakes because the bias correction would be
        defined by that single pair. A lake must have at least two raw LakeSP–gauge matched
        pairs to support an independent gauge-based validation after datum alignment.

        For pointwise error metrics, we also exclude effectively zero residuals after
        bias correction to avoid self-calibrated datum-alignment artifacts. Such zero
        residuals can occur when a LakeSP-gauge pair used to estimate the bias correction
        is also used in the error calculation. This treatment is conservative for MAE
        because removing zero residuals cannot make MAE smaller; for signed median
        residuals, it prevents artificial centering around zero, although the sign may
        shift depending on the remaining residuals.
        """

        # Raw LakeSP
        mae_raw, medianE_raw, n_match_raw, n_match_raw_used = (
            _compute_mae_medianE_drop_zero_residuals(
                df["wse"],
                df["gauge_wse_bias_corrected"],
                min_n=min_matched_obs_for_error_metrics,
                zero_tol=zero_resid_tol
            )
        )

        # HALF-filtered LakeSP
        mae, medianE, n_match_half, n_match_half_used = (
            _compute_mae_medianE_drop_zero_residuals(
                df["wse_adjusted"],
                df["gauge_wse_bias_corrected"],
                min_n=min_matched_obs_for_error_metrics,
                zero_tol=zero_resid_tol
            )
        )

        # Stringent benchmark
        mae_benchmark_stringent, medianE_benchmark_stringent, n_match_benchmark_stringent, n_match_benchmark_stringent_used = (
            _compute_mae_medianE_drop_zero_residuals(
                df["wse_benchmark_stringent"],
                df["gauge_wse_bias_corrected"],
                min_n=min_matched_obs_for_error_metrics,
                zero_tol=zero_resid_tol
            )
        )

        # Permissive benchmark
        mae_benchmark_permissive, medianE_benchmark_permissive, n_match_benchmark_permissive, n_match_benchmark_permissive_used = (
            _compute_mae_medianE_drop_zero_residuals(
                df["wse_benchmark_permissive"],
                df["gauge_wse_bias_corrected"],
                min_n=min_matched_obs_for_error_metrics,
                zero_tol=zero_resid_tol
            )
        )


        # COMPUTE VARIABILITY METRICS (start)===============================================
        # If variability were computed only from gauge observations paired directly with
        # SWOT overpass times, the result could underestimate or misrepresent the full
        # temporal variability of the gauge record. This is especially important because
        # SWOT sampling is sparse relative to most gauge records.

        # To better evaluate whether the filtered LakeSP series preserves intra-annual
        # WSE variability, we compare daily time series over a common LakeSP-gauge
        # overlap window. The function convert_to_daily_series() first aggregates
        # observations to daily means, then interpolates LakeSP and gauge records to a
        # regular daily grid over the shared time period. Variability metrics are then
        # computed from these paired daily series.

        # Major gauge gaps are handled explicitly: if consecutive gauge observations are
        # separated by >= major_gap_days, the interior days of that gap are excluded from
        # the returned daily series. This prevents daily variability metrics from being
        # computed across long unsupported gauge-interpolation periods.

        # Remaining caveat:
        # Shorter gaps are still interpolated, and the filtered LakeSP series may involve
        # interpolation or edge filling within the raw LakeSP-gauge overlap window. Thus,
        # daily variability metrics should still be interpreted as interpolation-assisted
        # estimates rather than purely observation-only statistics.

        # Input/output behavior for variability functions:
        # - Series with valid data -> compute variability.
        # - Series with one valid value -> variability is 0.
        # - Series with all NaN values -> variability is undefined, returned as np.nan.
        # - Scalar NaN -> variability is undefined, returned as np.nan.
        # - Scalar numeric value -> variability is 0.

        # Compute daily WSE series using the same input time series for all variability metrics.
        # Metrics:
        #   - var_std_*: temporal standard deviation
        #   - var_amp_*: amplitude or range, i.e., max - min
        #   - var_p10p90_*: interdecile range, i.e., 90th - 10th percentile range
        daily_series = convert_to_daily_series(
            df, gauge_df,
            time_col='datetime',
            gauge_time_col='gauge_datetime',
            wse_col='wse',
            wse_filtered_col='wse_adjusted',
            gauge_wse_col='gauge_wse',
            interp_method='linear',
            major_gap_days=major_gauge_gap_days
        )

        daily_series_benchmark_stringent = convert_to_daily_series(
            df, gauge_df,
            time_col='datetime',
            gauge_time_col='gauge_datetime',
            wse_col='wse',
            wse_filtered_col='wse_benchmark_stringent',
            gauge_wse_col='gauge_wse',
            interp_method='linear',
            major_gap_days=major_gauge_gap_days
        )

        daily_series_benchmark_permissive = convert_to_daily_series(
            df, gauge_df,
            time_col='datetime',
            gauge_time_col='gauge_datetime',
            wse_col='wse',
            wse_filtered_col='wse_benchmark_permissive',
            gauge_wse_col='gauge_wse',
            interp_method='linear',
            major_gap_days=major_gauge_gap_days
        )
        # The three outputs share the same raw LakeSP-gauge overlap window
        # because convert_to_daily_series() defines the overlap from raw wse_col and gauge WSE, not from wse_filtered_col.
        # The output is truncated to the daily period where raw LakeSP and gauge records overlap.

        # Minimum required daily overlap span for computing variability metrics
        # is configured in VALIDATION_CONFIG and exposed as min_daily_span_days.
        def _daily_series_span_days(daily_series_dict):
            """
            Return the calendar span, in days, of the raw LakeSP-gauge daily overlap window.

            This explicitly uses the shared date index of daily_gauge and daily_wse,
            not daily_wse_filtered.
            """

            daily_gauge = daily_series_dict.get("daily_gauge", np.nan)
            daily_wse = daily_series_dict.get("daily_wse", np.nan)

            if not isinstance(daily_gauge, pd.Series) or not isinstance(daily_wse, pd.Series):
                return np.nan

            # Use dates where both raw daily LakeSP and gauge series are present.
            idx = daily_gauge.index.intersection(daily_wse.index)
            idx = pd.to_datetime(idx)
            idx = idx[~pd.isna(idx)]

            if len(idx) == 0:
                return np.nan

            return (idx.max() - idx.min()) / pd.Timedelta(days=1)


        def _daily_variability_metrics(x, allow_compute=True):
            """
            Compute all daily variability metrics from the same input series.
            If allow_compute is False, return NaN for all metrics.
            """
            if not allow_compute:
                return {
                    'std': np.nan,
                    'amp': np.nan,
                    'p10p90': np.nan
                }

            return {
                'std': compute_variability_std(x),
                'amp': compute_variability_amplitude(x),
                'p10p90': compute_variability_p10_p90_range(x)
            }

        # Check whether the common daily overlap span is sufficiently long.
        daily_span_days = _daily_series_span_days(daily_series)
        compute_daily_variability = (
            pd.notna(daily_span_days) and
            daily_span_days >= min_daily_span_days
        )

        # Gauge daily variability over the same overlapping window
        v_gauge = _daily_variability_metrics(
            daily_series.get('daily_gauge', np.nan),
            allow_compute=compute_daily_variability
        )

        # HALF-filtered daily variability
        v_half = _daily_variability_metrics(
            daily_series.get('daily_wse_filtered', np.nan),
            allow_compute=compute_daily_variability
        )

        # Raw LakeSP daily variability
        v_raw = _daily_variability_metrics(
            daily_series.get('daily_wse', np.nan),
            allow_compute=compute_daily_variability
        )

        # Stringent benchmark daily variability
        v_benchmark_stringent = _daily_variability_metrics(
            daily_series_benchmark_stringent.get('daily_wse_filtered', np.nan),
            allow_compute=compute_daily_variability
        )

        # Permissive benchmark daily variability
        v_benchmark_permissive = _daily_variability_metrics(
            daily_series_benchmark_permissive.get('daily_wse_filtered', np.nan),
            allow_compute=compute_daily_variability
        )

        # Standard deviation
        var_std_gauge_daily = v_gauge['std']
        var_std_swot_daily = v_half['std']
        var_std_swot_daily_raw = v_raw['std']
        var_std_swot_daily_benchmark_stringent = v_benchmark_stringent['std']
        var_std_swot_daily_benchmark_permissive = v_benchmark_permissive['std']

        # Amplitude/range
        var_amp_gauge_daily = v_gauge['amp']
        var_amp_swot_daily = v_half['amp']
        var_amp_swot_daily_raw = v_raw['amp']
        var_amp_swot_daily_benchmark_stringent = v_benchmark_stringent['amp']
        var_amp_swot_daily_benchmark_permissive = v_benchmark_permissive['amp']

        # 10th–90th percentile range
        var_p10p90_gauge_daily = v_gauge['p10p90']
        var_p10p90_swot_daily = v_half['p10p90']
        var_p10p90_swot_daily_raw = v_raw['p10p90']
        var_p10p90_swot_daily_benchmark_stringent = v_benchmark_stringent['p10p90']
        var_p10p90_swot_daily_benchmark_permissive = v_benchmark_permissive['p10p90']


        # Retained-observation HALF WSE variability
        # Aggregate retained LakeSP observations to one value per observed day,
        # without interpolating to a regular daily grid.
        if (
            compute_daily_variability and
            isinstance(daily_series.get("daily_gauge", np.nan), pd.Series) and
            len(daily_series["daily_gauge"].index) > 0
        ):
            overlap_start = daily_series["daily_gauge"].index.min()
            overlap_end = daily_series["daily_gauge"].index.max()

            retained_daily_obs = (
                df.assign(date=df["datetime"].dt.floor("D"))
                  .loc[
                      lambda x:
                      (x["date"] >= overlap_start) &
                      (x["date"] <= overlap_end) &
                      x["wse_adjusted"].notna(),
                      ["date", "wse_adjusted"]
                  ]
                  .groupby("date")["wse_adjusted"]
                  .mean()  #To be consistent with convert_to_daily_series, which uses mean, not median.
            )
        else:
            retained_daily_obs = pd.Series(dtype=float)

        v_half_retained = _daily_variability_metrics(
            retained_daily_obs,
            allow_compute=compute_daily_variability
        )

        var_std_swot_retained = v_half_retained["std"]
        var_amp_swot_retained = v_half_retained["amp"]
        var_p10p90_swot_retained = v_half_retained["p10p90"]
        # COMPUTE VARIABILITY METRICS (end)=====================================




        # Compute correlation coefficients based on daily time series.
        # Use the same configured daily-overlap-span criterion as the variability metrics.
        if compute_daily_variability:
            # Using all raw LakeSP daily values
            correlation_raw = compute_correlation(
                daily_series['daily_wse'],
                daily_series['daily_gauge'],
                method='pearson'
            )

            # Using only filtered HALF daily values
            correlation = compute_correlation(
                daily_series['daily_wse_filtered'],
                daily_series['daily_gauge'],
                method='pearson'
            )

            # Using stringent benchmark daily values
            correlation_benchmark_stringent = compute_correlation(
                daily_series_benchmark_stringent['daily_wse_filtered'],
                daily_series_benchmark_stringent['daily_gauge'],
                method='pearson'
            )

            # Using permissive benchmark daily values
            correlation_benchmark_permissive = compute_correlation(
                daily_series_benchmark_permissive['daily_wse_filtered'],
                daily_series_benchmark_permissive['daily_gauge'],
                method='pearson'
            )

        else:
            correlation_raw = np.nan
            correlation = np.nan
            correlation_benchmark_stringent = np.nan
            correlation_benchmark_permissive = np.nan

    else: # if this lake has no gauge data

        # Statistics for matched gauge observations (0-residual was removed)
        n_match_raw = 0
        n_match_half = 0
        n_match_benchmark_stringent = 0
        n_match_benchmark_permissive = 0
        n_match_raw_used = 0
        n_match_half_used = 0
        n_match_benchmark_stringent_used = 0
        n_match_benchmark_permissive_used = 0

        # Gauge variability metrics
        var_std_gauge_daily = np.nan
        var_amp_gauge_daily = np.nan
        var_p10p90_gauge_daily = np.nan

        # HALF
        mae = np.nan
        medianE = np.nan
        correlation = np.nan
        var_std_swot_daily = np.nan
        var_amp_swot_daily = np.nan
        var_p10p90_swot_daily = np.nan

        # Retained-observation HALF WSE variability
        var_std_swot_retained = np.nan
        var_amp_swot_retained = np.nan
        var_p10p90_swot_retained = np.nan

        # Raw LakeSP
        mae_raw = np.nan
        medianE_raw = np.nan
        correlation_raw = np.nan
        var_std_swot_daily_raw = np.nan
        var_amp_swot_daily_raw = np.nan
        var_p10p90_swot_daily_raw = np.nan

        # Stringent benchmark
        mae_benchmark_stringent = np.nan
        medianE_benchmark_stringent = np.nan
        correlation_benchmark_stringent = np.nan
        var_std_swot_daily_benchmark_stringent = np.nan
        var_amp_swot_daily_benchmark_stringent = np.nan
        var_p10p90_swot_daily_benchmark_stringent = np.nan

        # Permissive benchmark
        mae_benchmark_permissive = np.nan
        medianE_benchmark_permissive = np.nan
        correlation_benchmark_permissive = np.nan
        var_std_swot_daily_benchmark_permissive = np.nan
        var_amp_swot_daily_benchmark_permissive = np.nan
        var_p10p90_swot_daily_benchmark_permissive = np.nan

    if len(df) > 0: # If this lake has valid SWOT observations
        retention_n = len(df_filtered)
        retention_rate = len(df_filtered)/len(df)
        retention_rate_benchmark_stringent = len(df.query(STRINGENT_BENCHMARK_SQL, engine="python")) / len(df)
        retention_rate_benchmark_permissive = len(df.query(PERMISSIVE_BENCHMARK_SQL, engine="python")) / len(df)

        # Compute the proportion of fully ice-covered period in the original time series
        ice_duration = (df['ice_clim_f'] == 2).sum() / len(df) # Record-based approximation rather than exact duration in days.

    else: # This lake has no valid SWOT observations (e.g., lake_id 4330037643)
        retention_n = np.nan # NaN means this metric is not applicable because the lake has no SWOT observations.
        retention_rate = np.nan
        retention_rate_benchmark_stringent = np.nan
        retention_rate_benchmark_permissive = np.nan
        ice_duration = np.nan

    # Construct a lake stats dataframe for this lake
    df_this_lake_stats = pd.DataFrame([{
        'lake_id': feature_id,
        'lake_area': p_ref_area,

        # Matched-observation counts used for MAE and median residual
        "n_match_raw": n_match_raw,
        "n_match_half": n_match_half,
        "n_match_benchmark_stringent": n_match_benchmark_stringent,
        "n_match_benchmark_permissive": n_match_benchmark_permissive,
        # Counts after excluding effectively zero residuals
        "n_match_raw_used": n_match_raw_used,
        "n_match_half_used": n_match_half_used,
        "n_match_benchmark_stringent_used": n_match_benchmark_stringent_used,
        "n_match_benchmark_permissive_used": n_match_benchmark_permissive_used,
        "min_matched_obs_for_error_metrics": min_matched_obs_for_error_metrics,
        "zero_resid_tol": zero_resid_tol,

        # Gauge variability metrics
        'var_std_gauge_daily': var_std_gauge_daily,
        'var_amp_gauge_daily': var_amp_gauge_daily,
        'var_p10p90_gauge_daily': var_p10p90_gauge_daily,

        # HALF
        'mae': mae,
        'medianE': medianE,
        'correlation': correlation,
        'var_std_swot_daily': var_std_swot_daily,
        'var_amp_swot_daily': var_amp_swot_daily,
        'var_p10p90_swot_daily': var_p10p90_swot_daily,

        'var_std_swot_retained': var_std_swot_retained, # retained-observation metric
        'var_amp_swot_retained': var_amp_swot_retained, # retained-observation metric
        'var_p10p90_swot_retained': var_p10p90_swot_retained, # retained-observation metric

        # Raw LakeSP
        'mae_raw': mae_raw,
        'medianE_raw': medianE_raw,
        'correlation_raw': correlation_raw,
        'var_std_swot_daily_raw': var_std_swot_daily_raw,
        'var_amp_swot_daily_raw': var_amp_swot_daily_raw,
        'var_p10p90_swot_daily_raw': var_p10p90_swot_daily_raw,

        # Stringent benchmark
        'mae_benchmark_stringent': mae_benchmark_stringent,
        'medianE_benchmark_stringent': medianE_benchmark_stringent,
        'correlation_benchmark_stringent': correlation_benchmark_stringent,
        'var_std_swot_daily_benchmark_stringent': var_std_swot_daily_benchmark_stringent,
        'var_amp_swot_daily_benchmark_stringent': var_amp_swot_daily_benchmark_stringent,
        'var_p10p90_swot_daily_benchmark_stringent': var_p10p90_swot_daily_benchmark_stringent,

        # Permissive benchmark
        'mae_benchmark_permissive': mae_benchmark_permissive,
        'medianE_benchmark_permissive': medianE_benchmark_permissive,
        'correlation_benchmark_permissive': correlation_benchmark_permissive,
        'var_std_swot_daily_benchmark_permissive': var_std_swot_daily_benchmark_permissive,
        'var_amp_swot_daily_benchmark_permissive': var_amp_swot_daily_benchmark_permissive,
        'var_p10p90_swot_daily_benchmark_permissive': var_p10p90_swot_daily_benchmark_permissive,

        'retention_n': retention_n,
        'retention_rate': retention_rate,
        'retention_rate_benchmark_stringent': retention_rate_benchmark_stringent,
        'retention_rate_benchmark_permissive': retention_rate_benchmark_permissive,

        'n_while': n_while,
        'n_while_r2': n_while_r2,
        'filter_attempt': filter_attempt,

        'ice_duration': ice_duration,
        'gauge_source': gauge_source,
        'intra_cycle_flag': intra_cycle_flag
        }])

    # Concatenate df_lake_time_series by df, df_lake_stats by df_this_lake_stats,
    # and df_lake_heuristic_thresholds by df_heuristic_thresholds
    df_lake_stats = pd.concat([df_lake_stats, df_this_lake_stats], ignore_index=True)
    df_lake_heuristic_thresholds = pd.concat([df_lake_heuristic_thresholds, df_heuristic_thresholds], ignore_index=True)
    df_lake_time_series = pd.concat([df_lake_time_series, df], ignore_index=True)
    # Note: for df_lake_time_series, if the lake has no valid SWOT data (i.e., df is empty), no record is added.
    #       for df_lake_heuristic_thresholds, if the lake has no or no valid SWOT data, no record is added.
    #       for df_lake_stats, if the lake has no or no valid SWOT data, lake_id will be kept, but other attributes are nan.


    # Continue to finish the plot.
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
    ax.plot(df.query('wse_std > 3').datetime, df.query('wse_std > 3').wse,
            label='wse_std > 3 m', marker='o', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor='yellow')

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
    if len(df_filtered) >= 1 and df_filtered['wse'].notna().any(): #at least one non-Nan value
        range_wse = np.nanmax(df_filtered.wse)-np.nanmin(df_filtered.wse)
        plt.ylim(np.nanmin(df_filtered.wse)-range_wse*2, np.nanmax(df_filtered.wse)+range_wse*2)
    elif len(df) >= 1 and df['wse'].notna().any():
        plt.ylim(np.nanmin(df.wse), np.nanmax(df.wse))

    # Axis labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('WSE (m)', fontsize=12)
    ax.set_title('Lake ID ' + str(feature_id) + ' Time Series Plot. Filter: ' + filter_type + '. Gauge: ' + str(gauge_source))

    # Add statistics as a text box
    if gauge_source is not None: # With gauge data
        textstr = f'MAE (raw) = {mae_raw:.3f}\nMAE (HALF) = {mae:.3f}'
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    #ax.legend()
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.) #legend outside of the plot

    # Save the plot.
    plt.savefig(os.path.join(plots_dir, f'Attempt_{filter_attempt}_lakeID_{feature_id}_{filter_type}_{script_version}_{version_filename}.png'),bbox_inches='tight')
    if show_filtering_evolution == 'no':
        plt.close()  # free up memory

# Save Hydrocron time series into local disk
if SP_retrieval_method == 'Hydrocron':
    df_Hydrocron.to_csv(os.path.join(work_dir, f"df_Hydrocron_{version_filename}.csv"), index=False)



"""
The following scripts are to report detailed validation statistics and generate validation figures.

Clean up data relationship:
    Hierarchy:
        1. All gauge lakes (in df_lake_stats)
        2. SWOT-null lakes:
            Lakes having no SWOT data or no valid SWOT data in the first place.
            These lakes need to be removed from reporting gauge statistics.
            They include:
                lakes missing Hydrocron result: df is retrieved empty
                lakes with Hydrocron result but wse is all fill values
        3. Lakes without valid gauge data joined:
            Even after removing SWOT-null lakes, there could be some lakes where gauge data show
            no overlap with raw wse values.
            These lakes need to be removed from validation purpose.

        ! Combination of 2 + 3: invalid lakes, which cannot be used for any filter validation.

Recall:
    Which lakes are included:
        df_lake_stats: all unique gauge lakes
        df_Hydrocron: excludes lakes having no Hydrocron data, but includes those where LakeSP wse is all invalid
        df_lake_time_series: excludes invalid lakes (lakes having no Hydrocron data; or having no valid wse in hydrocron data)
        df_lake_heuristic_thresholds: the same as df_lake_time_series
"""
df_lake_stats_intact = df_lake_stats.copy()

# Sanity check starts ================================================================
def _idset(df, col="lake_id"):
    return set(df[col].dropna().unique())
ids_gauge = _idset(df_lake_stats, "lake_id")
ids_hc    = _idset(df_Hydrocron, "lake_id")
ids_ts    = _idset(df_lake_time_series, "lake_id")
ids_thr   = _idset(df_lake_heuristic_thresholds, "lake_id")
# 1) lake_ids missing in df_Hydrocron compared to df_lake_stats (gauge universe)
missing_in_hc = ids_gauge - ids_hc
# 2) lake_ids missing in df_lake_time_series compared to df_lake_stats
missing_in_ts = ids_gauge - ids_ts
# 3) lake_ids missing in df_lake_heuristic_thresholds compared to df_lake_stats
missing_in_thr = ids_gauge - ids_thr
print("Lake count:")
print("  original gauge lakes:", len(ids_gauge))
print("  Hydrocron lakes:", len(ids_hc))
print("  time_series lakes:", len(ids_ts))
print("  thresholds lakes:", len(ids_thr))
print()
print("Missing (vs gauge lakes):")
print("  missing_in_hc :", len(missing_in_hc))
print("  missing_in_ts :", len(missing_in_ts))
print("  missing_in_thr:", len(missing_in_thr))
# Print lists (sorted)
print("\nLake IDs missing in df_Hydrocron (vs df_lake_stats):")
print(sorted(missing_in_hc))
print("\nLake IDs missing in df_lake_time_series (vs df_lake_stats):")
print(sorted(missing_in_ts))
print("\nLake IDs missing in df_lake_heuristic_thresholds (vs df_lake_stats):")
print(sorted(missing_in_thr))

# Identify invalid validation lakes.
# -----------------------------
# 1) SWOT-null lakes (missing SWOT data) already defined as: missing_in_ts
# -----------------------------
swot_null_lake_ids = set(missing_in_ts)
print(f"SWOT-null lake_ids (missing SWOT data): {len(swot_null_lake_ids)}")
print("SWOT-null lake_ids:", sorted(swot_null_lake_ids))
print()

# -----------------------------
# 2) Lakes without valid gauge data joined:
#    lake_ids in df_lake_time_series where gauge_wse is ALL NaN
# -----------------------------
all_nan_gauge = (
    df_lake_time_series.groupby("lake_id")["gauge_wse"]
    .apply(lambda s: s.isna().all())
)
no_gauge_join_lake_ids = set(all_nan_gauge[all_nan_gauge].index.tolist())

print(f"Lakes with NO gauge_wse joined: {len(no_gauge_join_lake_ids)}")
print("Lake IDs (no gauge joined):")
print(sorted(no_gauge_join_lake_ids))
print()

# -----------------------------
# 3) Lakes with too few raw LakeSP-gauge matched pairs:
#    finite raw WSE and finite gauge_wse pairs < min_raw_gauge_matches
# -----------------------------
if {"wse", "gauge_wse"}.issubset(df_lake_time_series.columns):
    raw_gauge_match_counts = (
        df_lake_time_series
        .assign(
            valid_raw_gauge_pair=lambda d:
                pd.to_numeric(d["wse"], errors="coerce").notna() &
                pd.to_numeric(d["gauge_wse"], errors="coerce").notna()
        )
        .groupby("lake_id")["valid_raw_gauge_pair"]
        .sum()
    )
    low_raw_gauge_match_lake_ids = set(
        raw_gauge_match_counts[
            raw_gauge_match_counts < min_raw_gauge_matches
        ].index.tolist()
    )
else:
    low_raw_gauge_match_lake_ids = set()
    print("Warning: Cannot check raw-WSE/gauge matched pairs because 'wse' or 'gauge_wse' is missing.")
print(
    f"Lakes with fewer than {min_raw_gauge_matches} raw-WSE/gauge matched pairs: "
    f"{len(low_raw_gauge_match_lake_ids)}"
)
print("Lake IDs (<2 raw-WSE/gauge pairs):")
print(sorted(low_raw_gauge_match_lake_ids))
print()

# -----------------------------
# 4) Remove SWOT-null lakes, no-gauge-join lakes, and lakes with <2 raw-WSE/gauge matches
# -----------------------------
invalid_lake_ids = (
    swot_null_lake_ids
    .union(no_gauge_join_lake_ids)
    .union(low_raw_gauge_match_lake_ids)
)
print(f"Total lake_ids to remove from df_lake_stats: {len(invalid_lake_ids)}")
print("Total lake_ids to remove (unique lake IDs to remove from all scenarios------):")
print(sorted(invalid_lake_ids))
print()
#df_lake_stats = df_lake_stats.loc[~df_lake_stats["lake_id"].isin(invalid_lake_ids)].copy()

print()
print("After removal:")
print("  df_lake_stats valid lakes:", len(df_lake_stats))
print()
# Sanity check ends ================================================================



#----------------------------------------------------------------------------------------------------------------------------
# After running the sanity check above in both Version C and Version D:
print("Gauge lake count:", len(df_lake_stats_intact))
# Invalid lakes include the following:
invalid_lake_ids_both_versions = \
    sorted(set([4330037643, 4530056563, 4530331562, 4530335233, 4530379283, 4550008462, 4550009812, 4550021542, 4550022152, \
                4550106402, 6220306162, 6320019482, 6320023772, 6320024182, 6320029102, 6420513573, 8221430182, 8221477072, 8221479422, 8221482072, \
                4330037643, 4530331562, 4530367342, 4530379283, 4540004873, 4550008462, 4550009812, 4550021542, 4550022152, \
                4550106402, 6220306162, 6320019482, 6320023772, 6320024182, 6320029102, 6420513573, 7121032712, 7251059143, 7421074132, 8221430182, \
                8221477072, 8221479422, 8221482072]))
print("Invalid lake count:", len(invalid_lake_ids_both_versions))
# Keep valid lakes; overwrite df_lake_stats for downstream summaries.
df_lake_stats = df_lake_stats.loc[~df_lake_stats["lake_id"].isin(invalid_lake_ids_both_versions)].copy()
#----------------------------------------------------------------------------------------------------------------------------

# Report gauge statistics by source before and after invalid-lake removal.
# unique lake_id counts by gauge_source
c_intact = (
    df_lake_stats_intact.dropna(subset=["lake_id", "gauge_source"])
    .drop_duplicates(subset=["lake_id", "gauge_source"])
    .groupby("gauge_source")["lake_id"]
    .nunique()
    .rename("n_unique_lakes_stats_intact")
)
c_stats = (
    df_lake_stats.dropna(subset=["lake_id", "gauge_source"])
    .drop_duplicates(subset=["lake_id", "gauge_source"])
    .groupby("gauge_source")["lake_id"]
    .nunique()
    .rename("n_unique_lakes_stats")
)
# Combine into one table.
tab = pd.concat([c_intact, c_stats], axis=1).fillna(0).astype(int).sort_index()
print(tab)



# -----------
# FIGURE: Map of valid gauges, color-coded by lake prior area
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

# Note: p_ref_area is the lake prior polygon area in km2

# 1) Build plot_df: unique lake_id in df_lake_stats, with lat/lon/area from df_Hydrocron
lake_ids = pd.Series(
    df_lake_stats["lake_id"].dropna().unique(),
    name="lake_id"
).to_frame()

hc_ll = df_Hydrocron.loc[:, ["lake_id", "p_lat", "p_lon", "p_ref_area"]].copy()

# Ensure numeric fields
hc_ll["p_lat"] = pd.to_numeric(hc_ll["p_lat"], errors="coerce")
hc_ll["p_lon"] = pd.to_numeric(hc_ll["p_lon"], errors="coerce")
hc_ll["p_ref_area"] = pd.to_numeric(hc_ll["p_ref_area"], errors="coerce")

# Keep one lat/lon/area record per lake_id
hc_ll = (
    hc_ll.dropna(subset=["lake_id", "p_lat", "p_lon", "p_ref_area"])
         .groupby("lake_id", as_index=False)[["p_lat", "p_lon", "p_ref_area"]]
         .first()
)

# Merge valid gauge lakes with lat/lon/area
plot_df = (
    lake_ids
    .merge(hc_ll, on="lake_id", how="left")
    .dropna(subset=["p_lat", "p_lon", "p_ref_area"])
    .copy()
)

# LogNorm requires positive values
plot_df = plot_df[plot_df["p_ref_area"] > 0].copy()

# Normalize longitude to [-180, 180]
plot_df["p_lon"] = ((plot_df["p_lon"] + 180) % 360) - 180


# 2) Plot: GCS PlateCarree, full globe, countries + lakes
fig = plt.figure(figsize=(16, 7), dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

# Ocean / land background
ocean_col = "#b9dce8"
land_col  = "whitesmoke"
border_col = "#b0b0b0"
coast_col  = "#8a8a8a"

ax.set_facecolor(ocean_col)
ax.add_feature(cfeature.OCEAN, facecolor=ocean_col, edgecolor="none", zorder=0)
ax.add_feature(cfeature.LAND, facecolor=land_col, edgecolor="none", zorder=1)

# Countries + coastlines
ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor=border_col, zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor=coast_col, zorder=2)

# Lake baselayer: Natural Earth
lakes = cfeature.NaturalEarthFeature(
    "physical", "lakes", "50m",
    edgecolor="#9bbfd0",
    facecolor="#9fcfe3"
)
ax.add_feature(lakes, zorder=2.5, linewidth=0.3, alpha=0.95)

# Optional rivers for context
rivers = cfeature.NaturalEarthFeature(
    "physical", "rivers_lake_centerlines", "50m",
    edgecolor="#bcd6df",
    facecolor="none"
)
ax.add_feature(rivers, zorder=2.4, linewidth=0.3, alpha=0.7)

# Valid gauge/lake dots, color-coded by prior lake area
sc = ax.scatter(
    plot_df["p_lon"], plot_df["p_lat"],
    c=plot_df["p_ref_area"],
    cmap="viridis",
    norm=LogNorm(
        vmin=plot_df["p_ref_area"].min(),
        vmax=plot_df["p_ref_area"].max()
    ),
    transform=ccrs.PlateCarree(),
    s=22,
    edgecolors="#2b2b2b",
    linewidths=0.35,
    alpha=0.9,
    zorder=5
)

ax.axis("off")

# Colorbar
cbar = plt.colorbar(
    sc,
    ax=ax,
    orientation="horizontal",
    pad=0.02,
    fraction=0.045,
    shrink=0.55
)
cbar.set_label("Prior lake area (km²)", fontsize=16)
cbar.ax.tick_params(labelsize=14)

# Remove all margins
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

out_png = os.path.join(
    plots_dir,
    f"global_lake_map_GCS_countries_lakes_area_{version_filename}.png"
)
out_pdf = os.path.join(
    plots_dir,
    f"global_lake_map_GCS_countries_lakes_area_{version_filename}.pdf"
)

n_points = len(plot_df)
min_area = plot_df["p_ref_area"].min()
max_area = plot_df["p_ref_area"].max()
median_area = plot_df["p_ref_area"].median()
mean_area = plot_df["p_ref_area"].mean()
print(f"Number of points plotted: {n_points}")
print(f"Minimum plotted lake area: {min_area:.4f} km²")
print(f"Maximum plotted lake area: {max_area:.4f} km²")
print(f"Median plotted lake area: {median_area:.4f} km²")
print(f"Mean plotted lake area: {mean_area:.4f} km²")

plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0)
plt.show()
# -----------



"""
Present summative statistics for all validated lakes

Recall: major outputs from sections above include:
    df_lake_time_series
    df_lake_stats (duplicated to df_lake_stats_valid_backup; df_lake_stats will later be replaced)
"""
# Make a backup copy before applying the reference-lake selection.
df_lake_stats_valid_backup = df_lake_stats.copy()

# Compute the proportion of lakes for each filtering scenario
proportion_filter_attempt_1 = (len(df_lake_stats_valid_backup[(df_lake_stats_valid_backup['filter_attempt']=='1_strict')])
                               / len(df_lake_stats_valid_backup))*100
print('Proportion (%) of lakes with strict filter (attempt 1): ' + str(proportion_filter_attempt_1))

proportion_filter_attempt_2 = (len(df_lake_stats_valid_backup[(df_lake_stats_valid_backup['filter_attempt']=='2_lenient')])
                               / len(df_lake_stats_valid_backup))*100
print('Proportion (%) of lakes with lenient filter (attempt 2): ' + str(proportion_filter_attempt_2))

proportion_filter_attempt_3 = (len(df_lake_stats_valid_backup[(df_lake_stats_valid_backup['filter_attempt']=='3_tukey')])
                               / len(df_lake_stats_valid_backup))*100
print('Proportion (%) of lakes with baseline tukey IQR filter (attempt 3): ' + str(proportion_filter_attempt_3))

proportion_filter_attempt_4 = (len(df_lake_stats_valid_backup[(df_lake_stats_valid_backup['filter_attempt']=='4_none')])
                               / len(df_lake_stats_valid_backup))*100
print('Proportion (%) of lakes with no filter (attempt 4): ' + str(proportion_filter_attempt_4))

proportion_filter_no_data = (len(df_lake_stats_valid_backup[(df_lake_stats_valid_backup['filter_attempt']=='no data')])
                             / len(df_lake_stats_valid_backup))*100
print('Proportion (%) of lakes with no data (attempt 0): ' + str(proportion_filter_no_data))

print('Total valid (100%): ' +
      str(proportion_filter_attempt_1 +
          proportion_filter_attempt_2 +
          proportion_filter_attempt_3 +
          proportion_filter_attempt_4 +
          proportion_filter_no_data))


# =============================================================================
# Select reference lakes for validation summaries and figures
# =============================================================================
# The validation summaries below are computed from a reference set of lakes.
# This selection is applied after HALF filtering and before figure generation.
#
# Available options:
#   - "strict_only":
#       Keep only lakes successfully processed by Attempt 1 ("1_strict").
#       These lakes already satisfy the strict HALF temporal-coverage criteria
#       used during filtering.
#
#   - "strict_plus_gap_checked_lenient":
#       Keep all Attempt-1 lakes, plus selected Attempt-2 ("2_lenient") lakes
#       whose final retained HALF time series still satisfies post-filtering
#       temporal-coverage criteria configured in VALIDATION_CONFIG:
#           * retained time span >= reference_min_retained_span_days
#           * maximum retained-observation gap <= reference_max_retained_gap_days
#
# The default option follows Trudel et al. (2026): it includes lakes that failed
# the strict internal gap criterion but still have sufficiently continuous final
# retained records for validation.
# reference_lake_selection is configured in VALIDATION_CONFIG.

if reference_lake_selection == "strict_only":
    keep_mask = df_lake_stats_valid_backup["filter_attempt"].isin(["1_strict"])

elif reference_lake_selection == "strict_plus_gap_checked_lenient":
    # Work from the observation-level output to evaluate the final retained
    # temporal coverage of each lake.
    df_select = df_lake_time_series.copy()
    df_select = df_select.sort_values(["lake_id", "datetime"])

    # Per-lake filtering attempt, stored at the observation level.
    filter_attempt_by_lake = (
        df_select.groupby("lake_id", sort=False)["filter_attempt"]
        .first()
        .str.lower()
        .str.strip()
    )

    # Retained HALF observations only.
    df_half_retained = df_select[df_select["filter_flag"] == 1].copy()
    retained_by_lake = df_half_retained.groupby("lake_id", sort=False)

    # Final retained temporal coverage diagnostics.
    retained_span_days = (
        retained_by_lake["datetime"].max() -
        retained_by_lake["datetime"].min()
    ).dt.days

    retained_max_gap_days = retained_by_lake["datetime"].apply(
        lambda s: s.diff().max() / pd.Timedelta(days=1)
    )

    # Attempt-1 lakes are kept directly.
    strict_ids = filter_attempt_by_lake[
        filter_attempt_by_lake.eq("1_strict")
    ].index

    # Attempt-2 lakes are kept only if their final retained time series is
    # sufficiently long and not too fragmented.
    lenient_ids = filter_attempt_by_lake[
        filter_attempt_by_lake.eq("2_lenient")
    ].index

    lenient_ids_with_valid_span = retained_span_days[
        retained_span_days.fillna(0) >= reference_min_retained_span_days
    ].index

    lenient_ids_with_valid_gap = retained_max_gap_days[
        retained_max_gap_days.fillna(np.inf) <= reference_max_retained_gap_days
    ].index

    keep_ids = strict_ids.union(
        lenient_ids
        .intersection(lenient_ids_with_valid_span)
        .intersection(lenient_ids_with_valid_gap)
    )

    keep_mask = df_lake_stats_valid_backup["lake_id"].isin(keep_ids)

else:
    raise ValueError(
        f"Unsupported reference_lake_selection={reference_lake_selection!r}. "
        "Use 'strict_only' or 'strict_plus_gap_checked_lenient'."
    )

# Save the full lake-level statistics table with a 1/0 flag indicating whether
# each lake is included in the reference validation subset.
df_lake_stats_valid_backup = df_lake_stats_valid_backup.copy()
df_lake_stats_valid_backup["keep_mask"] = keep_mask.astype("int8")

df_lake_stats_valid_backup.to_csv(
    os.path.join(plots_dir, f"validation_lake_stats_{version_filename}.csv"),
    index=False,
)

# Use only the selected reference lakes for subsequent validation summaries
# and figures. Do not apply another filter_attempt subset below; keep_mask is
# the single source of truth for the validation-lake selection.
df_lake_stats = df_lake_stats_valid_backup[keep_mask].copy()

proportion_kept_lakes = (
    len(df_lake_stats) / len(df_lake_stats_valid_backup)
) * 100

print(f"Proportion (%) of lakes kept: {proportion_kept_lakes:.2f}")



# =============================================================================
# Consolidated figure making
# =============================================================================
# Figures and summary statistics below use df_lake_stats, which has already been
# subset to the selected reference lakes above.
# =============================================================================

# Consistent colors for raw LakeSP, benchmark filters, and HALF.
raw_color = "#7F7F7F"
stringent_color = "#0072B2"
permissive_color = "#E69F00"
HALF_color = "#CC79A7"

# Lakes available for gauge-based cross-filter validation.
# These are reference lakes that also have valid HALF-vs-gauge MAE after datum
# alignment and zero-residual exclusion. This list is useful for figures or
# statistics that compare raw LakeSP, benchmark filters, and HALF over the same
# set of gauge-validated lakes.
lakes_for_validation = df_lake_stats.loc[
    df_lake_stats["mae"].notna(), "lake_id"
].tolist()

proportion_validated_lakes = (
    len(lakes_for_validation) / len(df_lake_stats_valid_backup)
) * 100

print(
    "Proportion (%) of lakes available for gauge-based cross-filter validation: "
    f"{proportion_validated_lakes:.2f}"
)

# Customizable percentile used for non-median summary statistics and CDF reference lines.
# For higher-is-worse metrics, use this percentile directly.
# For higher-is-better metrics plotted as standard CDFs, use the complementary percentile.
SUMMARY_PERCENTILE = 68.27 #user-defined summary percentile: 68.27, 95, 99, etc.
def _format_percentile_label_value(p):
    return f"{p:.2f}".rstrip("0").rstrip(".")

# Convert the user-defined summary percentile into clean text for labels.
# Example: 68.27 -> "68.27"; 95.0 -> "95".
SUMMARY_PERCENTILE_TEXT = _format_percentile_label_value(SUMMARY_PERCENTILE)

# Label for ordinary percentile reporting, used for lower-is-better metrics
# such as MAE, normalized error, and residual error.
# Example: SUMMARY_PERCENTILE = 68.27 -> "P68.27".
SUMMARY_PERCENTILE_LABEL = f"P{SUMMARY_PERCENTILE_TEXT}"

# For higher-is-better metrics, such as correlation and retention rate,
# the threshold exceeded by SUMMARY_PERCENTILE percent of lakes is located at
# CDF = 100 - SUMMARY_PERCENTILE in a standard low-to-high CDF.
# Example: 68.27% exceedance -> CDF level = 31.73.
SUMMARY_EXCEEDANCE_CDF = 100.0 - SUMMARY_PERCENTILE

# Convert the exceedance CDF level into clean text for labels.
# Example: 31.73 -> "31.73"; 5.0 -> "5".
SUMMARY_EXCEEDANCE_CDF_TEXT = _format_percentile_label_value(SUMMARY_EXCEEDANCE_CDF)

# Percentile label for higher-is-better metrics.
# Example: 68.27% exceedance corresponds to "P31.73";
#          95% exceedance corresponds to "P5".
SUMMARY_EXCEEDANCE_PERCENTILE_LABEL = f"P{SUMMARY_EXCEEDANCE_CDF_TEXT}"

# Short text label used in plot annotations and printed summaries for
# higher-is-better metrics.
# Example: "68.27% exceedance" or "95% exceedance".
SUMMARY_EXCEEDANCE_LABEL = f"{SUMMARY_PERCENTILE_TEXT}% exceedance"


# =============================================================================
# Print global summary percentile of extra residual relative variability error
# No plotting; this only prints values for std, amplitude, and IDR.
#
# Definition for each lake and metric:
#   r_daily    = |V_SWOT_daily - V_gauge_daily| / V_SWOT_daily
#   r_retained = |V_SWOT_retained - V_gauge_daily| / V_SWOT_daily #do not use V_gauge_daily
#   r_extra    = sqrt(max(0, r_daily^2 - r_retained^2))
#
# Interpretation:
#   r_extra quantifies the additional residual relative variability error in the
#   daily reconstructed/interpolated LakeSP variability that is not already
#   represented by the retained-observation variability error. The error is
#   normalized by daily reconstructed/interpolated LakeSP variability so that it
#   can be applied to LakeSP-derived time series where gauge variability is unavailable.
# =============================================================================
def _print_extra_residual_relative_variability(df_stats, lakes_for_validation):
    dfv_extra = df_stats[df_stats["lake_id"].isin(lakes_for_validation)].copy()

    metric_specs = [
        {
            "label": "std",
            "gauge_daily": "var_std_gauge_daily",
            "swot_daily": "var_std_swot_daily",
            "swot_retained": "var_std_swot_retained",
        },
        {
            "label": "amplitude",
            "gauge_daily": "var_amp_gauge_daily",
            "swot_daily": "var_amp_swot_daily",
            "swot_retained": "var_amp_swot_retained",
        },
        {
            "label": "IDR",
            "gauge_daily": "var_p10p90_gauge_daily",
            "swot_daily": "var_p10p90_swot_daily",
            "swot_retained": "var_p10p90_swot_retained",
        },
    ]

    print("\n=== Extra residual relative WSE variability error: HALF ===")
    print("Definition:")
    print("  r_daily    = |V_SWOT_daily - V_gauge_daily| / V_SWOT_daily")
    print("  r_retained = |V_SWOT_retained - V_gauge_daily| / V_SWOT_daily")
    print("  r_extra    = sqrt(max(0, r_daily^2 - r_retained^2))")

    for spec in metric_specs:
        label = spec["label"]
        gcol = spec["gauge_daily"]
        dcol = spec["swot_daily"]
        rcol = spec["swot_retained"]

        missing = [c for c in [gcol, dcol, rcol] if c not in dfv_extra.columns]
        if missing:
            print(f"  {label:<10s}: missing columns {missing}")
            continue

        g = pd.to_numeric(dfv_extra[gcol], errors="coerce")
        d = pd.to_numeric(dfv_extra[dcol], errors="coerce")
        r = pd.to_numeric(dfv_extra[rcol], errors="coerce")

        valid = (
            np.isfinite(g) &
            np.isfinite(d) &
            np.isfinite(r) &
            (d > 0)
        )

        if valid.sum() == 0:
            print(f"  {label:<10s}: N=0, {SUMMARY_PERCENTILE_LABEL}=NaN")
            continue

        r_daily = np.abs(d[valid] - g[valid]) / d[valid]
        r_retained = np.abs(r[valid] - g[valid]) / d[valid]

        r_extra = np.sqrt(
            np.maximum(
                0,
                r_daily.to_numpy(dtype=float)**2 -
                r_retained.to_numpy(dtype=float)**2
            )
        )

        r_extra = r_extra[np.isfinite(r_extra)]

        if r_extra.size == 0:
            print(f"  {label:<10s}: N=0, {SUMMARY_PERCENTILE_LABEL}=NaN")
        else:
            print(
                f"  {label:<10s}: "
                f"N={r_extra.size:4d}, "
                f"{SUMMARY_PERCENTILE_LABEL} extra residual={np.nanpercentile(r_extra, SUMMARY_PERCENTILE):.4f}"
            )

_print_extra_residual_relative_variability(df_lake_stats, lakes_for_validation)
# =============================================================================

subset_cross_pass_corrected = df_lake_stats[
    (df_lake_stats["lake_id"].isin(lakes_for_validation)) &
    (df_lake_stats["intra_cycle_flag"] == 1)
].copy()
print("the number of validated lakes: " + str(len(lakes_for_validation)))
print("the number of lakes with cross-pass bias correction: " + str(len(subset_cross_pass_corrected)))
print(str(len(subset_cross_pass_corrected) / len(lakes_for_validation) * 100.0)+"%")


# FIGURE: histogram summary
def plot_hist_overlay_4groups(df_lake_stats_valid_backup,
                             lakes_for_validation,
                             col="lake_area",
                             bins_per_decade=10,
                             outname=None):
    """
    Overlay histogram with 4 groups:
      - All lakes (df_lake_stats_valid_backup): filled bars (keep existing blue style)
      - 1_strict lakes: red step line
      - kept lakes (keep_mask==1): orange step line
      - validated lakes (lake_id in lakes_for_validation): black step line

    Also prints counts and percentages relative to df_lake_stats_valid_backup.
    Uses log-spaced bins based on the pooled positive values across groups.
    """

    # -------------------------
    # Build group data
    # -------------------------
    # All lakes
    all_df = df_lake_stats_valid_backup.copy()

    # Strict-only (attempt 1)
    strict_df = all_df[all_df["filter_attempt"] == "1_strict"]

    # Kept (selection mask already stored as 0/1)
    kept_df = all_df[all_df["keep_mask"] == 1]

    # Validated lakes (passed downstream filters + has gauge validation)
    validated_df = all_df[all_df["lake_id"].isin(lakes_for_validation)]

    # Extract positive values for histogram variable
    def _pos_series(df, colname):
        s = df[colname].dropna().astype(float)
        return s[s > 0]

    data_all = _pos_series(all_df, col)
    data_strict = _pos_series(strict_df, col)
    data_kept = _pos_series(kept_df, col)
    data_validated = _pos_series(validated_df, col)

    # -------------------------
    # Print counts + percentages
    # -------------------------
    n_total = len(all_df)
    n_strict = len(strict_df)
    n_kept = len(kept_df)
    n_validated = len(validated_df)

    def _pct(n, d):
        return 100.0 * n / d if d else np.nan

    print("\n=== Lake counts and percentages (relative to df_lake_stats_valid_backup) ===")
    print(f"All valid lakes:       n={n_total:4d}  ({_pct(n_total, n_total):6.2f}%)")
    print(f"1_strict lakes:  n={n_strict:4d}  ({_pct(n_strict, n_total):6.2f}%)")
    print(f"Kept lakes:      n={n_kept:4d}  ({_pct(n_kept, n_total):6.2f}%)")
    print(f"Validated lakes: n={n_validated:4d}  ({_pct(n_validated, n_total):6.2f}%)")

    # -------------------------
    # Bin construction (log)
    # -------------------------
    pooled = pd.concat([data_all, data_strict, data_kept, data_validated], ignore_index=True)
    pooled = pooled[pooled > 0]
    if pooled.empty:
        print("No positive values available to plot.")
        return

    min_val, max_val = pooled.min(), pooled.max()
    n_bins = max(5, int(np.log10(max_val / min_val)) * bins_per_decade)
    bins = np.logspace(np.log10(min_val), np.log10(max_val), n_bins)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(8, 6))

    # Base: all lakes
    plt.hist(
        data_all, bins=bins,
        edgecolor="black",
        color="skyblue", alpha=0.6,
        label=f"All lakes: {n_total}"
    )

    #1_strict: red
    plt.hist(
        data_strict, bins=bins,
        histtype="step", linewidth=2.5,
        color="red",
        label=f"HALF (Level-1 strict): {n_strict}"
    )

    # kept: orange
    plt.hist(
        data_kept, bins=bins,
        histtype="step", linewidth=2.5,
        color="orange",
        label=f"HALF: {n_kept}"
    )

    # validated: black
    plt.hist(
        data_validated, bins=bins,
        histtype="step", linewidth=2.5,
        color="black",
        label=f"Validated: {n_validated}"
    )

    plt.xscale("log")
    plt.xlabel("Lake area (km²)", fontsize=18)
    plt.ylabel("Number of lakes", fontsize=18)

    plt.rcParams["font.family"] = "Arial"
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()

    if outname is None:
        outname_png = f"Lake_validation_hist_2_{script_version}_{version_filename}.png"
        outname_pdf = f"Lake_validation_hist_2_{script_version}_{version_filename}.pdf"

    plt.savefig(os.path.join(plots_dir, outname_png), bbox_inches="tight")
    plt.savefig(os.path.join(plots_dir, outname_pdf), bbox_inches="tight")
    plt.show()

plot_hist_overlay_4groups(df_lake_stats_valid_backup, lakes_for_validation, col="lake_area")



if lakes_for_validation:
    # -----------------------------
    # Helpers
    # -----------------------------
    def _max_gap_days(dt_series: pd.Series) -> float:
        s = pd.to_datetime(dt_series).dropna().sort_values()
        if len(s) < 2:
            return np.nan
        diffs = s.diff().dropna() / pd.Timedelta(days=1)
        return float(np.nanmax(diffs)) if len(diffs) else np.nan

    def _median_gap_days(dt_series: pd.Series) -> float:
        s = pd.to_datetime(dt_series).dropna().sort_values()
        if len(s) < 2:
            return np.nan
        diffs = s.diff().dropna() / pd.Timedelta(days=1)
        return float(np.nanmedian(diffs)) if len(diffs) else np.nan

    def _as_box_data(arr):
        """Matplotlib boxplot cannot take empty arrays; return None if <1 finite."""
        v = np.asarray(arr, dtype=float)
        v = v[np.isfinite(v)]
        return v if v.size >= 1 else None

    def _darken(hex_color, factor=0.7):
        """factor<1 makes darker; factor=0.7 is a good default."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * factor); g = int(g * factor); b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _grouped_boxplot(ax, data_by_group, base_positions, offsets, colors,
                         widths=0.16, alpha=0.65, showfliers=False,
                         flier_ms=3, median_lw=2.0):
        """
        Grouped boxplots at categorical base_positions.
        data_by_group: dict[group] -> list of arrays (len = n_categories)
        offsets: dict[group] -> float
        """
        for gname, series_list in data_by_group.items():
            positions = [p + offsets[gname] for p in base_positions]

            for arr, x0 in zip(series_list, positions):
                v = _as_box_data(arr)
                if v is None:
                    continue

                bp = ax.boxplot(
                    [v],
                    positions=[x0],
                    widths=widths,
                    patch_artist=True,
                    showfliers=showfliers,
                    manage_ticks=False,
                    whis=(5, 95)   # whiskers at 5th and 95th percentiles
                )

                edge_col = colors[gname]
                median_col = edge_col #_darken(edge_col, 0.65)  # darker but same hue. #'black'

                # boxes (face + boundary)
                for box in bp['boxes']:
                    box.set_facecolor(colors[gname])
                    box.set_edgecolor(edge_col)
                    box.set_alpha(alpha)
                    box.set_linewidth(0.9)

                # medians (darker same hue)
                for med in bp['medians']:
                    med.set_color(median_col)
                    med.set_linewidth(median_lw)

                # whiskers + caps (same hue as boundary)
                for w in bp['whiskers']:
                    w.set_color(edge_col)
                    w.set_linewidth(0.9)
                for cap in bp['caps']:
                    cap.set_color(edge_col)
                    cap.set_linewidth(0.9)

                # fliers (if shown) in same hue
                if showfliers:
                    for fl in bp['fliers']:
                        fl.set_marker('o')
                        fl.set_markersize(flier_ms)
                        fl.set_markerfacecolor(colors[gname])
                        fl.set_markeredgecolor(edge_col)
                        fl.set_alpha(min(1.0, alpha + 0.15))

    # =============================================================================
    # FIGURE: validation statistics (4 panels)
    # =============================================================================
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 12,
        "figure.titlesize": 24
    })

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()

    # Panel 1: anomalies scatter
    min_val, max_val = [], []
    swot_anom_half, swot_anom_raw, swot_anom_b1, swot_anom_b2, gauge_anom_all = [], [], [], [], []

    for lake_id in lakes_for_validation:
        sub = df_lake_time_series[df_lake_time_series['lake_id'] == lake_id]
        wse_mean = np.nanmean(sub['wse_adjusted'])

        g = sub['gauge_wse_bias_corrected'] - wse_mean
        half = sub['wse_adjusted'] - wse_mean
        raw  = sub['wse'] - wse_mean
        b1   = sub['wse_benchmark_stringent'] - wse_mean
        b2   = sub['wse_benchmark_permissive'] - wse_mean

        swot_anom_half.extend(half.tolist())
        swot_anom_raw.extend(raw.tolist())
        swot_anom_b1.extend(b1.tolist())
        swot_anom_b2.extend(b2.tolist())
        gauge_anom_all.extend(g.tolist())

        min_val.append(min(np.nanmin(g), np.nanmin(half)))
        max_val.append(max(np.nanmax(g), np.nanmax(half)))

    alpha = 0.25
    axes[0].scatter(gauge_anom_all, swot_anom_raw,  color=raw_color,        label='Raw',        s=30, linewidth=0, alpha=0.3)
    axes[0].scatter(gauge_anom_all, swot_anom_b2,   color=permissive_color, label='Permissive', s=30, linewidth=0, alpha=alpha)
    axes[0].scatter(gauge_anom_all, swot_anom_half, color=HALF_color,       label='HALF',       s=30, linewidth=0, alpha=alpha)
    axes[0].scatter(gauge_anom_all, swot_anom_b1,   color=stringent_color,  label='Stringent',  s=20, linewidth=0, alpha=alpha)

    vmin, vmax = min(min_val), max(max_val)
    #axes[0].plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1)
    axes[0].plot([-40, 40], [-40, 40], 'k--', linewidth=1)
    axes[0].set_xlim(vmin, vmax)
    #axes[0].set_ylim(vmin, vmax)
    axes[0].set_xlim(-40, 40)
    axes[0].set_ylim(-40, 40)
    axes[0].set_xticks([-40, -20, 0, 20, 40])
    axes[0].set_xlabel('Gauge WSE anomaly (m)')
    axes[0].set_ylabel('LakeSP WSE anomaly (m)')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    subset_stats = df_lake_stats[df_lake_stats['lake_id'].isin(lakes_for_validation)].copy()

    # Panel 2: seasonal variability scatter
    axes[1].scatter(subset_stats.var_std_gauge_daily, subset_stats.var_std_swot_daily_raw, color=raw_color, s=30, linewidth=0, alpha=0.3)
    axes[1].scatter(subset_stats.var_std_gauge_daily, subset_stats.var_std_swot_daily_benchmark_stringent,  color=stringent_color, s=30, linewidth=0, alpha=alpha)
    axes[1].scatter(subset_stats.var_std_gauge_daily, subset_stats.var_std_swot_daily_benchmark_permissive,  color=permissive_color, s=30, linewidth=0, alpha=alpha)
    axes[1].scatter(subset_stats.var_std_gauge_daily, subset_stats.var_std_swot_daily,            color=HALF_color, s=30, linewidth=0, alpha=alpha)

    vals_min = [np.nanmin(subset_stats.var_std_gauge_daily), np.nanmin(subset_stats.var_std_swot_daily_raw), np.nanmin(subset_stats.var_std_swot_daily)]
    valid_vals = [v for v in vals_min if not np.isnan(v) and v > 0]
    vmin = min(valid_vals) if valid_vals else np.nan
    vmax = max(np.nanmax(subset_stats.var_std_gauge_daily), np.nanmax(subset_stats.var_std_swot_daily_raw), np.nanmax(subset_stats.var_std_swot_daily))

    #axes[1].plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1)
    axes[1].plot([0.01, 100], [0.01, 100], 'k--', linewidth=1)
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    #axes[1].set_xlim(vmin, vmax); axes[1].set_ylim(vmin, vmax)
    axes[1].set_xlim(0.01, 100); axes[1].set_ylim(0.01, 100)
    axes[1].set_xlabel('Gauge WSE variability (m)')
    axes[1].set_ylabel('LakeSP WSE variability (m)')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Panel 3: correlation CDF
    def _cdf(arr):
        arr = np.asarray(arr, dtype=float)
        arr = np.sort(arr[np.isfinite(arr)])
        if len(arr) == 0:
            return arr, arr
        return arr, np.arange(1, len(arr) + 1) / len(arr) * 100.0

    def _annotate_half_cdf_percentiles_inside(
        ax,
        half_vals,
        color,
        percentiles=(50, SUMMARY_EXCEEDANCE_CDF),
        value_fmt="{:.2f}",
        fontsize=9,
        x_pad_frac=0.06,
        y_pad_frac=0.03
    ):
        """
        Annotate the HALF CDF curve at selected percentiles while keeping
        labels inside the plotting box. CDF y-axis is in percentage units.

        Labels are placed at a fixed right-side location so that their right
        edges align across percentile annotations. Vertically, labels use the
        original placement: immediately above their corresponding reference lines.
        """
        half_vals = np.asarray(half_vals, dtype=float)
        half_vals = half_vals[np.isfinite(half_vals)]

        if len(half_vals) == 0:
            return

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        x_pad = (xmax - xmin) * x_pad_frac
        y_pad = (ymax - ymin) * y_pad_frac

        # Fixed right-side anchor for all labels.
        # ha="right" aligns the right edge of all labels.
        x_text = xmax - x_pad

        for p in percentiles:
            x_val = np.nanpercentile(half_vals, p)
            y_val = p

            if np.isclose(p, 50, atol=1e-8):
                p_label = "50%"
            elif np.isclose(p, SUMMARY_EXCEEDANCE_CDF, atol=1e-8):
                p_label = SUMMARY_EXCEEDANCE_LABEL
            elif abs(p - round(p)) < 1e-8:
                p_label = f"{int(round(p))}%"
            else:
                p_label = f"{p:.2f}%"

            label = f"{value_fmt.format(x_val)} ({p_label})"

            # Original vertical placement: use the line value itself.
            # With va="bottom", the label sits immediately above the dashed line.
            y_text = np.clip(y_val, ymin + y_pad, ymax - y_pad)

            ax.text(
                x_text,
                y_text,
                label,
                color=color,
                fontsize=fontsize,
                ha="right",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.95
                ),
                clip_on=True,
                zorder=10
            )

    # -----------------------------
    # Panel 3: correlation CDF
    # -----------------------------
    x, y = _cdf(subset_stats['correlation_raw'].values)
    axes[2].plot(x, y, color=raw_color, label='Raw')

    x, y = _cdf(subset_stats['correlation_benchmark_stringent'].values)
    axes[2].plot(x, y, color=stringent_color, label='Stringent')

    x, y = _cdf(subset_stats['correlation_benchmark_permissive'].values)
    axes[2].plot(x, y, color=permissive_color, label='Permissive')

    half_corr_vals = subset_stats['correlation'].values
    x, y = _cdf(half_corr_vals)
    axes[2].plot(x, y, color=HALF_color, label='HALF')

    # Reference lines in percentage units
    axes[2].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(SUMMARY_EXCEEDANCE_CDF, color="black", linestyle="--", linewidth=1)

    axes[2].set_xlim(-1, 1)
    axes[2].set_ylim(0, 100)
    axes[2].set_xticks([-1, -0.5, 0, 0.5, 1])
    axes[2].set_xlabel('Correlation (Pearson)')
    axes[2].set_ylabel('CDF (%)')
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Label HALF median and summary exceedance statistics inside the plotting box
    _annotate_half_cdf_percentiles_inside(
        axes[2],
        half_corr_vals,
        color=HALF_color,
        percentiles=(50, SUMMARY_EXCEEDANCE_CDF),
        value_fmt="{:.2f}",
        fontsize=13
    )


    # -----------------------------
    # Panel 4: retention CDF
    # -----------------------------
    x, y = _cdf(subset_stats['retention_rate_benchmark_stringent'].values * 100.0)
    axes[3].plot(x, y, color=stringent_color, label='Stringent')

    x, y = _cdf(subset_stats['retention_rate_benchmark_permissive'].values * 100.0)
    axes[3].plot(x, y, color=permissive_color, label='Permissive')

    half_ret_vals = subset_stats['retention_rate'].values * 100.0
    x, y = _cdf(half_ret_vals)
    axes[3].plot(x, y, color=HALF_color, label='HALF')

    # Reference lines in percentage units
    axes[3].axhline(50, color="black", linestyle="--", linewidth=1)
    axes[3].axhline(SUMMARY_EXCEEDANCE_CDF, color="black", linestyle="--", linewidth=1)

    axes[3].set_xlim(0, 100)
    axes[3].set_ylim(0, 100)
    axes[3].set_xlabel('Retention rate (%)')
    axes[3].set_ylabel('CDF (%)')
    axes[3].grid(True, which='both', linestyle='--', linewidth=0.5)

    # Label HALF median and summary exceedance statistics inside the plotting box
    _annotate_half_cdf_percentiles_inside(
        axes[3],
        half_ret_vals,
        color=HALF_color,
        percentiles=(50, SUMMARY_EXCEEDANCE_CDF),
        value_fmt="{:.1f}",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'Lake_validation_stats_{script_version}_{version_filename}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f'Lake_validation_stats_{script_version}_{version_filename}.pdf'), bbox_inches='tight')
    # plt.close(fig)


    # -----------------------------
    # PRINT STATS
    # -----------------------------
    subset_stats = df_lake_stats[df_lake_stats["lake_id"].isin(lakes_for_validation)].copy()

    def _safe_rel_err(num, den):
        """|num|/den with den<=0 -> NaN."""
        den = den.astype(float)
        num = num.astype(float)
        out = np.full(len(num), np.nan, dtype=float)
        m = np.isfinite(num) & np.isfinite(den) & (den > 0)
        out[m] = np.abs(num[m]) / den[m]
        return out

    def _p50_summary(x):
        x = np.asarray(x, dtype=float)
        return np.nanpercentile(x, 50), np.nanpercentile(x, SUMMARY_PERCENTILE)

    def _p50_summary_exceedance(x):
        x = np.asarray(x, dtype=float)
        return np.nanpercentile(x, 50), np.nanpercentile(x, SUMMARY_EXCEEDANCE_CDF)

    # ---- MAE (if available) ----
    print(f"MAE (m): P50, {SUMMARY_PERCENTILE_LABEL}")
    mae_cols = [
        ("Raw",        "mae_raw"),
        ("Stringent",  "mae_benchmark_stringent"),
        ("Permissive", "mae_benchmark_permissive"),
        ("HALF",       "mae"),
    ]
    for label, col in mae_cols:
        if col in subset_stats.columns:
            p50, p_summary = _p50_summary(subset_stats[col].values)
            print(f"  {label:<10s}: {p50:.4f}, {p_summary:.4f}")
        else:
            print(f"  {label:<10s}: (missing column '{col}')")

    # ---- Relative error of WSE variability (std) ----
    print(f"\nNormalized variability error |σ_swot - σ_gauge| / σ_gauge: P50, {SUMMARY_PERCENTILE_LABEL}")
    rel_cols = [
        ("Raw",        "var_std_swot_daily_raw"),
        ("Stringent",  "var_std_swot_daily_benchmark_stringent"),
        ("Permissive", "var_std_swot_daily_benchmark_permissive"),
        ("HALF",       "var_std_swot_daily"),
    ]
    if "var_std_gauge_daily" in subset_stats.columns:
        den = subset_stats["var_std_gauge_daily"].values
        for label, col in rel_cols:
            if col in subset_stats.columns:
                rel = _safe_rel_err(subset_stats[col].values - den, den)
                p50, p_summary = _p50_summary(rel)
                print(f"  {label:<10s}: {p50:.4f}, {p_summary:.4f}")
            else:
                print(f"  {label:<10s}: (missing column '{col}')")
    else:
        print("  (missing column 'var_std_gauge_daily')")

    # ---- Correlation ----
    print(f"\nCorrelation (Pearson): P50, {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})")
    corr_cols = [
        ("Raw",        "correlation_raw"),
        ("Stringent",  "correlation_benchmark_stringent"),
        ("Permissive", "correlation_benchmark_permissive"),
        ("HALF",       "correlation"),
    ]
    for label, col in corr_cols:
        if col in subset_stats.columns:
            p50, p_exceedance = _p50_summary_exceedance(subset_stats[col].values)
            print(f"  {label:<10s}: P50={p50:.3f}, {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})={p_exceedance:.3f}")
        else:
            print(f"  {label:<10s}: (missing column '{col}')")

    # ---- Retention rate (in %) ----
    print(f"\nRetention rate (%): P50, {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})")
    ret_cols = [
        ("Stringent",  "retention_rate_benchmark_stringent"),
        ("Permissive", "retention_rate_benchmark_permissive"),
        ("HALF",       "retention_rate"),
    ]
    for label, col in ret_cols:
        if col in subset_stats.columns:
            vals_pct = subset_stats[col].values * 100.0
            p50, p_exceedance = _p50_summary_exceedance(vals_pct)
            print(f"  {label:<10s}: P50={p50:.1f}%, {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})={p_exceedance:.1f}%")
        else:
            print(f"  {label:<10s}: (missing column '{col}')")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # =============================================================================
    # FIGURE: BOX plot of per-lake retention rate by SWOT cycle (triplets), label EVERY cycle/date
    # =============================================================================
    # --- cycle table prep
    cyc_local = cyc.copy()
    cyc_local.columns = [c.strip().lower().replace(' ', '_') for c in cyc_local.columns]
    cid_col = 'cycle'
    start_col = 'start_time_(utc)'

    cyc_local[cid_col] = pd.to_numeric(cyc_local[cid_col], errors='coerce').astype('Int64')
    cyc_local = cyc_local.dropna(subset=[cid_col]).copy()
    cyc_local[cid_col] = cyc_local[cid_col].astype(int)
    cyc_local[start_col] = pd.to_datetime(cyc_local[start_col], errors='coerce')

    if 'start_cycle' in locals() and 'end_cycle' in locals():
        sc, ec = int(start_cycle), int(end_cycle)
    else:
        sc, ec = int(cyc_local[cid_col].min()), int(cyc_local[cid_col].max())

    cycles = list(range(sc, ec + 1))
    date_map = (cyc_local[cyc_local[cid_col].between(sc, ec)].set_index(cid_col)[start_col].to_dict())

    tmp = df_lake_time_series.loc[df_lake_time_series['lake_id'].isin(lakes_for_validation),
                     ['lake_id', 'cycle_id', 'wse', 'wse_adjusted', 'wse_benchmark_stringent', 'wse_benchmark_permissive']].copy()
    tmp = tmp[tmp['cycle_id'].between(sc, ec)]

    grp = (tmp.groupby(['lake_id', 'cycle_id'], as_index=False)
              .agg(n_raw=('wse', lambda s: s.notna().sum()),
                   n_half=('wse_adjusted', lambda s: s.notna().sum()),
                   n_b1=('wse_benchmark_stringent', lambda s: s.notna().sum()),
                   n_b2=('wse_benchmark_permissive', lambda s: s.notna().sum())))
    grp['ret_half'] = np.where(grp['n_raw'] > 0, grp['n_half'] / grp['n_raw']*100.0, np.nan)
    grp['ret_b1']   = np.where(grp['n_raw'] > 0, grp['n_b1']   / grp['n_raw']*100.0, np.nan)
    grp['ret_b2']   = np.where(grp['n_raw'] > 0, grp['n_b2']   / grp['n_raw']*100.0, np.nan)

    data_b1   = [grp.loc[grp['cycle_id'] == c, 'ret_b1'  ].dropna().values for c in cycles]
    data_b2   = [grp.loc[grp['cycle_id'] == c, 'ret_b2'  ].dropna().values for c in cycles]
    data_half = [grp.loc[grp['cycle_id'] == c, 'ret_half'].dropna().values for c in cycles]

    fig, ax = plt.subplots(figsize=(20, 4.6))

    # triplets per cycle (LEFT->RIGHT = stringent, permissive, HALF)
    offset = 0.19
    width = 0.15
    pos_b1 = [c - offset for c in cycles]
    pos_b2 = [c for c in cycles]
    pos_h  = [c + offset for c in cycles]

    # draw boxplots (skip empty)
    _grouped_boxplot(ax,
                     data_by_group={'Stringent': data_b1},
                     base_positions=cycles,
                     offsets={'Stringent': -offset},
                     colors={'Stringent': stringent_color},
                     widths=width, alpha=0.45, median_lw=2.5, showfliers=False)

    _grouped_boxplot(ax,
                     data_by_group={'Permissive': data_b2},
                     base_positions=cycles,
                     offsets={'Permissive': 0.0},
                     colors={'Permissive': permissive_color},
                     widths=width, alpha=0.45, median_lw=2.5, showfliers=False)

    _grouped_boxplot(ax,
                     data_by_group={'HALF': data_half},
                     base_positions=cycles,
                     offsets={'HALF': +offset},
                     colors={'HALF': HALF_color},
                     widths=width, alpha=0.45, median_lw=2.5, showfliers=False)

    ax.set_ylabel('Retention rate (%)')
    ax.set_ylim(-4, 104)
    ax.grid(False)

    # label EVERY cycle and date
    ax.set_xlim(min(cycles) - 0.5, max(cycles) + 0.5)
    ax.set_xticks(cycles)
    ax.set_yticks(list([0,25,50,75,100]))
    date_labels = [(date_map.get(c).strftime("%Y-%m-%d") if (i % 2 == 0 and pd.notna(date_map.get(c, pd.NaT))) else "")
                   for i, c in enumerate(cycles)] #label every 2 cycles
    ax.set_xticklabels(date_labels, rotation=30, ha='right')
    ax.set_xlabel('Cycle start date (UTC)')

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(cycles)
    ax_top.set_xticklabels([str(c) for c in cycles])
    ax_top.set_xlabel('SWOT cycle ID', labelpad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'Lake_retention_box3_by_cycle_{script_version}_{version_filename}.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f'Lake_retention_box3_by_cycle_{script_version}_{version_filename}.pdf'),
                bbox_inches='tight')


    # =============================================================================
    # FIGURES: Boxplots
    #   1) MAE (ice-free vs ice-covered), grouped Raw/Stringent/Permissive/HALF
    #   2) rel_err (ice-free vs ice-covered), grouped Raw/Stringent/Permissive/HALF
    #   3) rel_err_var vs lake size bins, grouped Raw/Stringent/Permissive/HALF
    # =============================================================================
    method_colors = {'Raw': raw_color, 'Stringent': stringent_color, 'Permissive': permissive_color, 'HALF': HALF_color}

    # =============================================================================
    # Helper for MAE / median residual in ice-condition summaries
    # Consistent with the main-block pointwise error calculation:
    #   1. keep finite LakeSP-gauge pairs
    #   2. remove effectively zero residuals after bias correction
    #   3. compute MAE / median residual only from remaining pairs
    # =============================================================================
    # ---- Build per-lake MAE + relative error for each method, split by ice-free vs ice-covered
    records = []
    for lake_id in lakes_for_validation:
        sub = df_lake_time_series[df_lake_time_series['lake_id'] == lake_id].copy()
        if sub.empty:
            continue

        area_vals = sub['p_ref_area'].dropna().astype(float) if 'p_ref_area' in sub.columns else pd.Series(dtype=float)
        lake_area = float(area_vals.iloc[-1]) if len(area_vals) else np.nan

        std_scale = sub['wse_adjusted'].std()
        if not np.isfinite(std_scale) or std_scale <= 0:
            std_scale = np.nan

        ice = sub['ice_clim_f'].to_numpy()
        m_icefree = (ice < 1)
        m_ice = (ice >= 1)

        method_specs = {
            'Raw':        ('wse',           sub['wse'].notna()),
            'Stringent':  ('wse_benchmark_stringent', sub['wse_benchmark_stringent'].notna()),
            'Permissive': ('wse_benchmark_permissive', sub['wse_benchmark_permissive'].notna()),
            'HALF':       ('wse_adjusted',  sub['wse_adjusted'].notna())
        }

        for method, (wcol, base_mask) in method_specs.items():
            base = base_mask & sub['gauge_wse_bias_corrected'].notna()

            # Define 3 ice-condition masks (same length as sub)
            cond_specs = [
                ("Ice-free",   m_icefree),
                ("Ice-covered", m_ice),
                ("Ice-both",   np.ones_like(m_icefree, dtype=bool))  # NEW: ignore ice flag
            ]

            for cond_name, cond_mask in cond_specs:
                mm = base & cond_mask
                if mm.any():
                    swot_vals = sub.loc[mm, wcol].to_numpy()
                    gauge_vals = sub.loc[mm, 'gauge_wse_bias_corrected'].to_numpy()

                    mae, medianE, n_match_cond, n_match_cond_used = (
                        _compute_mae_medianE_drop_zero_residuals(
                            swot_vals,
                            gauge_vals,
                            min_n=min_matched_obs_for_error_metrics,
                            zero_tol=zero_resid_tol
                        )
                    )
                else:
                    mae = np.nan
                    medianE = np.nan
                    n_match_cond = 0
                    n_match_cond_used = 0

                rel = (mae / std_scale) if (np.isfinite(mae) and np.isfinite(std_scale) and std_scale > 0) else np.nan

                records.append({
                    'lake_id': lake_id,
                    'lake_area': lake_area,
                    'method': method,
                    'ice_cond': cond_name,
                    'mae': mae,
                    'medianE': medianE,
                    'rel_err': rel,
                    'n_match_cond': n_match_cond,
                    'n_match_cond_used': n_match_cond_used,
                    'zero_resid_tol': zero_resid_tol
                })

    df_err = pd.DataFrame(records)

    # -----------------------------
    # PRINT MAE + REL_ERR by ice condition (Ice-both / Ice-free / Ice-covered) and method
    # NOTE: df_err is per-lake (one row per lake×method×ice_cond), not per-observation.
    # -----------------------------
    def _p50_summary(x):
        x = np.asarray(x, dtype=float)
        return np.nanpercentile(x, 50), np.nanpercentile(x, SUMMARY_PERCENTILE)

    methods = ["Raw", "Stringent", "Permissive", "HALF"]
    conds   = ["Ice-both", "Ice-free", "Ice-covered"]

    print(f"\n=== MAE (m): P50, {SUMMARY_PERCENTILE_LABEL} by ice condition and method ===")
    for cond in conds:
        print(f"\n{cond}:")
        for m in methods:
            vals = df_err.loc[(df_err["ice_cond"] == cond) & (df_err["method"] == m), "mae"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                p50, p_summary = _p50_summary(vals)
                print(f"  {m:<10s}  n={vals.size:4d}  MAE P50={p50:.4f} m, {SUMMARY_PERCENTILE_LABEL}={p_summary:.4f} m")
            else:
                print(f"  {m:<10s}  n=   0  MAE = NaN")

    print(f"\n=== MAE normalized by WSE variability (rel_err): P50, {SUMMARY_PERCENTILE_LABEL} by ice condition and method ===")
    for cond in conds:
        print(f"\n{cond}:")
        for m in methods:
            vals = df_err.loc[(df_err["ice_cond"] == cond) & (df_err["method"] == m), "rel_err"].values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                p50, p_summary = _p50_summary(vals)
                print(f"  {m:<10s}  n={vals.size:4d}  rel_err P50={p50:.3f}, {SUMMARY_PERCENTILE_LABEL}={p_summary:.3f}")
            else:
                print(f"  {m:<10s}  n=   0  rel_err = NaN")

    # ---- 1) MAE boxplot (ice-free vs ice-covered), grouped 4 methods
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 17,
        "legend.fontsize": 12,
        "figure.titlesize": 24
    })
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.grid(False)

    cats = ['Ice-free', 'Ice-covered']
    base_pos = [1, 2]
    offsets = {'Raw': -0.30, 'Stringent': -0.10, 'Permissive': 0.10, 'HALF': 0.30}

    data_by_group = {
        m: [df_err.loc[(df_err['method'] == m) & (df_err['ice_cond'] == c), 'mae'].dropna().to_numpy()
            for c in cats]
        for m in ['Raw', 'Stringent', 'Permissive', 'HALF']
    }
    _grouped_boxplot(ax, data_by_group, base_pos, offsets, method_colors, widths=0.16, alpha=0.45, showfliers=False)

    ax.set_xticks(base_pos)
    ax.set_xticklabels(cats)
    ax.set_ylabel('MAE (m)')
    ax.set_xlim(0.5, 2.5)

    if version_filename == 'vD':
        ax.set_yticks([0,2,4,6,8])
        ax.set_ylim(-0.2,8)
        #ax.set_yticks([0,1,2,3,4,5,6])
        #ax.set_ylim(-0.2,6)
    else: #version C
        ax.set_yticks([0,2,4,6,8])
        ax.set_ylim(-0.2,8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'MAE_box_icefree_vs_ice_{script_version}_{version_filename}.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f'MAE_box_icefree_vs_ice_{script_version}_{version_filename}.pdf'),
                bbox_inches='tight')
    #plt.close(fig)


    # ---- 2) Relative error BOX (ice-free vs ice-covered), grouped 4 methods
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.grid(False)

    data_by_group = {
        m: [df_err.loc[(df_err['method'] == m) & (df_err['ice_cond'] == c), 'rel_err'].dropna().to_numpy()
            for c in cats]
        for m in ['Raw', 'Stringent', 'Permissive', 'HALF']
    }
    _grouped_boxplot(ax, data_by_group, base_pos, offsets, method_colors, widths=0.16, alpha=0.45, showfliers=False)

    ax.set_xticks(base_pos)
    ax.set_xticklabels(cats)
    ax.set_ylabel('MAE normalized\nby WSE variability ')
    ax.set_xlim(0.5, 2.5)

    if version_filename == 'vD':
        ax.set_yticks([0,2,4,6,8])
        ax.set_ylim(-0.2,8)
    else: #version C
        ax.set_yticks([0,2,4,6,8])
        ax.set_ylim(-0.2,8)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'RelErr_box_icefree_vs_ice_{script_version}_{version_filename}.png'),
                bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, f'RelErr_box_icefree_vs_ice_{script_version}_{version_filename}.pdf'),
                bbox_inches='tight')
    #plt.close(fig)


    # ---- 3) rel_err_var vs lake size bins (GROUPED BOX, 4 methods)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 12,
        "figure.titlesize": 24
    })
    dfv = df_lake_stats[df_lake_stats['lake_id'].isin(lakes_for_validation)].copy()
    needed = ['lake_area', 'var_std_gauge_daily',
              'var_std_swot_daily_raw', 'var_std_swot_daily_benchmark_stringent', 'var_std_swot_daily_benchmark_permissive', 'var_std_swot_daily']
    if all(c in dfv.columns for c in needed):
        dfv = dfv.dropna(subset=['lake_area', 'var_std_gauge_daily']).copy()
        dfv = dfv[dfv['var_std_gauge_daily'] > 0].copy()

        dfv['rel_err_var_raw'] = np.abs(dfv['var_std_swot_daily_raw'] - dfv['var_std_gauge_daily']) / dfv['var_std_gauge_daily']
        dfv['rel_err_var_stringent'] = np.abs(dfv['var_std_swot_daily_benchmark_stringent'] - dfv['var_std_gauge_daily']) / dfv['var_std_gauge_daily']
        dfv['rel_err_var_permissive'] = np.abs(dfv['var_std_swot_daily_benchmark_permissive'] - dfv['var_std_gauge_daily']) / dfv['var_std_gauge_daily']
        dfv['rel_err_var_half'] = np.abs(dfv['var_std_swot_daily'] - dfv['var_std_gauge_daily']) / dfv['var_std_gauge_daily']

        areas = dfv['lake_area'].astype(float)
        areas = areas[np.isfinite(areas) & (areas > 0)]
        if len(areas) >= 5:
            a_min, a_max = float(areas.min()), float(areas.max())
            n_bins = 7
            #edges = np.logspace(np.log10(a_min), np.log10(a_max), n_bins + 1)
            #labels = [f'{edges[i]:.2g}–{edges[i+1]:.2g}' for i in range(n_bins)]
            edges = [a_min, 0.1, 1, 10, 100, 1000, 10000, a_max]
            labels = ['0.016–0.1','≤ 1','≤ 10','≤ 100','≤ 1,000','≤ 10,000','≤ 57,791']
            dfv['area_bin'] = pd.cut(dfv['lake_area'], bins=edges, labels=labels, include_lowest=True)

            #======================================================================
            # ---------------------------------------------------------------------
            # Report how much HALF reduces median normalized variability error
            # relative to the stringent filter, by lake-area bin
            # ---------------------------------------------------------------------
            area_bin_reduction_records = []
            for lab in labels:
                sub_bin = dfv[dfv["area_bin"] == lab].copy()

                stringent_vals = sub_bin["rel_err_var_stringent"].dropna().to_numpy()
                half_vals = sub_bin["rel_err_var_half"].dropna().to_numpy()

                if len(stringent_vals) > 0 and len(half_vals) > 0:
                    median_stringent = np.nanmedian(stringent_vals)
                    median_half = np.nanmedian(half_vals)

                    if np.isfinite(median_stringent) and median_stringent > 0:
                        half_reduction_pct = (
                            (median_stringent - median_half) / median_stringent * 100
                        )
                    else:
                        half_reduction_pct = np.nan

                    area_bin_reduction_records.append({
                        "area_bin": lab,
                        "n_lakes": len(sub_bin),
                        "median_stringent": median_stringent,
                        "median_half": median_half,
                        "half_reduction_pct": half_reduction_pct
                    })
            df_area_bin_reduction = pd.DataFrame(area_bin_reduction_records)
            print("\n=== Reduction in median normalized variability error: HALF vs stringent ===")
            print(df_area_bin_reduction.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

            # Summary across bins
            valid_reductions = df_area_bin_reduction["half_reduction_pct"].dropna()

            if len(valid_reductions) > 0:
                min_reduction = valid_reductions.min()
                max_reduction = valid_reductions.max()
                median_reduction = valid_reductions.median()

                print(
                    f"\nAcross lake-area bins, HALF reduces median normalized variability error "
                    f"relative to the stringent filter by {min_reduction:.1f}–{max_reduction:.1f}% "
                    f"(median across bins: {median_reduction:.1f}%)."
                )
            #=============================================================

            bins_list = list(range(1, n_bins + 1))
            data_by_group = {
                'Raw':       [dfv.loc[dfv['area_bin'] == labels[i], 'rel_err_var_raw'].dropna().to_numpy() for i in range(n_bins)],
                'Stringent': [dfv.loc[dfv['area_bin'] == labels[i], 'rel_err_var_stringent'].dropna().to_numpy() for i in range(n_bins)],
                'Permissive':[dfv.loc[dfv['area_bin'] == labels[i], 'rel_err_var_permissive'].dropna().to_numpy() for i in range(n_bins)],
                'HALF':      [dfv.loc[dfv['area_bin'] == labels[i], 'rel_err_var_half'].dropna().to_numpy() for i in range(n_bins)]
            }

            fig, ax = plt.subplots(figsize=(12, 3.5))
            ax.grid(False)

            offsets2 = {'Raw': -0.30, 'Stringent': -0.10, 'Permissive': 0.10, 'HALF': 0.30}
            _grouped_boxplot(ax, data_by_group, bins_list, offsets2, method_colors, widths=0.16, alpha=0.45, showfliers=False)

            ax.set_xticks(bins_list)
            ax.set_xticklabels(labels)
            #ax.set_xticklabels(labels, rotation=20, ha='right')
            ax.set_xlabel('Lake area bin (km²)')
            ax.set_ylabel("WSE variability\nerror (normalized)")

            if version_filename == 'vD':
                ax.set_yticks([0,2,4,6,8,10])
                ax.set_ylim(-0.2, 11.5)
                #ax.set_yticks([0,1,2,3,4,5,6])
                #ax.set_ylim(-0.2, 6)
            else: #version C
                ax.set_yticks([0,2,4,6,8,10])
                ax.set_ylim(-0.2, 11.5)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'RelErrVar_box_by_area_{script_version}_{version_filename}.png'),
                        bbox_inches='tight')
            plt.savefig(os.path.join(plots_dir, f'RelErrVar_box_by_area_{script_version}_{version_filename}.pdf'),
                        bbox_inches='tight')
            #plt.close(fig)




    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Supplementary plots
    # SI Fig. A
    # =============================================================================
    # Figure: Global dot maps of lake MAE for HALF
    #   upper panel: ice-free
    #   lower panel: ice-covered
    # =============================================================================

    def plot_global_mae_maps(df_err, hc_ll, plots_dir, script_version, version_filename):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.colors import Normalize

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        # Keep HALF only and only the two requested ice conditions
        df_map = df_err[
            (df_err["method"] == "HALF") &
            (df_err["ice_cond"].isin(["Ice-free", "Ice-covered"]))
        ].copy()

        # Merge lat/lon
        df_map = df_map.merge(hc_ll, on="lake_id", how="left")
        df_map = df_map.dropna(subset=["p_lat", "p_lon", "mae"]).copy()
        #df_map = df_map[df_map["mae"] > 0].copy() # no need for this.

        if df_map.empty:
            print("No valid data available for global HALF MAE maps.")
            return

        # Shared color scale across both panels
        vmin = 0
        vmax = 2

        # Safety
        if not np.isfinite(vmin) or vmin < 0:
            vmin = max(np.nanmin(df_map["mae"]), 0.0)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = np.nanmax(df_map["mae"])

        ocean_col = "#b9dce8"
        land_col = "whitesmoke"
        border_col = "#b0b0b0"
        coast_col = "#8a8a8a"

        # Same layout style as plot_official_timing_error_maps()
        fig_width = 7.2
        fig_height = 7.4
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)

        map_left = 0.02
        map_w = 0.96

        # Preserve world-map aspect, slightly shrunk to create spacing
        map_h = (map_w * fig_width / 2.0) / fig_height * 0.94

        # Same panel locations
        map_high_bottom = 0.590
        map_low_bottom = 0.100

        ax_top = fig.add_axes(
            [map_left, map_high_bottom, map_w, map_h],
            projection=ccrs.PlateCarree()
        )

        ax_bottom = fig.add_axes(
            [map_left, map_low_bottom, map_w, map_h],
            projection=ccrs.PlateCarree()
        )

        axes = [ax_top, ax_bottom]
        conds = ["Ice-free", "Ice-covered"]

        sc = None

        for ax, cond in zip(axes, conds):
            sub = df_map[df_map["ice_cond"] == cond].copy()

            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            ax.set_facecolor(ocean_col)
            ax.add_feature(cfeature.OCEAN, facecolor=ocean_col, edgecolor="none", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor=land_col, edgecolor="none", zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor=border_col, zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.45, edgecolor=coast_col, zorder=2)

            lakes = cfeature.NaturalEarthFeature(
                "physical", "lakes", "50m",
                edgecolor="#9bbfd0",
                facecolor="#9fcfe3"
            )
            ax.add_feature(lakes, zorder=2.3, linewidth=0.25, alpha=0.9)

            sc = ax.scatter(
                sub["p_lon"], sub["p_lat"],
                c=sub["mae"],
                s=6.5,
                cmap="plasma",
                norm=Normalize(vmin=vmin, vmax=vmax),
                transform=ccrs.PlateCarree(),
                edgecolors="#4d4d4d",
                linewidths=0.12,
                alpha=0.85,
                zorder=5
            )

            ax.set_title(f"HALF MAE ({cond}; N = {len(sub)})", fontsize=12, pad=4)
            ax.axis("off")

        # Colorbar on the FIRST / upper panel
        cax = ax_top.inset_axes([0.15, 0.120, 0.70, 0.055])
        cbar = fig.colorbar(
            sc,
            cax=cax,
            orientation="horizontal",
            extend="max"
        )
        cbar.set_label("MAE (m)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=9)

        out_path = os.path.join(
            plots_dir,
            f"Global_HALF_MAE_maps_{script_version}_{version_filename}.pdf"
        )

        plt.savefig(out_path, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    plot_global_mae_maps(df_err, hc_ll, plots_dir, script_version, version_filename)


    # =============================================================================
    # Shared helper for CDF plots
    # Labels the HALF curve at selected percentile levels
    # =============================================================================
    def _annotate_half_cdf_percentiles(
        ax,
        half_vals,
        color,
        percentiles=(50, SUMMARY_PERCENTILE),
        value_fmt="{:.2f}",
        fontsize=10,
        x_text_offset=5,
        y_text_offset=4,
        align_right=False,
        x_right_pad_frac=0.02
    ):
        """
        Annotate the HALF CDF curve at selected percentiles.

        By default, labels are placed near their percentile x-value, as before.
        If align_right=True, labels are placed at a fixed right-side location
        inside the plot box, with their right edges aligned, while preserving
        the original vertical offset above the reference line.
        """
        import numpy as np

        half_vals = np.asarray(half_vals, dtype=float)
        half_vals = half_vals[np.isfinite(half_vals)]

        if len(half_vals) == 0:
            return

        # Fixed right-side x anchor, used only when align_right=True.
        if align_right:
            xmin, xmax = ax.get_xlim()
            x_anchor_right = xmax - (xmax - xmin) * x_right_pad_frac

        for p in percentiles:
            x_val = np.nanpercentile(half_vals, p)
            y_val = p  # percentage units, consistent with CDF (%) y-axis

            if np.isclose(p, 50, atol=1e-8):
                p_label = "50%"
            elif np.isclose(p, SUMMARY_EXCEEDANCE_CDF, atol=1e-8):
                p_label = SUMMARY_EXCEEDANCE_LABEL
            elif abs(p - round(p)) < 1e-8:
                p_label = f"{int(round(p))}%"
            else:
                p_label = f"{p:.2f}%"

            if align_right:
                # Put all labels at the same right-side x position.
                # Use the same vertical point offset as before.
                xy = (x_anchor_right, y_val)
                xytext = (0, y_text_offset)
                ha = "right"
            else:
                # Original behavior: label is placed near the percentile x-value.
                xy = (x_val, y_val)
                xytext = (x_text_offset, y_text_offset)
                ha = "left"

            ax.annotate(
                f"{value_fmt.format(x_val)} ({p_label})",
                xy=xy,
                xytext=xytext,
                textcoords="offset points",
                color=color,
                fontsize=fontsize,
                ha=ha,
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85
                ),
                clip_on=True,
                zorder=10
            )

    # =============================================================================
    # Figure: CDFs of per-lake MAE
    #   left panel: ice-free
    #   right panel: ice-covered
    # =============================================================================

    def plot_mae_cdfs(
        df_err,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        def _cdf(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            arr = np.sort(arr)
            if len(arr) == 0:
                return np.array([]), np.array([])
            return arr, np.arange(1, len(arr) + 1) / len(arr) * 100.0

        method_colors = {
            "Raw": raw_color,
            "Stringent": stringent_color,
            "Permissive": permissive_color,
            "HALF": HALF_color
        }

        # Same figure dimensions as the timing-error CDF figure
        fig_width = 7.2
        fig_height = 3.0

        fig, axes = plt.subplots(
            1, 2,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            sharey=True
        )

        conds = ["Ice-free", "Ice-covered"]

        for ax, cond in zip(axes, conds):
            sub = df_err[df_err["ice_cond"] == cond].copy()

            half_vals_for_label = None

            for method in ["Raw", "Stringent", "Permissive", "HALF"]:
                vals = sub.loc[sub["method"] == method, "mae"].values
                x, y = _cdf(vals)

                if len(x) > 0:
                    ax.plot(
                        x,
                        y,
                        linewidth=2.2,
                        color=method_colors[method],
                        label=method
                    )

                if method == "HALF":
                    half_vals_for_label = vals

            # Reference lines in percentage units
            ax.axhline(50, color="black", linestyle="--", linewidth=1)
            ax.axhline(SUMMARY_PERCENTILE, color="black", linestyle="--", linewidth=1)

            ax.set_title(cond, fontsize=12)
            ax.set_xlabel("MAE (m)", fontsize=11)
            ax.set_ylabel("CDF (%)", fontsize=11)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.tick_params(labelsize=10)

            # Label HALF statistics at the 50% and summary-percentile CDF levels
            _annotate_half_cdf_percentiles(
                ax,
                half_vals_for_label,
                color=HALF_color,
                percentiles=(50, SUMMARY_PERCENTILE),
                value_fmt="{:.2f}",
                fontsize=10
            )

        # Legend on the LEFT panel
        axes[0].legend(
            frameon=False,
            loc="lower right",
            fontsize=10
        )

        # Same spacing style as the timing-error CDF function
        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.20,
            top=0.86,
            wspace=0.18
        )

        out_name = f"Lake_MAE_CDFs_{script_version}_{version_filename}.{save_format}"
        out_path = os.path.join(plots_dir, out_name)

        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    plot_mae_cdfs(
        df_err,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    )

    # Print P50 and summary percentile for MAE by ice condition and method
    print(f"\n=== MAE percentiles (P50 and {SUMMARY_PERCENTILE_LABEL}) by ice condition and method ===")
    for cond in ["Ice-free", "Ice-covered"]:
        print(f"\n{cond}:")
        for method in ["Raw", "Stringent", "Permissive", "HALF"]:
            vals = df_err.loc[
                (df_err["ice_cond"] == cond) & (df_err["method"] == method),
                "mae"
            ].dropna().values

            if len(vals) > 0:
                p50 = np.nanpercentile(vals, 50)
                p_summary = np.nanpercentile(vals, SUMMARY_PERCENTILE)
                print(f"  {method:<10s} n={len(vals):4d}  P50={p50:.4f} m   {SUMMARY_PERCENTILE_LABEL}={p_summary:.4f} m")
            else:
                print(f"  {method:<10s} n=   0  P50=nan   {SUMMARY_PERCENTILE_LABEL}=nan")



    # SI Fig. B
    # =============================================================================
    # Figure: Global dot maps of lake median residual (bias) for HALF
    #   upper panel: ice-free
    #   lower panel: ice-covered
    # =============================================================================

    def plot_global_medianE_maps(df_err, hc_ll, plots_dir, script_version, version_filename):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.colors import TwoSlopeNorm

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        # Keep HALF only and only the two requested ice conditions
        df_map = df_err[
            (df_err["method"] == "HALF") &
            (df_err["ice_cond"].isin(["Ice-free", "Ice-covered"]))
        ].copy()

        # Merge lat/lon
        df_map = df_map.merge(hc_ll, on="lake_id", how="left")
        df_map = df_map.dropna(subset=["p_lat", "p_lon", "medianE"]).copy()

        if df_map.empty:
            print("No valid data available for global HALF median-residual maps.")
            return

        # Use a symmetric diverging scale because median residual is signed.
        # Positive and negative values indicate opposite WSE bias directions.
        vmax = np.nanpercentile(np.abs(df_map["medianE"]), 95)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(df_map["medianE"]))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        ocean_col = "#b9dce8"
        land_col = "whitesmoke"
        border_col = "#b0b0b0"
        coast_col = "#8a8a8a"

        # Same layout style as the final reference map figure
        fig_width = 7.2
        fig_height = 7.4
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)

        map_left = 0.02
        map_w = 0.96

        # Preserve world-map aspect, slightly shrunk to create spacing
        map_h = (map_w * fig_width / 2.0) / fig_height * 0.94

        map_high_bottom = 0.590
        map_low_bottom = 0.100

        ax_top = fig.add_axes(
            [map_left, map_high_bottom, map_w, map_h],
            projection=ccrs.PlateCarree()
        )

        ax_bottom = fig.add_axes(
            [map_left, map_low_bottom, map_w, map_h],
            projection=ccrs.PlateCarree()
        )

        axes = [ax_top, ax_bottom]
        conds = ["Ice-free", "Ice-covered"]

        sc = None

        for ax, cond in zip(axes, conds):
            sub = df_map[df_map["ice_cond"] == cond].copy()

            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
            ax.set_facecolor(ocean_col)
            ax.add_feature(cfeature.OCEAN, facecolor=ocean_col, edgecolor="none", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor=land_col, edgecolor="none", zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor=border_col, zorder=2)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.45, edgecolor=coast_col, zorder=2)

            lakes = cfeature.NaturalEarthFeature(
                "physical", "lakes", "50m",
                edgecolor="#9bbfd0",
                facecolor="#9fcfe3"
            )
            ax.add_feature(lakes, zorder=2.3, linewidth=0.25, alpha=0.9)

            sc = ax.scatter(
                sub["p_lon"], sub["p_lat"],
                c=sub["medianE"],
                s=6.5,
                cmap="coolwarm",
                norm=norm,
                transform=ccrs.PlateCarree(),
                edgecolors="#4d4d4d",
                linewidths=0.12,
                alpha=0.85,
                zorder=5
            )

            ax.set_title(f"HALF median residual ({cond}; N = {len(sub)})", fontsize=12, pad=4)
            ax.axis("off")

        # Colorbar on the FIRST / upper panel
        cax = ax_top.inset_axes([0.15, 0.120, 0.70, 0.055])
        cbar = fig.colorbar(
            sc,
            cax=cax,
            orientation="horizontal",
            extend="both"
        )
        cbar.set_label("Median residual (m)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=9)

        out_path = os.path.join(
            plots_dir,
            f"Global_HALF_medianE_maps_{script_version}_{version_filename}.pdf"
        )

        plt.savefig(out_path, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    plot_global_medianE_maps(
        df_err, hc_ll, plots_dir, script_version, version_filename
    )


    # =============================================================================
    # Figure: CDFs of per-lake median residuals
    #   left panel: ice-free
    #   right panel: ice-covered
    # =============================================================================

    def plot_medianE_cdfs(
        df_err,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        def _cdf(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            arr = np.sort(arr)
            if len(arr) == 0:
                return np.array([]), np.array([])
            return arr, np.arange(1, len(arr) + 1) / len(arr) * 100.0

        method_colors = {
            "Raw": raw_color,
            "Stringent": stringent_color,
            "Permissive": permissive_color,
            "HALF": HALF_color
        }

        # Same figure dimensions as the timing-error and MAE CDF figures
        fig_width = 7.2
        fig_height = 3.0

        fig, axes = plt.subplots(
            1, 2,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            sharey=True
        )

        conds = ["Ice-free", "Ice-covered"]

        for ax, cond in zip(axes, conds):
            sub = df_err[df_err["ice_cond"] == cond].copy()

            half_vals_for_label = None

            for method in ["Raw", "Stringent", "Permissive", "HALF"]:
                vals = sub.loc[sub["method"] == method, "medianE"].values
                x, y = _cdf(vals)

                if len(x) > 0:
                    ax.plot(
                        x,
                        y,
                        linewidth=2.2,
                        color=method_colors[method],
                        label=method
                    )

                if method == "HALF":
                    half_vals_for_label = vals

            # Reference lines in percentage units
            ax.axhline(50, color="black", linestyle="--", linewidth=1)
            ax.axhline(SUMMARY_PERCENTILE, color="black", linestyle="--", linewidth=1)
            ax.axvline(0, color="black", linestyle=":", linewidth=1)

            ax.set_title(cond, fontsize=12)
            ax.set_xlabel("Median residual (LakeSP - Gauge, m)", fontsize=11)
            ax.set_ylabel("CDF (%)", fontsize=11)
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.tick_params(labelsize=10)

            # Label HALF statistics at the 50% and summary-percentile CDF levels
            _annotate_half_cdf_percentiles(
                ax,
                half_vals_for_label,
                color=HALF_color,
                percentiles=(50, SUMMARY_PERCENTILE),
                value_fmt="{:.2f}",
                fontsize=10
            )

        # Legend on the LEFT panel
        axes[0].legend(
            frameon=False,
            loc="lower right",
            fontsize=10
        )

        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.20,
            top=0.86,
            wspace=0.18
        )

        out_name = f"Lake_medianE_CDFs_{script_version}_{version_filename}.{save_format}"
        out_path = os.path.join(plots_dir, out_name)

        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    plot_medianE_cdfs(
        df_err,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    )


    # =============================================================================
    # Print P50 and summary percentile for median residual by ice condition and method
    # =============================================================================
    print(f"\n=== Median residual percentiles (P50 and {SUMMARY_PERCENTILE_LABEL}) by ice condition and method ===")
    for cond in ["Ice-free", "Ice-covered"]:
        print(f"\n{cond}:")
        for method in ["Raw", "Stringent", "Permissive", "HALF"]:
            vals = df_err.loc[
                (df_err["ice_cond"] == cond) &
                (df_err["method"] == method),
                "medianE"
            ].dropna().values

            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) > 0:
                p50 = np.nanpercentile(vals, 50)
                p_summary = np.nanpercentile(vals, SUMMARY_PERCENTILE)
                print(
                    f"  {method:<10s} n={len(vals):4d}  "
                    f"P50={p50:.4f} m   {SUMMARY_PERCENTILE_LABEL}={p_summary:.4f} m"
                )
            else:
                print(f"  {method:<10s} n=   0  P50=nan   {SUMMARY_PERCENTILE_LABEL}=nan")





    # SI Fig. C
    # =============================================================================
    # Figure: Global dot map of lake correlation for HALF
    #   no ice stratification
    # =============================================================================
    def plot_global_correlation_map(
        df_lake_stats,
        hc_ll,
        lakes_for_validation,
        plots_dir,
        script_version,
        version_filename
    ):
        import os
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.colors import Normalize

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        # Keep only validated lakes and HALF correlation
        df_map = df_lake_stats[df_lake_stats["lake_id"].isin(lakes_for_validation)].copy()
        df_map = df_map.loc[:, ["lake_id", "correlation"]].copy()

        # Merge lat/lon
        df_map = df_map.merge(hc_ll, on="lake_id", how="left")
        df_map = df_map.dropna(subset=["p_lat", "p_lon", "correlation"]).copy()

        if df_map.empty:
            print("No valid data available for global HALF correlation map.")
            return

        ocean_col = "#b9dce8"
        land_col = "whitesmoke"
        border_col = "#b0b0b0"
        coast_col = "#8a8a8a"

        # Same full-width style as the reference map figure, but single-panel
        fig_width = 7.2
        fig_height = 3.9
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)

        map_left = 0.02
        map_w = 0.96

        # Preserve world-map aspect, slightly shrunk to create room for title/colorbar
        map_h = (map_w * fig_width / 2.0) / fig_height * 0.94
        map_bottom = 0.185

        ax = fig.add_axes(
            [map_left, map_bottom, map_w, map_h],
            projection=ccrs.PlateCarree()
        )

        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.set_facecolor(ocean_col)
        ax.add_feature(cfeature.OCEAN, facecolor=ocean_col, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=land_col, edgecolor="none", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor=border_col, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.45, edgecolor=coast_col, zorder=2)

        lakes = cfeature.NaturalEarthFeature(
            "physical", "lakes", "50m",
            edgecolor="#9bbfd0",
            facecolor="#9fcfe3"
        )
        ax.add_feature(lakes, zorder=2.3, linewidth=0.25, alpha=0.9)

        sc = ax.scatter(
            df_map["p_lon"], df_map["p_lat"],
            c=df_map["correlation"],
            s=6.5,
            cmap="coolwarm",
            norm=Normalize(vmin=-1, vmax=1),
            transform=ccrs.PlateCarree(),
            edgecolors="#4d4d4d",
            linewidths=0.12,
            alpha=0.85,
            zorder=5
        )

        ax.set_title(f"HALF correlation (Pearson; N = {len(df_map)})", fontsize=12, pad=4)
        ax.axis("off")

        # Colorbar inside map over Antarctica
        cax = ax.inset_axes([0.15, 0.120, 0.70, 0.055])
        cbar = fig.colorbar(
            sc,
            cax=cax,
            orientation="horizontal",
            extend="neither"
        )
        cbar.set_label("Correlation (Pearson)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=9)

        out_path = os.path.join(
            plots_dir,
            f"Global_HALF_correlation_map_{script_version}_{version_filename}.pdf"
        )

        plt.savefig(out_path, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    plot_global_correlation_map(
        df_lake_stats,
        hc_ll,
        lakes_for_validation,
        plots_dir,
        script_version,
        version_filename
    )


    # =============================================================================
    # Figure: CDF of per-lake correlations
    #   Raw / Stringent / Permissive / HALF
    # =============================================================================

    def plot_correlation_cdf(
        df_lake_stats,
        lakes_for_validation,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        def _cdf(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            arr = np.sort(arr)
            if len(arr) == 0:
                return np.array([]), np.array([])
            return arr, np.arange(1, len(arr) + 1) / len(arr) * 100.0

        subset_stats = df_lake_stats[df_lake_stats["lake_id"].isin(lakes_for_validation)].copy()

        method_specs = [
            ("Raw",        "correlation_raw",        raw_color),
            ("Stringent",  "correlation_benchmark_stringent",  stringent_color),
            ("Permissive", "correlation_benchmark_permissive",  permissive_color),
            ("HALF",       "correlation",            HALF_color),
        ]

        # Same CDF width/style as the reference CDF figures
        fig_width = 7.2
        fig_height = 3.0
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        half_vals_for_label = None

        for label, col, color in method_specs:
            vals = subset_stats[col].values
            x, y = _cdf(vals)

            if len(x) > 0:
                ax.plot(
                    x,
                    y,
                    linewidth=2.2,
                    color=color,
                    label=label
                )

            if label == "HALF":
                half_vals_for_label = vals

        # Reference lines in percentage units
        ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.axhline(SUMMARY_EXCEEDANCE_CDF, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Correlation (Pearson)", fontsize=11)
        ax.set_ylabel("CDF (%)", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=10)

        # Label HALF median and summary exceedance statistics
        _annotate_half_cdf_percentiles(
            ax,
            half_vals_for_label,
            color=HALF_color,
            percentiles=(50, SUMMARY_EXCEEDANCE_CDF),
            value_fmt="{:.2f}",
            fontsize=10,
            align_right=True,
            x_right_pad_frac=0.02
        )

        ax.legend(
            frameon=False,
            loc="upper left",
            fontsize=10
        )

        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.20,
            top=0.86
        )

        out_name = f"Lake_correlation_CDF_{script_version}_{version_filename}.{save_format}"
        out_path = os.path.join(plots_dir, out_name)

        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    plot_correlation_cdf(
        df_lake_stats,
        lakes_for_validation,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    )


    # =============================================================================
    # Print P50 and summary exceedance percentile for correlations by method
    # =============================================================================
    print(f"\n=== Correlation percentiles (P50 and {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})) by method ===")
    for label, col in [
        ("Raw", "correlation_raw"),
        ("Stringent", "correlation_benchmark_stringent"),
        ("Permissive", "correlation_benchmark_permissive"),
        ("HALF", "correlation"),
    ]:
        vals = df_lake_stats.loc[
            df_lake_stats["lake_id"].isin(lakes_for_validation),
            col
        ].dropna().values

        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) > 0:
            p50 = np.nanpercentile(vals, 50)
            p_exceedance = np.nanpercentile(vals, SUMMARY_EXCEEDANCE_CDF)
            print(
                f"  {label:<10s} n={len(vals):4d}  "
                f"P50={p50:.4f}   "
                f"{SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})={p_exceedance:.4f}"
            )
        else:
            print(f"  {label:<10s} n=   0  P50=nan   {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})=nan")




    # SI Fig. D
    # =============================================================================
    # Figure: Global dot map of lake retention rate (%) for HALF
    #   no ice stratification
    # =============================================================================

    def plot_global_retention_map(
        df_plot,
        hc_ll,
        plots_dir,
        script_version,
        version_filename
    ):
        import os
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.colors import Normalize

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        # Keep only lake_id + HALF retention rate
        df_map = df_plot.loc[:, ["lake_id", "retention_rate"]].copy()
        df_map["retention_pct"] = df_map["retention_rate"] * 100.0

        # Merge lat/lon
        df_map = df_map.merge(hc_ll, on="lake_id", how="left")
        df_map = df_map.dropna(subset=["p_lat", "p_lon", "retention_pct"]).copy()

        if df_map.empty:
            print("No valid data available for global HALF retention map.")
            return

        ocean_col = "#b9dce8"
        land_col = "whitesmoke"
        border_col = "#b0b0b0"
        coast_col = "#8a8a8a"

        # Same full-width style as the reference map figure, single-panel version
        fig_width = 7.2
        fig_height = 3.9
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)

        map_left = 0.02
        map_w = 0.96

        # Preserve world-map aspect, slightly shrunk to create room for title/colorbar
        map_h = (map_w * fig_width / 2.0) / fig_height * 0.94
        map_bottom = 0.185

        ax = fig.add_axes(
            [map_left, map_bottom, map_w, map_h],
            projection=ccrs.PlateCarree()
        )

        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.set_facecolor(ocean_col)
        ax.add_feature(cfeature.OCEAN, facecolor=ocean_col, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=land_col, edgecolor="none", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor=border_col, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.45, edgecolor=coast_col, zorder=2)

        lakes = cfeature.NaturalEarthFeature(
            "physical", "lakes", "50m",
            edgecolor="#9bbfd0",
            facecolor="#9fcfe3"
        )
        ax.add_feature(lakes, zorder=2.3, linewidth=0.25, alpha=0.9)

        sc = ax.scatter(
            df_map["p_lon"], df_map["p_lat"],
            c=df_map["retention_pct"],
            s=6.5,
            cmap="viridis",
            norm=Normalize(vmin=0, vmax=100),
            transform=ccrs.PlateCarree(),
            edgecolors="#4d4d4d",
            linewidths=0.12,
            alpha=0.85,
            zorder=5
        )

        ax.set_title(f"HALF retention rate (N = {len(df_map)})", fontsize=12, pad=4)
        ax.axis("off")

        # Colorbar inside map over Antarctica
        cax = ax.inset_axes([0.15, 0.120, 0.70, 0.055])
        cbar = fig.colorbar(
            sc,
            cax=cax,
            orientation="horizontal",
            extend="neither"
        )
        cbar.set_label("Retention rate (%)", fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=9)

        out_path = os.path.join(
            plots_dir,
            f"Global_HALF_retention_map_{script_version}_{version_filename}.pdf"
        )

        plt.savefig(out_path, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    # =============================================================================
    # Figure: CDF of per-lake retention rates (%)
    #   Raw / Stringent / Permissive / HALF
    # =============================================================================

    def plot_retention_cdf(
        df_plot,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        def _cdf(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            arr = np.sort(arr)
            if len(arr) == 0:
                return np.array([]), np.array([])
            return arr, np.arange(1, len(arr) + 1) / len(arr) * 100.0

        subset_stats = df_plot.copy()

        # Raw retention is always 100%
        raw_ret = np.full(len(subset_stats), 100.0, dtype=float)

        method_specs = [
            ("Raw",        raw_ret,                                             raw_color),
            ("Stringent",  subset_stats["retention_rate_benchmark_stringent"].values * 100.0, stringent_color),
            ("Permissive", subset_stats["retention_rate_benchmark_permissive"].values * 100.0, permissive_color),
            ("HALF",       subset_stats["retention_rate"].values * 100.0,           HALF_color),
        ]

        # Same CDF width/style as the reference CDF figures
        fig_width = 7.2
        fig_height = 3.0
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        half_vals_for_label = None

        for label, vals, color in method_specs:
            x, y = _cdf(vals)

            if len(x) > 0:
                ax.plot(
                    x,
                    y,
                    linewidth=2.2,
                    color=color,
                    label=label
                )

            if label == "HALF":
                half_vals_for_label = vals

        # Reference lines in percentage units
        ax.axhline(50, color="black", linestyle="--", linewidth=1)
        ax.axhline(SUMMARY_EXCEEDANCE_CDF, color="black", linestyle="--", linewidth=1)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Retention rate (%)", fontsize=11)
        ax.set_ylabel("CDF (%)", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=10)

        # Label HALF median and summary exceedance statistics
        _annotate_half_cdf_percentiles(
            ax,
            half_vals_for_label,
            color=HALF_color,
            percentiles=(50, SUMMARY_EXCEEDANCE_CDF),
            value_fmt="{:.1f}",
            fontsize=10,
            align_right=True,
            x_right_pad_frac=0.02
        )

        ax.legend(
            frameon=False,
            loc="upper left",
            fontsize=10
        )

        plt.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.20,
            top=0.86
        )

        out_name = f"Lake_retention_CDF_{script_version}_{version_filename}.{save_format}"
        out_path = os.path.join(plots_dir, out_name)

        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)


    # =============================================================================
    # Calls
    # =============================================================================
    df_ret_plot = df_lake_stats[df_lake_stats["lake_id"].isin(lakes_for_validation)].copy()

    plot_global_retention_map(
        df_ret_plot,
        hc_ll,
        plots_dir,
        script_version,
        version_filename
    )

    plot_retention_cdf(
        df_ret_plot,
        plots_dir,
        script_version,
        version_filename,
        raw_color,
        stringent_color,
        permissive_color,
        HALF_color,
        save_format="pdf",
        dpi=300
    )


    # =============================================================================
    # Print retention-rate statistics
    # =============================================================================
    print(f"\n=== Retention-rate percentiles (P50 and {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})) by method ===")

    raw_ret = np.full(len(df_ret_plot), 100.0, dtype=float)

    method_specs_print = [
        ("Raw",        raw_ret),
        ("Stringent",  df_ret_plot["retention_rate_benchmark_stringent"].values * 100.0),
        ("Permissive", df_ret_plot["retention_rate_benchmark_permissive"].values * 100.0),
        ("HALF",       df_ret_plot["retention_rate"].values * 100.0),
    ]

    for label, vals in method_specs_print:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) > 0:
            p50 = np.nanpercentile(vals, 50)
            p_exceedance = np.nanpercentile(vals, SUMMARY_EXCEEDANCE_CDF)
            p25 = np.nanpercentile(vals, 25)
            p75 = np.nanpercentile(vals, 75)

            print(
                f"  {label:<10s} n={len(vals):4d}  "
                f"P25={p25:.2f}%   P50={p50:.2f}%   "
                f"{SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})={p_exceedance:.2f}%   "
                f"P75={p75:.2f}%"
            )
        else:
            print(f"  {label:<10s} n=   0  P25=nan   P50=nan   {SUMMARY_EXCEEDANCE_PERCENTILE_LABEL} ({SUMMARY_EXCEEDANCE_LABEL})=nan   P75=nan")





    # =============================================================================
    # SI Fig. E
    # Normalized variability error figures for three variability metrics:
    #   - std    : daily WSE standard deviation
    #   - amp    : daily WSE amplitude/range = max - min
    #   - p10p90 : daily WSE 90th - 10th percentile range (interdecile range or IDR)
    #   - idr    : alias for p10p90
    #
    # For each metric:
    #   1. Global map of HALF normalized variability error
    #   2. Scatter plot + CDF
    #   3. Boxplot by lake area bins
    #
    # Normalized variability error:
    #   |V_swot - V_gauge| / V_gauge
    # =============================================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import Normalize


    # =============================================================================
    # Shared helpers
    # =============================================================================

    def _annotate_half_error_cdf_percentiles(
        ax,
        half_vals,
        color,
        percentiles=(50, SUMMARY_PERCENTILE),
        value_fmt="{:.2f}",
        fontsize=10,
        x_text_offset=5,
        y_text_offset=4
    ):
        """
        Annotate the HALF CDF curve at selected percentiles.
        CDF is assumed to be plotted in percentage units (0-100).
        """

        half_vals = np.asarray(half_vals, dtype=float)
        half_vals = half_vals[np.isfinite(half_vals)]

        if len(half_vals) == 0:
            return

        for p in percentiles:
            x_val = np.nanpercentile(half_vals, p)
            y_val = p

            if abs(p - round(p)) < 1e-8:
                p_label = f"{int(round(p))}%"
            else:
                p_label = f"{p:.2f}%"

            ax.annotate(
                f"{value_fmt.format(x_val)} ({p_label})",
                xy=(x_val, y_val),
                xytext=(x_text_offset, y_text_offset),
                textcoords="offset points",
                color=color,
                fontsize=fontsize,
                ha="left",
                va="bottom",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75
                )
            )


    def _cdf_percent(arr):
        """
        Return empirical CDF in percentage units.
        """

        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = np.sort(arr)

        if len(arr) == 0:
            return np.array([]), np.array([])

        return arr, np.arange(1, len(arr) + 1) / len(arr) * 100.0


    def _get_variability_metric_cfg(variability_metric):
        """
        Shared config for std / amp / p10p90(idr).
        """

        if variability_metric == "idr":
            variability_metric = "p10p90"

        metric_map = {
            "std": {
                "short": "std",
                "file_tag": "std",
                "gauge": "var_std_gauge_daily",
                "raw": "var_std_swot_daily_raw",
                "stringent": "var_std_swot_daily_benchmark_stringent",
                "permissive": "var_std_swot_daily_benchmark_permissive",
                "half": "var_std_swot_daily",

                # Scatter-axis labels
                "scatter_label": "std (m)",

                # Map colorbar label
                "map_cbar_label": "Normalized variability error, standard deviation (std)",

                # CDF and boxplot labels
                "rel_label_one_line": "Normalized variability error, std",
                "rel_label_two_line": "Normalized variability error\nstd"
            },

            "amp": {
                "short": "amplitude",
                "file_tag": "amp",
                "gauge": "var_amp_gauge_daily",
                "raw": "var_amp_swot_daily_raw",
                "stringent": "var_amp_swot_daily_benchmark_stringent",
                "permissive": "var_amp_swot_daily_benchmark_permissive",
                "half": "var_amp_swot_daily",

                # Scatter-axis labels
                "scatter_label": "amplitude (m)",

                # Map colorbar label
                "map_cbar_label": "Normalized variability error, amplitude (max - min)",

                # CDF and boxplot labels
                "rel_label_one_line": "Normalized variability error, amplitude",
                "rel_label_two_line": "Normalized variability error\namplitude"
            },

            "p10p90": {
                "short": "IDR",
                "file_tag": "idr",
                "gauge": "var_p10p90_gauge_daily",
                "raw": "var_p10p90_swot_daily_raw",
                "stringent": "var_p10p90_swot_daily_benchmark_stringent",
                "permissive": "var_p10p90_swot_daily_benchmark_permissive",
                "half": "var_p10p90_swot_daily",

                # Scatter-axis labels
                "scatter_label": "IDR (m)",

                # Map colorbar label
                "map_cbar_label": "Normalized variability error, interdecile range (IDR, 90th - 10th range)",

                # CDF and boxplot labels
                "rel_label_one_line": "Normalized variability error, IDR",
                "rel_label_two_line": "Normalized variability error\nIDR"
            }
        }

        if variability_metric not in metric_map:
            raise ValueError("variability_metric must be one of: 'std', 'amp', 'p10p90', or 'idr'")

        return metric_map[variability_metric]


    def _print_normvar_summary(dfv, cfg):
        """
        Print percentile summaries of normalized variability error.
        """

        print(f"\n=== Normalized variability error summary: {cfg['short']} ===")
        print("Definition: |V_swot - V_gauge| / V_gauge")

        for label, col in [
            ("Raw", "rel_err_var_raw"),
            ("Stringent", "rel_err_var_stringent"),
            ("Permissive", "rel_err_var_permissive"),
            ("HALF", "rel_err_var_half")
        ]:
            vals = pd.to_numeric(dfv[col], errors="coerce").dropna().values

            if len(vals) == 0:
                print(f"  {label:<10s} N=0")
                continue

            print(
                f"  {label:<10s} "
                f"N={len(vals):4d}, "
                f"median={np.nanmedian(vals):.3f}, "
                f"mean={np.nanmean(vals):.3f}, "
                f"{SUMMARY_PERCENTILE_LABEL}={np.nanpercentile(vals, SUMMARY_PERCENTILE):.3f}, "
                f"P90={np.nanpercentile(vals, 90):.3f}"
            )


    # =============================================================================
    # Function 1:
    # Global dot map of normalized variability error for final HALF results
    # =============================================================================

    def plot_normalized_variability_error_map(
        df_lake_stats,
        hc_ll,
        lakes_for_validation,
        plots_dir,
        script_version,
        version_filename,
        variability_metric="std",
        map_vmax=None,
        point_size=8,
        save_format="png",
        dpi=300
    ):
        """
        Make a global dot map of HALF normalized variability error.
        """

        save_format = save_format.lower().strip()
        if save_format not in ["png", "pdf"]:
            raise ValueError("save_format must be either 'png' or 'pdf'.")

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        cfg = _get_variability_metric_cfg(variability_metric)

        needed_cols = ["lake_id", cfg["gauge"], cfg["half"]]
        missing_cols = [c for c in needed_cols if c not in df_lake_stats.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in df_lake_stats: {missing_cols}")

        df_map = df_lake_stats[df_lake_stats["lake_id"].isin(lakes_for_validation)].copy()
        df_map = df_map.dropna(subset=[cfg["gauge"], cfg["half"]]).copy()
        df_map = df_map[df_map[cfg["gauge"]] > 0].copy()

        if df_map.empty:
            print(f"No valid lakes available for map using variability metric: {cfg['short']}.")
            return None

        df_map["rel_err_var_half"] = (
            np.abs(df_map[cfg["half"]] - df_map[cfg["gauge"]]) /
            df_map[cfg["gauge"]]
        )

        df_map = df_map.loc[:, ["lake_id", "rel_err_var_half"]].merge(
            hc_ll,
            on="lake_id",
            how="left"
        )
        df_map = df_map.dropna(subset=["p_lat", "p_lon", "rel_err_var_half"]).copy()

        if df_map.empty:
            print(f"No valid lat/lon data available for map using variability metric: {cfg['short']}.")
            return None

        if map_vmax is None:
            vmax = np.nanpercentile(df_map["rel_err_var_half"], 95)

            if not np.isfinite(vmax) or vmax <= 0:
                vmax = np.nanmax(df_map["rel_err_var_half"])

            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
        else:
            vmax = map_vmax

        ocean_col = "#b9dce8"
        land_col = "whitesmoke"
        border_col = "#b0b0b0"
        coast_col = "#8a8a8a"

        fig_width = 7.2
        fig_height = 4.0
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        map_left = 0.02
        map_w = 0.96
        map_h = (map_w * fig_width / 2.0) / fig_height * 0.94
        map_bottom = 0.24

        ax = fig.add_axes(
            [map_left, map_bottom, map_w, map_h],
            projection=ccrs.PlateCarree()
        )

        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.set_facecolor(ocean_col)
        ax.add_feature(cfeature.OCEAN, facecolor=ocean_col, edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=land_col, edgecolor="none", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor=border_col, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor=coast_col, zorder=2)

        lakes = cfeature.NaturalEarthFeature(
            "physical", "lakes", "50m",
            edgecolor="#9bbfd0",
            facecolor="#9fcfe3"
        )
        ax.add_feature(lakes, zorder=2.3, linewidth=0.25, alpha=0.9)

        sc = ax.scatter(
            df_map["p_lon"], df_map["p_lat"],
            c=df_map["rel_err_var_half"],
            s=point_size,
            cmap="viridis",
            norm=Normalize(vmin=0, vmax=vmax),
            transform=ccrs.PlateCarree(),
            edgecolors="#4d4d4d",
            linewidths=0.12,
            alpha=0.9,
            zorder=5
        )

        map_title_metric = cfg["short"]
        if cfg["file_tag"] == "std":
            map_title_metric = "standard deviation"

        ax.set_title(
            f"HALF normalized variability error, {map_title_metric} (N = {len(df_map)})",
            fontsize=12,
            pad=4
        )
        ax.axis("off")

        cax = ax.inset_axes([0.15, 0.120, 0.70, 0.055])
        cbar = fig.colorbar(
            sc,
            cax=cax,
            orientation="horizontal",
            extend="max"
        )
        cbar.set_label(cfg["map_cbar_label"], fontsize=10, labelpad=2)
        cbar.ax.tick_params(labelsize=9)

        out_path = os.path.join(
            plots_dir,
            f"Global_HALF_norm_var_error_map_{cfg['file_tag']}_{script_version}_{version_filename}.{save_format}"
        )
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)
        print(f"Map N ({cfg['short']}): {len(df_map)}")

        return df_map


    # =============================================================================
    # Function 2:
    # Variability scatter + normalized variability error CDF
    # =============================================================================

    def plot_variability_scatter_and_normalized_error_cdf(
        df_lake_stats,
        lakes_for_validation,
        plots_dir,
        script_version,
        version_filename,
        variability_metric="std",
        raw_color="#7F7F7F",
        stringent_color="#0072B2",
        permissive_color="#E69F00",
        HALF_color="#CC79A7",
        scatter_xlim=None,
        scatter_ylim=None,
        cdf_xlim=None,
        save_format="png",
        dpi=300
    ):
        """
        Make a two-panel figure:
          Left:  Gauge variability vs LakeSP variability scatterplot.
          Right: CDF of normalized variability error.
        """

        save_format = save_format.lower().strip()
        if save_format not in ["png", "pdf"]:
            raise ValueError("save_format must be either 'png' or 'pdf'.")

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        cfg = _get_variability_metric_cfg(variability_metric)

        needed_cols = [
            "lake_id",
            cfg["gauge"],
            cfg["raw"],
            cfg["stringent"],
            cfg["permissive"],
            cfg["half"]
        ]
        missing_cols = [c for c in needed_cols if c not in df_lake_stats.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in df_lake_stats: {missing_cols}")

        df_var = df_lake_stats[df_lake_stats["lake_id"].isin(lakes_for_validation)].copy()
        df_var = df_var.dropna(subset=[cfg["gauge"]]).copy()
        df_var = df_var[df_var[cfg["gauge"]] > 0].copy()

        if df_var.empty:
            print(f"No valid lakes available for variability metric: {cfg['short']}.")
            return None

        gauge = df_var[cfg["gauge"]]

        df_var["rel_err_var_raw"] = np.abs(df_var[cfg["raw"]] - gauge) / gauge
        df_var["rel_err_var_stringent"] = np.abs(df_var[cfg["stringent"]] - gauge) / gauge
        df_var["rel_err_var_permissive"] = np.abs(df_var[cfg["permissive"]] - gauge) / gauge
        df_var["rel_err_var_half"] = np.abs(df_var[cfg["half"]] - gauge) / gauge

        if scatter_xlim is None or scatter_ylim is None:
            vals_for_limits = np.concatenate([
                df_var[cfg["gauge"]].to_numpy(dtype=float),
                df_var[cfg["raw"]].to_numpy(dtype=float),
                df_var[cfg["stringent"]].to_numpy(dtype=float),
                df_var[cfg["permissive"]].to_numpy(dtype=float),
                df_var[cfg["half"]].to_numpy(dtype=float)
            ])
            vals_for_limits = vals_for_limits[np.isfinite(vals_for_limits) & (vals_for_limits > 0)]

            if len(vals_for_limits) > 0:
                vmin = 10 ** np.floor(np.log10(np.nanpercentile(vals_for_limits, 1)))
                vmax = 10 ** np.ceil(np.log10(np.nanpercentile(vals_for_limits, 99)))
            else:
                vmin, vmax = 0.01, 100

            if scatter_xlim is None:
                scatter_xlim = (vmin, vmax)
            if scatter_ylim is None:
                scatter_ylim = (vmin, vmax)

        if cdf_xlim is None:
            all_norm_vals = np.concatenate([
                df_var["rel_err_var_raw"].to_numpy(dtype=float),
                df_var["rel_err_var_stringent"].to_numpy(dtype=float),
                df_var["rel_err_var_permissive"].to_numpy(dtype=float),
                df_var["rel_err_var_half"].to_numpy(dtype=float)
            ])
            all_norm_vals = all_norm_vals[np.isfinite(all_norm_vals)]

            if len(all_norm_vals) > 0:
                xmax = np.nanpercentile(all_norm_vals, 95)
                xmax = max(2, xmax)
            else:
                xmax = 2

            cdf_xlim = (0, xmax)

        # Figure layout:
        # scatter panel made narrower to look more square
        fig = plt.figure(figsize=(7.2, 3.2), dpi=dpi)

        # [left, bottom, width, height]
        ax_scatter = fig.add_axes([0.13, 0.22, 0.32, 0.66])
        ax_cdf     = fig.add_axes([0.57, 0.22, 0.37, 0.66])

        alpha = 0.25

        # -------------------------
        # Left: scatter
        # -------------------------
        ax_scatter.scatter(
            df_var[cfg["gauge"]],
            df_var[cfg["raw"]],
            color=raw_color,
            s=14,
            linewidth=0,
            alpha=0.30
        )
        ax_scatter.scatter(
            df_var[cfg["gauge"]],
            df_var[cfg["stringent"]],
            color=stringent_color,
            s=14,
            linewidth=0,
            alpha=alpha
        )
        ax_scatter.scatter(
            df_var[cfg["gauge"]],
            df_var[cfg["permissive"]],
            color=permissive_color,
            s=14,
            linewidth=0,
            alpha=alpha
        )
        ax_scatter.scatter(
            df_var[cfg["gauge"]],
            df_var[cfg["half"]],
            color=HALF_color,
            s=14,
            linewidth=0,
            alpha=alpha
        )

        vmin = min(scatter_xlim[0], scatter_ylim[0])
        vmax = max(scatter_xlim[1], scatter_ylim[1])
        ax_scatter.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1)

        ax_scatter.set_xscale("log")
        ax_scatter.set_yscale("log")
        ax_scatter.set_xlim(scatter_xlim)
        ax_scatter.set_ylim(scatter_ylim)

        ax_scatter.set_xlabel(f"Gauge WSE variability, {cfg['scatter_label']}")
        ax_scatter.set_ylabel(f"LakeSP WSE variability\n{cfg['scatter_label']}")
        ax_scatter.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)


        # -------------------------
        # Right: CDF
        # -------------------------
        method_specs = [
            ("Raw",        df_var["rel_err_var_raw"].values,        raw_color),
            ("Stringent",  df_var["rel_err_var_stringent"].values,  stringent_color),
            ("Permissive", df_var["rel_err_var_permissive"].values, permissive_color),
            ("HALF",       df_var["rel_err_var_half"].values,       HALF_color),
        ]

        half_vals_for_label = None

        for label, vals, color in method_specs:
            x, y = _cdf_percent(vals)
            if len(x) > 0:
                ax_cdf.plot(x, y, linewidth=2.2, color=color, label=label)
            if label == "HALF":
                half_vals_for_label = vals

        ax_cdf.axhline(50, color="black", linestyle="--", linewidth=1)
        ax_cdf.axhline(SUMMARY_PERCENTILE, color="black", linestyle="--", linewidth=1)

        ax_cdf.set_xlim(cdf_xlim)
        ax_cdf.set_ylim(0, 100)
        ax_cdf.set_xlabel(cfg["rel_label_one_line"])
        ax_cdf.set_ylabel("CDF (%)")
        ax_cdf.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        _annotate_half_error_cdf_percentiles(
            ax_cdf,
            half_vals_for_label,
            color=HALF_color,
            percentiles=(50, SUMMARY_PERCENTILE),
            value_fmt="{:.2f}",
            fontsize=10
        )

        # Legend on the CDF panel only
        ax_cdf.legend(frameon=False, loc="lower right", fontsize=10)

        out_path = os.path.join(
            plots_dir,
            f"Lake_variability_scatter_and_norm_error_CDF_{cfg['file_tag']}_{script_version}_{version_filename}.{save_format}"
        )
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)
        _print_normvar_summary(df_var, cfg)

        return df_var


    # =============================================================================
    # Function 3:
    # Normalized variability error vs lake size bins
    # =============================================================================

    def plot_norm_var_error_by_area_bins(
        df_lake_stats,
        lakes_for_validation,
        plots_dir,
        script_version,
        version_filename,
        variability_metric="std",
        raw_color="#7F7F7F",
        stringent_color="#0072B2",
        permissive_color="#E69F00",
        HALF_color="#CC79A7",
        area_edges=None,
        area_labels=None,
        ylim=None,
        yticks=None,
        showfliers=False,
        save_format="png",
        dpi=300
    ):
        """
        Make grouped boxplots of normalized variability error vs lake area bins.
        """

        save_format = save_format.lower().strip()
        if save_format not in ["png", "pdf"]:
            raise ValueError("save_format must be either 'png' or 'pdf'.")

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42
        })

        cfg = _get_variability_metric_cfg(variability_metric)

        needed_cols = [
            "lake_id", "lake_area",
            cfg["gauge"], cfg["raw"], cfg["stringent"], cfg["permissive"], cfg["half"]
        ]
        missing_cols = [c for c in needed_cols if c not in df_lake_stats.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in df_lake_stats: {missing_cols}")

        dfv = df_lake_stats[df_lake_stats["lake_id"].isin(lakes_for_validation)].copy()
        dfv = dfv.dropna(subset=["lake_area", cfg["gauge"]]).copy()
        dfv = dfv[dfv["lake_area"] > 0].copy()
        dfv = dfv[dfv[cfg["gauge"]] > 0].copy()

        if dfv.empty:
            print(f"No valid lakes available for area-bin plot using variability metric: {cfg['short']}.")
            return None

        dfv["rel_err_var_raw"] = np.abs(dfv[cfg["raw"]] - dfv[cfg["gauge"]]) / dfv[cfg["gauge"]]
        dfv["rel_err_var_stringent"] = np.abs(dfv[cfg["stringent"]] - dfv[cfg["gauge"]]) / dfv[cfg["gauge"]]
        dfv["rel_err_var_permissive"] = np.abs(dfv[cfg["permissive"]] - dfv[cfg["gauge"]]) / dfv[cfg["gauge"]]
        dfv["rel_err_var_half"] = np.abs(dfv[cfg["half"]] - dfv[cfg["gauge"]]) / dfv[cfg["gauge"]]

        if area_edges is None:
            area_edges = [0, 0.1, 1, 10, 100, 1000, 10000, np.inf]

        if area_labels is None:
            area_labels = [
                "≤ 0.1",
                "≤ 1",
                "≤ 10",
                "≤ 100",
                "≤ 1,000",
                "≤ 10,000",
                "> 10,000"
            ]

        if len(area_edges) != len(area_labels) + 1:
            raise ValueError("area_edges must have length = len(area_labels) + 1")

        dfv["area_bin"] = pd.cut(
            dfv["lake_area"],
            bins=area_edges,
            labels=area_labels,
            include_lowest=True
        )

        present_labels = [lab for lab in area_labels if (dfv["area_bin"] == lab).any()]

        if len(present_labels) == 0:
            print("No non-empty lake-area bins available for plotting.")
            return dfv

        bins_list = list(range(1, len(present_labels) + 1))

        data_by_group = {
            "Raw": [
                dfv.loc[dfv["area_bin"] == lab, "rel_err_var_raw"].dropna().to_numpy()
                for lab in present_labels
            ],
            "Stringent": [
                dfv.loc[dfv["area_bin"] == lab, "rel_err_var_stringent"].dropna().to_numpy()
                for lab in present_labels
            ],
            "Permissive": [
                dfv.loc[dfv["area_bin"] == lab, "rel_err_var_permissive"].dropna().to_numpy()
                for lab in present_labels
            ],
            "HALF": [
                dfv.loc[dfv["area_bin"] == lab, "rel_err_var_half"].dropna().to_numpy()
                for lab in present_labels
            ]
        }

        method_colors = {
            "Raw": raw_color,
            "Stringent": stringent_color,
            "Permissive": permissive_color,
            "HALF": HALF_color
        }

        offsets = {
            "Raw": -0.30,
            "Stringent": -0.10,
            "Permissive": 0.10,
            "HALF": 0.30
        }

        fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=dpi)
        ax.grid(False)

        _grouped_boxplot(
            ax,
            data_by_group,
            bins_list,
            offsets,
            method_colors,
            widths=0.16,
            alpha=0.45,
            showfliers=showfliers
        )

        ax.set_xticks(bins_list)
        ax.set_xticklabels(present_labels)
        ax.set_xlabel("Lake area bin (km²)")
        ax.set_ylabel(cfg["rel_label_two_line"])

        if yticks is not None:
            ax.set_yticks(yticks)
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.subplots_adjust(
            left=0.10,
            right=0.98,
            bottom=0.24,
            top=0.92
        )

        out_path = os.path.join(
            plots_dir,
            f"NormVarError_box_by_area_{cfg['file_tag']}_{script_version}_{version_filename}.{save_format}"
        )
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.show()

        print("Saved figure:", out_path)

        print(f"\n=== Normalized variability error by lake area bin: {cfg['short']} ===")
        for lab in present_labels:
            sub = dfv[dfv["area_bin"] == lab]
            print(f"  Area bin {lab}: n = {len(sub)}")

            vals_half = pd.to_numeric(sub["rel_err_var_half"], errors="coerce").dropna().values
            if len(vals_half) > 0:
                print(
                    f"    HALF: median={np.nanmedian(vals_half):.3f}, "
                    f"mean={np.nanmean(vals_half):.3f}, "
                    f"{SUMMARY_PERCENTILE_LABEL}={np.nanpercentile(vals_half, SUMMARY_PERCENTILE):.3f}"
                )

        return dfv


    # =============================================================================
    # Calls
    # Make map, scatter+CDF, and area-bin boxplot for each variability metric
    # =============================================================================

    si5_save_format = "pdf"

    # Customize scatter-axis limits here after experimentation.
    scatter_limits_by_metric = {
        "amp": (0.1, 1000),
        "std": (0.01, 100),
        "idr": (0.02, 400)
    }

    # Customize CDF x-axis limits here.
    cdf_limits_by_metric = {
        "amp": (0, 2),
        "std": (0, 2),
        "idr": (0, 2)
    }

    for metric in ["amp", "std", "idr"]:

        print("\n" + "=" * 80)
        print(f"Processing SI Fig. 5 variability metric: {metric}")
        print("=" * 80)

        scatter_lim = scatter_limits_by_metric[metric]
        cdf_lim = cdf_limits_by_metric[metric]

        df_var_map = plot_normalized_variability_error_map(
            df_lake_stats=df_lake_stats,
            hc_ll=hc_ll,
            lakes_for_validation=lakes_for_validation,
            plots_dir=plots_dir,
            script_version=script_version,
            version_filename=version_filename,
            variability_metric=metric,
            map_vmax=2,
            point_size=8,
            save_format=si5_save_format,
            dpi=300
        )

        df_var_scatter_cdf = plot_variability_scatter_and_normalized_error_cdf(
            df_lake_stats=df_lake_stats,
            lakes_for_validation=lakes_for_validation,
            plots_dir=plots_dir,
            script_version=script_version,
            version_filename=version_filename,
            variability_metric=metric,
            raw_color=raw_color,
            stringent_color=stringent_color,
            permissive_color=permissive_color,
            HALF_color=HALF_color,
            scatter_xlim=scatter_lim,
            scatter_ylim=scatter_lim,
            cdf_xlim=cdf_lim,
            save_format=si5_save_format,
            dpi=300
        )

        df_normvar_area = plot_norm_var_error_by_area_bins(
            df_lake_stats=df_lake_stats,
            lakes_for_validation=lakes_for_validation,
            plots_dir=plots_dir,
            script_version=script_version,
            version_filename=version_filename,
            variability_metric=metric,
            raw_color=raw_color,
            stringent_color=stringent_color,
            permissive_color=permissive_color,
            HALF_color=HALF_color,
            ylim=(-0.2, 11.5),
            yticks=[0, 2, 4, 6, 8, 10],
            save_format=si5_save_format,
            dpi=300
        )
