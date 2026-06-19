"""
Core functions for the Heuristic Adaptive Lake Filter (HALF).

This module contains reusable routines for calibrating lake-specific heuristic
thresholds, applying threshold and low-pass filters to SWOT LakeSP WSE time
series, computing validation metrics, and reducing cross-pass WSE offsets.

The functions are intentionally written as standalone utilities so they can be
imported by other scripts. No function reads project-specific files directly.

Version
-------
HALF v1.0
Last updated: 2026-06-17

Script by:
-------
Jida Wang (jidaw@illinois.edu)
Mélanie Trudel (melanie.trudel@usherbrooke.ca)

Parts of this script were developed with assistance from ChatGPT (OpenAI) for 
brainstorming, debugging, drafting, documentation, and related editorial suggestions.

The authors are responsible for the conceptual design, maintenance, methodology,
review, testing, and validation of the final implementation.

Citation
--------
Trudel, M., Wang, J., Biancamaria, S., Harlan, M.E., Shah, D., Gao, H.,
Collins, E., Getirana, A., Song, C., Reis Alencar Oliveira, R., Gosset, M.,
Rodrigues Martins, E.S., Fleischmann, A., Hymans, D., Grippa, M., Girard, F.,
Kergoat, L., Pottier, C., Fjørtoft, R., Oubanas, H., & Pavelsky, T.M. (2026).
A Heuristic Adaptive Filter for SWOT Lake Vector Data Products.
Geophysical Research Letters, in review.


Function overview
-------
This overview summarizes the public functions and constants listed in __all__.
Internal helper functions whose names begin with "_" are not listed here.
See the docstring of each function for detailed definitions, functionality,
parameters, and return values.

Public constants
  SUPPORTED_LOW_PASS_FILTERS           Names of supported smoothing filters.

Key HALF functions
  calibrate_heuristic_thresholds       Lake-specific threshold calibration.
  apply_heuristic_thresholds           Apply calibrated thresholds to LakeSP.
  apply_customized_filter              Full HALF iterative filtering workflow.
  apply_baseline_tukey_filter          Tukey fallback for sparse cases.
  sp_cycle_adjustment                  Intra-cycle cross-pass bias correction.

Time-series smoothing filters
  filter_lowess                       LOWESS smoothing.
  filter_savgol                       Savitzky-Golay smoothing.
  filter_wavelet                      Wavelet denoising.
  filter_hampel                       Hampel filtering.
  filter_spline                       Spline smoothing.
  filter_median                       Median filtering.
  filter_kalman                       Kalman filtering.

Supporting utilities
  remove_tukey_outliers               General Tukey/IQR outlier removal.
  filter_ice_outliers                 Optional ice-period outlier screening.
  convert_to_daily_series             Daily interpolation for validation.
  signed_min_abs_residual             Residual selection from filter envelopes.
  drop_eval_in_apply_gaps             Exclude residual tests inside baseline gaps.

Basic validation metrics
  compute_mae                         Mean absolute error, NaN robust.
  compute_median_residual             Signed median residual, NaN robust.
  compute_correlation                 Pearson or Spearman correlation.
  compute_variability_std             Variability as standard deviation.
  compute_variability_amplitude       Variability as max-min range.
  compute_variability_p10_p90_range   Variability as P90-P10, or interdecile range (IDR).
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import pywt
except ImportError:  # Optional dependency; required only for filter_wavelet().
    pywt = None
import statsmodels.api as sm
from joblib import Parallel, delayed
try:
    from pykalman import KalmanFilter
except ImportError:  # Optional dependency; required only for filter_kalman().
    KalmanFilter = None
from scipy.interpolate import PchipInterpolator, UnivariateSpline, interp1d
from scipy.signal import medfilt, savgol_filter
from scipy.stats import pearsonr, spearmanr

SUPPORTED_LOW_PASS_FILTERS = (
    "lowess", "wavelet", "savgol", "kalman", "spline", "median", "hampel"
)

# This list controls what is imported when users run: 
# from half_v1_0_functions import *
# It includes the functions/constants intended for public use.
# Internal helper functions whose names begin with "_" are intentionally excluded.
__all__ = [
    "SUPPORTED_LOW_PASS_FILTERS",
    "compute_median_residual",
    "compute_mae",
    "compute_correlation",
    "compute_variability_std",
    "compute_variability_amplitude",
    "compute_variability_p10_p90_range",
    "remove_tukey_outliers",
    "calibrate_heuristic_thresholds",
    "apply_heuristic_thresholds",
    "filter_ice_outliers",
    "convert_to_daily_series",
    "signed_min_abs_residual",
    "filter_lowess",
    "filter_savgol",
    "filter_wavelet",
    "filter_hampel",
    "filter_spline",
    "filter_median",
    "filter_kalman",
    "drop_eval_in_apply_gaps",
    "apply_customized_filter",
    "apply_baseline_tukey_filter",
    "sp_cycle_adjustment",
]

def _crid_suffix2(x):
    """
    Return the two-character CRID scenario suffix used for threshold grouping.

    Examples are "C0", "C2", and "D0". Missing or malformed CRID
    values return "unknown". LakeSP CRID conventions can evolve, so review
    this helper when adding support for future product versions.
    """
    if x is None or pd.isna(x):
        return "unknown"
    s = str(x).strip()
    return s[-2:] if len(s) >= 2 else "unknown"

def compute_median_residual(y, y_hat):
    """
    Computes median signed residual.

    Residual is defined as:
        y_hat - y

    Therefore:
        positive value = y_hat is higher than y
        negative value = y_hat is lower than y

    Parameters:
        y (numeric array-like): Ground truth/reference values.
        y_hat (numeric array-like): Predicted or comparison values.

    Returns:
        median_residual: Median signed residual, or np.nan if no valid data points exist.
    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    mask = ~np.isnan(y) & ~np.isnan(y_hat)  # Valid non-NaN pairs

    if np.sum(mask) == 0:
        return np.nan  # Return NaN if all data is invalid

    median_residual = np.median(y_hat[mask] - y[mask])
    return median_residual

def compute_mae(y, y_hat):
    """
    Compute mean absolute error (MAE).

    Parameters:
        y and y_hat (numeric array-like): The two vectors to compare (order does not matter.)

    Returns:
        mae: Computed MAE value, or np.nan if no valid data points exist.
    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    mask = ~np.isnan(y) & ~np.isnan(y_hat)  # Valid (non-NaN) pairs

    if np.sum(mask) == 0:
        return np.nan  # Return NaN if all data is invalid

    mae = np.mean(abs(y[mask] - y_hat[mask])) #rmse: np.sqrt(np.mean((y[mask] - y_hat[mask])**2))
    return mae

def compute_correlation(y, y_hat, method='pearson'):
    """
    Compute the correlation coefficient between two paired numeric arrays.

    Parameters:
        y (array-like): Ground truth values.
        y_hat (array-like): Predicted or comparison values.
        method (str): 'spearman' or 'pearson' (default: 'pearson').

    Returns:
        float: Correlation coefficient (rho or r), or np.nan if insufficient valid data.
    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    mask = ~np.isnan(y) & ~np.isnan(y_hat)

    if np.sum(mask) < 2: # Need at least two valid points to compute correlation
        return np.nan

    y_valid, y_hat_valid = y[mask], y_hat[mask]

    if method == 'spearman':
        rho, _ = spearmanr(y_valid, y_hat_valid)
        return rho
    elif method == 'pearson':
        r, _ = pearsonr(y_valid, y_hat_valid)
        return r
    else:
        raise ValueError("Invalid method. Choose 'spearman' or 'pearson'.")

def _to_finite_1d_array(x):
    """
    Convert scalar/array-like/Series input to a 1D finite numeric array.
    NaN, None, inf, and non-numeric values are removed.
    """
    if x is None:
        return np.array([], dtype=float)

    if isinstance(x, (pd.Series, pd.Index)):
        arr = pd.to_numeric(x, errors='coerce').to_numpy(dtype=float)
    elif np.isscalar(x):
        arr = pd.to_numeric(pd.Series([x]), errors='coerce').to_numpy(dtype=float)
    else:
        arr = pd.to_numeric(pd.Series(np.asarray(x).ravel()), errors='coerce').to_numpy(dtype=float)

    return arr[np.isfinite(arr)]

def compute_variability_std(x, ddof=0):
    """
    Computes variability as standard deviation.

    Parameters:
        x (numeric scalar or array-like): Input WSE time series.
        ddof (int): Delta degrees of freedom. Default is 0; pass ddof=1 to match pandas Series.std().

    Returns:
        float: Standard deviation, 0 for a single valid value, or np.nan if no valid values exist.
    """
    arr = _to_finite_1d_array(x)

    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0

    return float(np.std(arr, ddof=ddof))

def compute_variability_amplitude(x):
    """
    Computes variability as amplitude/range: max(x) - min(x).

    Parameters:
        x (numeric scalar or array-like): Input WSE time series.

    Returns:
        float: Range, 0 for a single valid value, or np.nan if no valid values exist.
    """
    arr = _to_finite_1d_array(x)

    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0

    return float(np.max(arr) - np.min(arr))

def compute_variability_p10_p90_range(x):
    """
    Computes variability as the 10th–90th percentile range.

    Parameters:
        x (numeric scalar or array-like): Input WSE time series.

    Returns:
        float: P90 - P10 range, 0 for a single valid value, or np.nan if no valid values exist.
    """
    arr = _to_finite_1d_array(x)

    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0

    return float(np.nanpercentile(arr, 90) - np.nanpercentile(arr, 10))


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

def calibrate_heuristic_thresholds(
    df,
    conservative_SQL,
    by_crid_scenario=[False, False, False],
    by_pass_id=[False, False, True],
    by_ice=[True, True, True]
):
    """
    Calibrate lake-specific heuristic thresholds for SWOT LakeSP filtering.

    This function estimates threshold values for three diagnostic variables used
    by HALF:

        - wse_std:    within-lake spatial variability of WSE, in metres
        - wse_u:      reported LakeSP WSE uncertainty, in metres
        - xtrk_dist:  cross-track distance from the lake polygon centroid to
                      nadir, in metres; abs(xtrk_dist) is used

    Overview
    --------
    The thresholds are calibrated from a conservative, high-quality subset
    (hereafter the "calibration subset") of the LakeSP time series, defined by
    conservative_SQL.

    The purpose is not to use the LakeSP quality flags as the final filter, but
    to use them to identify a reliable training subset from which lake-specific
    threshold values can be learned.

    Thresholds can be calibrated for the lake globally or stratified by:

        - CRID scenario, such as C0, C2, or D0;
        - SWOT orbit pass;
        - ice condition, based on ice_clim_f.

    These stratification choices are controlled independently for each
    diagnostic variable. This is useful because different variables may depend
    on different physical or processing conditions. For example, xtrk_dist is
    naturally pass-dependent because a lake can be located differently within
    the swath for different passes, while wse_std and wse_u may or may not
    require pass-specific or version-specific calibration depending on the
    application.

    Returned threshold table
    ------------------------
    The returned DataFrame is designed to support downstream filtering by
    apply_heuristic_thresholds(). For each observed (crid_scenario, pass_id)
    combination, the function returns threshold rows for:

        - ice-free observations;
        - ice-covered observations;
        - both ice conditions pooled together.

    The auxiliary "both" row represents thresholds calibrated without
    separating ice-free and ice-covered observations. This row is useful when
    the user chooses to apply a pooled threshold rather than an ice-specific
    threshold.

    If a threshold cannot be computed directly because the calibration subset
    has too few observations for a requested group, structured fallback rules
    are used. For example, a missing ice-covered, pass-specific wse_std
    threshold may fall back to a threshold from the same ice condition and then
    to broader groupings. If no related threshold exists, the value remains NaN
    and is later handled by bounds and missing-value defaults in
    apply_heuristic_thresholds().

    Parameters
    ----------
    df : pandas.DataFrame
        Input LakeSP time series for one lake. The DataFrame must contain at
        least:

            ['lake_id', 'crid', 'pass_id', 'ice_clim_f',
             'wse_std', 'wse_u', 'xtrk_dist']

    conservative_SQL : str
        pandas query expression used to select the conservative, high-quality
        calibration subset.

        Example:

            conservative_SQL = "(xovr_cal_q < 1) & (quality_f < 1)"

        Only input observations satisfying this expression are used to compute
        the initial calibrated thresholds. The returned threshold table may
        still include groups that are present in the full input time series but
        not represented in the calibration subset; missing thresholds for those
        groups are filled using the fallback rules described below when possible.

    by_crid_scenario : list of bool, length 3
        Per-metric control for whether thresholds are calibrated separately by
        CRID scenario. The entries correspond to:

            [wse_std, wse_u, xtrk_dist]

        Example:

            by_crid_scenario = [False, False, False]

        means that none of the three thresholds is calibrated separately by
        CRID scenario.

    by_pass_id : list of bool, length 3
        Per-metric control for whether thresholds are calibrated separately by
        SWOT pass_id.

        Example:

            by_pass_id = [False, False, True]

        means that wse_std and wse_u thresholds are not pass-specific, while
        xtrk_dist thresholds are calibrated separately for each pass.

    by_ice : list of bool, length 3
        Per-metric control for whether thresholds are calibrated separately by
        ice condition.

        Example:

            by_ice = [True, True, True]

        means that thresholds for wse_std, wse_u, and xtrk_dist are calibrated
        separately for ice-free and ice-covered observations when sufficient
        calibration observations are available.

    Notes
    -----
    xtrk_dist is the distance, in metres, from the lake polygon centroid to the
    spacecraft nadir track. Negative and positive values indicate opposite sides
    of the swath relative to the spacecraft velocity vector. HALF uses
    abs(xtrk_dist) for threshold calibration because distance from nadir, rather
    than swath side, is the relevant diagnostic quantity.

    Key steps
    ---------
    1) Base-row construction:
       - Extract unique combinations of
         (crid_scenario, pass_id, ice_condition) from df.
       - If only one ice_condition is present for a given
         (crid_scenario, pass_id), add the missing ice-specific row and mark it
         with grouping_scheme = 3.
       - Add an auxiliary "both" row, representing pooled ice conditions, for
         every observed (crid_scenario, pass_id) pair.
       - Preserve the order of (crid_scenario, pass_id) pairs as they first
         appear in df.

    2) Grouping-scheme marking, before fallback rules:
       - For ice-specific rows (ice_condition = "ice-free" or "ice-covered"):
           * grouping_scheme = 1 -> the exact
             (crid_scenario, pass_id, ice_condition) combination exists in the
             conservative calibration subset.
           * grouping_scheme = 2 -> the exact
             (crid_scenario, pass_id, ice_condition) combination exists in the
             full input df but not in the conservative calibration subset.
           * grouping_scheme = 3 -> the ice-specific row is synthetic, meaning
             this ice condition was not observed for that
             (crid_scenario, pass_id) pair in the full input df. The row is
             added only for table completeness and later threshold-selection
             flexibility.

       - For auxiliary "both" rows (ice_condition = "both"):
           * One "both" row is added for every observed
             (crid_scenario, pass_id) pair. This row represents a pooled
             threshold that ignores ice condition.
           * grouping_scheme = 1 -> the (crid_scenario, pass_id) pair has at
             least one observation in the conservative calibration subset,
             regardless of ice condition.
           * grouping_scheme = 2 -> the (crid_scenario, pass_id) pair exists in
             the full input df but has no observation in the conservative
             calibration subset.
           * grouping_scheme = 3 is not used for "both" rows.

    3) Threshold calibration from the calibration subset:
       - Always compute initial thresholds from df.query(conservative_SQL).
       - Ice-specific rows use the per-metric stratification controls
         by_crid_scenario, by_pass_id, and by_ice.
       - "Both" rows use the same per-metric CRID/pass controls but ignore
         by_ice.
       - Aggregation rules are:
           * wse_std_thr_cal and wse_u_thr_cal = maximum
           * xtrk_dist_thr_cal = minimum

    4) Fallback rules for missing calibrated thresholds:
       - Ice-specific rows ("ice-covered" or "ice-free"):
           * wse_std and wse_u, using ice-condition-centered fallback:
               L1: same ice_condition + pass_id -> maximum
               L2: same ice_condition + crid_scenario -> maximum
               L3: same ice_condition -> maximum
               L4: remain NaN, awaiting a bounds-based default in
                   apply_heuristic_thresholds().
           * xtrk_dist, using pass-centered fallback:
               L1: same pass_id + ice_condition -> minimum
               L2: same pass_id + crid_scenario -> minimum
               L3: same pass_id -> minimum
               L4: remain NaN, awaiting a bounds-based default in
                   apply_heuristic_thresholds().

       - Auxiliary "both" rows, considering only other "both" rows:
           * wse_std and wse_u:
               L1: same pass_id -> maximum
               L2: same crid_scenario -> maximum
               L3: remain NaN, awaiting a bounds-based default in
                   apply_heuristic_thresholds().
           * xtrk_dist:
               L1: same pass_id -> minimum
               L2: same crid_scenario -> minimum
               L3: remain NaN, awaiting a bounds-based default in
                   apply_heuristic_thresholds().

    5) Finalization:
       - Sort rows by the order in which each (crid_scenario, pass_id) pair
         first appears in df, then by ice_condition.

    Returns
    -------
    pandas.DataFrame
        Lake-specific calibrated threshold table with the following columns:

        lake_id : int or str
            PLD lake ID.

        crid_scenario : str
            Two-character CRID suffix, such as C0, C2, or D0.

        pass_id : int
            SWOT orbit pass ID.

        ice_condition : str
            Threshold ice-condition group:
                - "ice-free"    : observations with ice_clim_f < 2
                - "ice-covered" : observations with ice_clim_f >= 2
                - "both"        : pooled ice-free and ice-covered observations

        wse_std_thr_cal : float
            Calibrated upper threshold for wse_std.

        wse_u_thr_cal : float
            Calibrated upper threshold for wse_u.

        xtrk_dist_thr_cal : float
            Calibrated lower threshold for abs(xtrk_dist).

        grouping_scheme : int
            Provenance flag describing how the threshold row was represented in
            the input and calibration subset. See Key step 2 for the exact
            interpretation for ice-specific and auxiliary "both" rows.

    Examples
    --------
    Default-style grouping used in the HALF workflow:

        df_thresholds = calibrate_heuristic_thresholds(
            df,
            conservative_SQL,
            by_crid_scenario=[False, False, False],
            by_pass_id=[False, False, True],
            by_ice=[True, True, True],
        )

    This means:

        - wse_std thresholds are calibrated by ice condition, but not separately
          by CRID scenario or pass.
        - wse_u thresholds are calibrated by ice condition, but not separately
          by CRID scenario or pass.
        - xtrk_dist thresholds are calibrated by pass and ice condition, but not
          separately by CRID scenario.
    """

    # -------------------------
    # Handle empty df
    # -------------------------
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                'lake_id','crid_scenario','pass_id','ice_condition',
                'wse_std_thr_cal','wse_u_thr_cal','xtrk_dist_thr_cal','grouping_scheme'
            ]
        )
    lake_id = df['lake_id'].iloc[0] if ('lake_id' in df.columns) else 'unknown'

    # -------------------------
    # Preprocess
    # -------------------------
    df = df.copy()
    df['crid_scenario'] = df['crid'].apply(_crid_suffix2)
    df['xtrk_dist_abs'] = df['xtrk_dist'].abs()
    df['ice_condition'] = np.where(df['ice_clim_f'] >= 2, 'ice-covered', 'ice-free')

    # Conservative subset used for threshold calibration
    try:
        df_cons = df.query(conservative_SQL, engine="python")
    except Exception as e:
        raise ValueError(f"Invalid conservative_SQL: {e}")

    # -------------------------
    # Build base_df to ensure BOTH ice states exist for each (crid_scenario, pass_id),
    # and then append a "both" row per pair
    # -------------------------
    # Keep original pair order for final sorting
    pairs_order = (
        df[['crid_scenario','pass_id']]
        .drop_duplicates(keep='first')
        .reset_index(drop=True)
    )
    pairs_order['__pair_order__'] = np.arange(len(pairs_order))

    # Ensure both ice states exist per pair; mark synthetic with grouping_scheme=3
    triples_df = df[['crid_scenario','pass_id','ice_condition']].drop_duplicates()
    full_rows = []
    for _, prow in pairs_order.iterrows():
        cs, ps = prow['crid_scenario'], prow['pass_id']
        have = set(
            triples_df[(triples_df['crid_scenario']==cs) &
                       (triples_df['pass_id']==ps)]['ice_condition']
        )
        # existing ice rows (mark later as 1/2)
        for ic in have:
            full_rows.append({'crid_scenario': cs, 'pass_id': ps,
                              'ice_condition': ic, 'grouping_scheme': np.nan})
        # add missing ice row → synthetic 3
        for ic in ['ice-covered','ice-free']:
            if ic not in have:
                full_rows.append({'crid_scenario': cs, 'pass_id': ps,
                                  'ice_condition': ic, 'grouping_scheme': 3})

    base_df = pd.DataFrame(full_rows)

    # Add one "both" row per pair (grouping_scheme set to 1/2 next)
    both_rows = pairs_order[['crid_scenario','pass_id']].copy()
    both_rows['ice_condition'] = 'both'
    both_rows['grouping_scheme'] = np.nan
    base_df = pd.concat([base_df, both_rows], ignore_index=True)

    # Attach pair order for final sorting
    base_df = base_df.merge(pairs_order, on=['crid_scenario','pass_id'], how='left')

    # -------------------------
    # Mark grouping_scheme BEFORE fallback
    # -------------------------
    # Ice rows:
    cons_triples = df_cons[['crid_scenario','pass_id','ice_condition']].drop_duplicates()
    cons_triple_set = set(cons_triples.apply(tuple, axis=1))

    mask_ice = base_df['ice_condition'].isin(['ice-covered','ice-free'])
    # Set 1 if triple in conservative subset (do not overwrite synthetic=3)
    ice_triples = base_df.loc[mask_ice, ['crid_scenario','pass_id','ice_condition']].apply(tuple, axis=1)
    is_cons = ice_triples.isin(cons_triple_set)
    idx_cons = base_df.index[mask_ice].where(is_cons).dropna().astype(int)
    base_df.loc[idx_cons, 'grouping_scheme'] = base_df.loc[idx_cons, 'grouping_scheme'].fillna(1)

    # Any remaining NaN among ice rows (i.e., present in df but not in cons) → 2
    base_df.loc[mask_ice & base_df['grouping_scheme'].isna(), 'grouping_scheme'] = 2
    # synthetic 3 already set

    # "Both" rows:
    cons_pairs = df_cons[['crid_scenario','pass_id']].drop_duplicates()
    cons_pair_set = set(cons_pairs.apply(tuple, axis=1))
    mask_both = base_df['ice_condition'].eq('both')
    both_pairs = base_df.loc[mask_both, ['crid_scenario','pass_id']].apply(tuple, axis=1)
    base_df.loc[mask_both & both_pairs.isin(cons_pair_set), 'grouping_scheme'] = 1
    base_df.loc[mask_both & base_df['grouping_scheme'].isna(), 'grouping_scheme'] = 2

    # Add lake_id
    base_df.insert(0, 'lake_id', lake_id)

    # -------------------------
    # Threshold calibration from conservative subset
    # -------------------------
    def get_group_keys(metric_index):
        """Grouping keys for ice rows (may include ice_condition)."""
        keys = []
        if by_crid_scenario[metric_index]:
            keys.append('crid_scenario')
        if by_pass_id[metric_index]:
            keys.append('pass_id')
        if by_ice[metric_index]:
            keys.append('ice_condition')
        return keys if keys else ['global']

    def get_group_keys_both(metric_index):
        """Grouping keys for BOTH rows (ignore by_ice)."""
        keys = []
        if by_crid_scenario[metric_index]:
            keys.append('crid_scenario')
        if by_pass_id[metric_index]:
            keys.append('pass_id')
        return keys if keys else ['global']

    metric_specs = [
        ('wse_std','wse_std_thr_cal',0,'max'),
        ('wse_u','wse_u_thr_cal',1,'max'),
        ('xtrk_dist_abs','xtrk_dist_thr_cal',2,'min'),
    ]

    # 1) Ice rows calibration
    for metric_name, out_col, idx, agg_func in metric_specs:
        group_keys = get_group_keys(idx)
        tmp = df_cons.copy()
        if group_keys == ['global']:
            tmp['global'] = 'global'
        grouped = (tmp.groupby(group_keys, dropna=False)[metric_name]
                     .agg(agg_func)
                     .reset_index()
                     .rename(columns={metric_name: out_col}))
        base_df = base_df.merge(grouped, on=group_keys, how='left')

    # 2) BOTH rows calibration (ignore by_ice)
    for metric_name, out_col, idx, agg_func in metric_specs:
        group_keys_both = get_group_keys_both(idx)
        tmp = df_cons.copy()
        if group_keys_both == ['global']:
            tmp['global'] = 'global'
        grouped_both = (tmp.groupby(group_keys_both, dropna=False)[metric_name]
                          .agg(agg_func)
                          .reset_index()
                          .rename(columns={metric_name: f'__{out_col}_both__'}))
        # Merge and then assign to BOTH rows only
        if group_keys_both == ['global']:
            base_df['global'] = 'global'
            base_df = base_df.merge(grouped_both, on='global', how='left')
            base_df.drop(columns=['global'], inplace=True)
        else:
            base_df = base_df.merge(grouped_both, on=group_keys_both, how='left')

        src = f'__{out_col}_both__'
        base_df.loc[mask_both, out_col] = base_df.loc[mask_both, src]
        base_df.drop(columns=[src], inplace=True)

    # -------------------------
    # Fallback rules (fill remaining NaNs)
    # -------------------------
    def _fill_by_group(df_in, mask, key_cols, col, reducer):
        """Fill NaNs in df_in[col] for rows where mask is True, using group reducer over key_cols."""
        df = df_in.copy()  # avoid SettingWithCopyWarning
        non_na = df.dropna(subset=[col])
        if non_na.empty:
            return df
        pool = non_na.groupby(key_cols, dropna=False)[col].agg(reducer)
        idx = df.index[mask & df[col].isna()]
        for i in idx:
            keys = tuple(df.loc[i, key_cols].tolist())
            cand = pool.get(keys, np.nan)
            if pd.notna(cand):
                df.loc[i, col] = cand
        return df

    # A) ICE-condition-centric fallback for wse_std_thr_cal and wse_u_thr_cal (only ice rows)
    for col in ['wse_std_thr_cal', 'wse_u_thr_cal']:
        tgt = base_df['ice_condition'].isin(['ice-covered','ice-free'])
        # Level 1: same ice_condition + pass_id → max
        base_df = _fill_by_group(base_df, tgt, ['ice_condition','pass_id'], col, 'max')
        # Level 2: same ice_condition + crid_scenario → max
        base_df = _fill_by_group(base_df, tgt, ['ice_condition','crid_scenario'], col, 'max')
        # Level 3: same ice_condition → max
        base_df = _fill_by_group(base_df, tgt, ['ice_condition'], col, 'max')
        ## Level 4: hard default = max(bounds)
        #base_df.loc[tgt & base_df[col].isna(), col] = max(bounds) # Leave it as NaN for now

    # B) Pass-centric fallback for xtrk_dist_thr_cal (only ice rows)
    col = 'xtrk_dist_thr_cal'
    tgt = base_df['ice_condition'].isin(['ice-covered','ice-free'])
    # Level 1: same pass_id + ice_condition → min
    base_df = _fill_by_group(base_df, tgt, ['pass_id','ice_condition'], col, 'min')
    # Level 2: same pass_id + crid_scenario → min
    base_df = _fill_by_group(base_df, tgt, ['pass_id','crid_scenario'], col, 'min')
    # Level 3: same pass_id → min
    base_df = _fill_by_group(base_df, tgt, ['pass_id'], col, 'min')
    ## Level 4: hard default = min(bounds)
    #base_df.loc[tgt & base_df[col].isna(), col] = min(bounds) # Leave it as NaN for now

    # C) BOTH rows fallback (only among "both" rows)
    tgt_both = base_df['ice_condition'].eq('both')
    # Level 1: same pass_id → max/max/min
    for col, red in [('wse_std_thr_cal','max'), ('wse_u_thr_cal','max'), ('xtrk_dist_thr_cal','min')]:
        base_df = _fill_by_group(base_df, tgt_both, ['pass_id'], col, red)
    # Level 2: same crid_scenario → max/max/min
    for col, red in [('wse_std_thr_cal','max'), ('wse_u_thr_cal','max'), ('xtrk_dist_thr_cal','min')]:
        base_df = _fill_by_group(base_df, tgt_both, ['crid_scenario'], col, red)
    # Level 3: hard defaults for remaining NaNs: Leave it as NaN for now.

    # -------------------------
    # Finalization
    # -------------------------
    # Final ordering by pair sequence in df
    base_df = base_df.sort_values(['__pair_order__','ice_condition'], kind='stable').reset_index(drop=True)
    base_df.drop(columns=['__pair_order__'], inplace=True)

    # Final columns
    return base_df[['lake_id','crid_scenario','pass_id','ice_condition',
                    'wse_std_thr_cal','wse_u_thr_cal','xtrk_dist_thr_cal','grouping_scheme']]

def apply_heuristic_thresholds(
    df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    # Bound overrides for applied thresholds_df
    wse_std_threshold_bounds = [0, 3],
    wse_u_threshold_bounds   = [0, 0.5],
    xtrk_dist_threshold_bounds = [0, 75000],

    # Ice overrides for applied thresholds on ice-affected rows
    wse_std_ice_min: float = 3.0,
    wse_u_ice_min: float   = 0.5,

    # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
    # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
    rules_for_ice_free_data    = ['ice-free', 'ice-free', 'not apply'],
    rules_for_ice_covered_data = ['ice-free', 'ice-free', 'ice-covered'],

    # Optional threshold-traceability outputs
    return_threshold_details: bool = False
):
    """
    Apply calibrated heuristic thresholds to a LakeSP observation table.

    This function takes the threshold table produced by
    calibrate_heuristic_thresholds() and applies selected thresholds to each
    LakeSP observation. It returns only the observations that pass all active
    threshold checks.

    Overview
    --------
    Each LakeSP observation can be screened using up to three diagnostic
    variables:

        - wse_std:
            Within-lake spatial variability of WSE, in metres. Larger values
            may indicate mixed water/land pixels, false water detections, or
            spatially inconsistent WSE retrievals.

        - wse_u:
            Reported LakeSP WSE uncertainty, in metres. Larger values indicate
            larger algorithm-reported WSE uncertainty.

        - xtrk_dist:
            Cross-track distance from the lake polygon centroid to nadir, in
            metres. The absolute value, abs(xtrk_dist), is used. When active,
            this variable is applied as a lower threshold.

    For each diagnostic variable, the user can decide whether to apply a
    threshold calibrated from ice-free observations, ice-covered observations,
    pooled observations, or not to apply that variable at all. This allows the
    same calibrated threshold table to support different filtering strategies.

    Threshold-column convention
    ---------------------------
    The calibrated threshold table should use the following column names:

        - wse_std_thr_cal
        - wse_u_thr_cal
        - xtrk_dist_thr_cal

    These columns are the calibrated thresholds returned by 
    calibrate_heuristic_thresholds().

    Definitions of threshold stages
    -------------------------------
    For each diagnostic variable, a threshold can be described at three stages:

        *_thr_cal
            The calibrated threshold returned by calibrate_heuristic_thresholds()
            and selected for this observation from thresholds_df. The selection
            first matches the observation's crid_scenario and pass_id, then uses
            the metric-specific rule ("ice-free", "ice-covered", "both", or
            "not apply") to choose which ice-condition threshold row is used.
            If the metric is set to "not apply", this value is left as NaN.

        *_thr_bound
            The selected calibrated threshold after missing-value fallback and
            user-specified bounds are applied, but before any observation-level
            ice override. In the threshold-summary output, this is stored once
            for each threshold row that is actually selected by at least one
            observation. In the observation-level output, it is the bounded
            threshold selected for that particular observation.

        *_thr_app
            The final observation-level threshold actually used in the filtering
            condition. For wse_std and wse_u, this includes the ice override
            when ice_clim_f >= 1. For xtrk_dist, no ice override is applied, so
            *_thr_app equals *_thr_bound whenever xtrk_dist is active.

    Matching keys and inputs
    ------------------------
    df : pandas.DataFrame
        Input LakeSP observation table. It must contain at least:

            ['crid', 'pass_id', 'ice_clim_f',
             'wse_std', 'wse_u', 'xtrk_dist']

        Each input row represents one LakeSP observation. If an 'index_col'
        column is present, it is preserved and used as the preferred key in the
        optional observation-level threshold-details output.

    thresholds_df : pandas.DataFrame
        Threshold table returned by calibrate_heuristic_thresholds(). It should
        contain threshold rows for combinations of crid_scenario, pass_id, and
        ice_condition.

        Expected key columns:

            ['crid_scenario', 'pass_id', 'ice_condition']

        Expected calibrated-threshold columns:

            ['wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal']

        ice_condition values are:

            'ice-free'
                Thresholds calibrated from observations with ice_clim_f < 2.

            'ice-covered'
                Thresholds calibrated from observations with ice_clim_f >= 2.

            'both'
                Pooled thresholds calibrated without separating ice-free and
                ice-covered observations.

    Threshold-selection rules
    -------------------------
    The two rule lists determine which threshold source is selected for each
    observation. Each list has three entries corresponding to:

        [wse_std, wse_u, xtrk_dist]

    Allowed rule values are:

        'ice-free'
            Select the threshold calibrated from ice-free observations.

        'ice-covered'
            Select the threshold calibrated from ice-covered observations.

        'both'
            Select the pooled threshold calibrated from observations without
            distinguishing ice condition.

        'not apply'
            Do not use this diagnostic variable as a filtering criterion.

    rules_for_ice_free_data
        Rules used for observations whose row-level calibration ice condition
        is ice-free, defined by ice_clim_f < 2.

    rules_for_ice_covered_data
        Rules used for observations whose row-level calibration ice condition
        is ice-covered, defined by ice_clim_f >= 2.

    Example
    -------
        rules_for_ice_free_data    = ['ice-free', 'ice-free', 'not apply']
        rules_for_ice_covered_data = ['ice-free', 'ice-free', 'not apply']

    This means:

        - For ice-free observations:
            apply the ice-free wse_std threshold;
            apply the ice-free wse_u threshold;
            ignore xtrk_dist.

        - For ice-covered observations:
            also apply the ice-free wse_std threshold;
            also apply the ice-free wse_u threshold;
            ignore xtrk_dist.

    In this example, ice-covered observations are screened using thresholds
    calibrated from ice-free observations, but may still receive the
    observation-level ice override described below.

    A metric marked as 'not apply' does not gate an observation. For example,
    when xtrk_dist is set to 'not apply', an observation may still be retained
    even if abs(xtrk_dist) is small, provided the active wse_std and wse_u checks
    pass. Its xtrk_dist *_thr_app value is left as NaN in the optional
    observation-level threshold-details output.

    Bounds and missing-threshold handling
    -------------------------------------
    This function does not recalibrate thresholds. It uses thresholds_df as
    input and applies two safeguards before filtering:

        1. Missing-value fallback:
           If a selected threshold is NaN and the metric is active:
               - wse_std uses the upper bound as the fallback threshold.
               - wse_u uses the upper bound as the fallback threshold.
               - xtrk_dist uses the lower bound as the fallback threshold.

        2. Bound clipping:
           Thresholds are clipped to the user-provided minimum and maximum.

    These safeguards produce the bounded-threshold values:

        - wse_std_thr_bound
        - wse_u_thr_bound
        - xtrk_dist_thr_bound

    The bounded thresholds are attempt-specific because strict and lenient HALF
    attempts may use different bounds. They do not yet include the row-specific
    ice override.

    Example for wse_std
    -------------------
    If:

        wse_std_thr_cal = 4.2
        wse_std_threshold_bounds = [0, 3]

    then:

        wse_std_thr_bound = 3.0

    If:

        wse_std_thr_cal = NaN
        wse_std_threshold_bounds = [0, 3]

    then:

        wse_std_thr_bound = 3.0

    Ice override
    ------------
    For observations with possible or full ice influence, defined here as
    ice_clim_f >= 1, wse_std and wse_u thresholds can be relaxed to
    user-defined minimum values:

        wse_std_thr_app = max(wse_std_thr_bound, wse_std_ice_min)
        wse_u_thr_app   = max(wse_u_thr_bound,   wse_u_ice_min)

    This override is applied only when the corresponding metric is active. It
    is observation-specific and is therefore reported in the observation-level
    threshold-details output. No ice override is applied to xtrk_dist.

    Example for the ice override
    ----------------------------
    If:

        wse_std_thr_bound = 0.491
        wse_std_ice_min = 3.0

    then:

        ice_clim_f = 0  ->  wse_std_thr_app = 0.491
        ice_clim_f = 1  ->  wse_std_thr_app = 3.0
        ice_clim_f = 2  ->  wse_std_thr_app = 3.0

    This override intentionally relaxes the initial threshold screening under
    possible ice influence. Ice-affected observations often have elevated
    wse_std and wse_u values, and some lakes may have too few conservative
    ice-covered observations to produce stable ice-specific thresholds. The
    override preserves a subset of potentially valid ice-affected observations
    for subsequent low-pass filtering and temporal-consistency evaluation.

    Filtering logic
    ---------------
    For each observation, active thresholds are applied as follows:

        wse_std        <= wse_std_thr_app
        wse_u          <= wse_u_thr_app
        abs(xtrk_dist) >= xtrk_dist_thr_app

    All active checks must pass for the observation to be retained. Inactive
    checks are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        LakeSP observation table for one lake.

    thresholds_df : pandas.DataFrame
        Calibrated threshold table returned by calibrate_heuristic_thresholds().

    wse_std_threshold_bounds : list of float, default [0, 3]
        Minimum and maximum allowed values for the bounded wse_std threshold.
        Missing active wse_std thresholds use the upper bound as fallback.

    wse_u_threshold_bounds : list of float, default [0, 0.5]
        Minimum and maximum allowed values for the bounded wse_u threshold.
        Missing active wse_u thresholds use the upper bound as fallback.

    xtrk_dist_threshold_bounds : list of float, default [0, 75000]
        Minimum and maximum allowed values for the bounded abs(xtrk_dist)
        threshold. Missing active xtrk_dist thresholds use the lower bound as
        fallback.

    wse_std_ice_min : float, default 3.0
        Minimum applied wse_std threshold for observations with ice_clim_f >= 1,
        when wse_std is active.

    wse_u_ice_min : float, default 0.5
        Minimum applied wse_u threshold for observations with ice_clim_f >= 1,
        when wse_u is active.

    rules_for_ice_free_data : list of str
        Threshold-selection rules for observations with ice_clim_f < 2.

    rules_for_ice_covered_data : list of str
        Threshold-selection rules for observations with ice_clim_f >= 2.

    return_threshold_details : bool, default False
        If False, return only the filtered DataFrame, preserving the historical
        behavior of this function.

        If True, return:

            filtered_df, threshold_summary, observation_thresholds

    Returns
    -------
    pandas.DataFrame
        If return_threshold_details=False, returns the filtered subset of df
        containing only observations that pass all active threshold checks. The
        returned DataFrame preserves only the original columns from df.

    If return_threshold_details=True, returns a tuple:

        filtered_df : pandas.DataFrame
            Filtered subset of df containing only observations that pass all
            active threshold checks. Only the original columns from df are
            returned in this table.

        threshold_summary : pandas.DataFrame
            Copy of the calibrated threshold table with additional bounded
            columns (*_thr_bound). 
            A bounded value is populated only when that
            threshold row is selected by at least one input observation and the
            corresponding metric is active. The table does not contain
            observation-specific ice overrides.

        observation_thresholds : pandas.DataFrame
            Observation-level threshold information for all input rows evaluated by
            this apply_heuristic_thresholds() call, including both retained and 
            removed observations. When index_col is present, it is included as 
            the preferred key for joining back to the original LakeSP table.

            For each variable:
                *_thr_source
                    threshold source selected by the rule: ice-free, ice-covered,
                    both, or not apply.
                *_thr_cal
                    calibrated threshold selected for this observation.
                *_thr_bound
                    selected threshold after missing-value fallback and bounds.
                *_thr_app
                    final threshold actually applied to this observation, including
                    the ice override for wse_std and wse_u.

            For xtrk_dist, *_thr_app equals *_thr_bound when xtrk_dist is active,
            because no ice override is applied to xtrk_dist.
    
    Notes
    -----
    - The threshold-summary table uses *_thr_bound rather than *_thr_app because
      *_thr_app can vary among observations that select the same calibrated
      threshold row when the ice override is triggered.
    - The observation-level table is the appropriate place to store *_thr_app,
      because each row has its own ice_clim_f value and therefore its own final
      applied threshold.
    - xtrk_dist is calibrated and bounded for traceability, but it is not used
      by default in the recommended HALF configuration for pre-Version-D2
      LakeSP products.
    """

    # ---------------------------------------------------------------------
    # Quick guards & rule validation
    # ---------------------------------------------------------------------
    if df is None:
        df = pd.DataFrame()
    if thresholds_df is None:
        thresholds_df = pd.DataFrame()

    # Keep only original df columns for the filtered return value.
    original_columns = df.columns.tolist()

    def _normalize_threshold_columns(thr_in: pd.DataFrame) -> pd.DataFrame:
        """Validate and return a threshold table using the *_thr_cal schema."""
        thr = thr_in.copy()

        # Require the calibrated-threshold names returned by
        # calibrate_heuristic_thresholds(). Older threshold-column names are
        # intentionally not accepted in this release.
        required_columns = [
            'crid_scenario', 'pass_id', 'ice_condition',
            'wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal'
        ]
        if not thr.empty:
            missing_columns = [
                col for col in required_columns if col not in thr.columns
            ]
            if missing_columns:
                raise ValueError(
                    "thresholds_df must use the calibrated-threshold column "
                    "names returned by calibrate_heuristic_thresholds(): "
                    "wse_std_thr_cal, wse_u_thr_cal, and "
                    "xtrk_dist_thr_cal. "
                    f"Missing required columns: {missing_columns}."
                )

        # Ensure the threshold table has a stable schema for empty inputs and
        # for optional metadata columns.
        for col in [
            'lake_id', 'crid_scenario', 'pass_id', 'ice_condition',
            'wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal',
            'grouping_scheme'
        ]:
            if col not in thr.columns:
                thr[col] = np.nan
        return thr

    def _empty_threshold_summary(thr_in: pd.DataFrame) -> pd.DataFrame:
        """Return calibrated threshold rows with blank *_thr_bound columns."""
        thr = _normalize_threshold_columns(thr_in)
        keep_cols = [
            'lake_id', 'crid_scenario', 'pass_id', 'ice_condition',
            'wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal',
            'grouping_scheme'
        ]
        out = thr[keep_cols].copy()
        out['wse_std_thr_bound'] = np.nan
        out['wse_u_thr_bound'] = np.nan
        out['xtrk_dist_thr_bound'] = np.nan
        return out

    observation_detail_columns = [
        'lake_id', 'index_col', 'obs_id', 'time', 'datetime',
        'crid_scenario', 'pass_id', 'ice_clim_f', 'row_ice_condition',
        'wse_std_thr_source', 'wse_std_thr_cal', 'wse_std_thr_bound',
        'wse_std_thr_app',
        'wse_u_thr_source', 'wse_u_thr_cal', 'wse_u_thr_bound',
        'wse_u_thr_app',
        'xtrk_dist_thr_source', 'xtrk_dist_thr_cal',
        'xtrk_dist_thr_bound', 'xtrk_dist_thr_app'
    ]

    if thresholds_df.empty or df.empty:
        #print("[apply_heuristic_thresholds] ALERT: df and thresholds_df are empty. "
        #      "No rows can be validated. Returning empty DataFrame.")
        filtered_df = df.iloc[0:0].copy()  # preserve schema
        if return_threshold_details:
            return (
                filtered_df,
                _empty_threshold_summary(thresholds_df),
                pd.DataFrame(columns=observation_detail_columns),
            )
        return filtered_df

    # Make sure rule lists have no typo.
    def _ok_rule_list(lst):
        # Each list must be length-3 with per-metric entries,
        # each entry is one of the allowed strings:
        allowed = {'ice-free', 'ice-covered', 'both', 'not apply'}
        return (
            isinstance(lst, (list, tuple))
            and len(lst) == 3
            and all(x in allowed for x in lst)
        )

    if not _ok_rule_list(rules_for_ice_free_data):
        raise ValueError(
            "rules_for_ice_free_data must be a 3-item list of "
            "['ice-free'|'ice-covered'|'both'|'not apply']."
        )
    if not _ok_rule_list(rules_for_ice_covered_data):
        raise ValueError(
            "rules_for_ice_covered_data must be a 3-item list of "
            "['ice-free'|'ice-covered'|'both'|'not apply']."
        )

    df = df.copy()
    thr_original = _normalize_threshold_columns(thresholds_df)
    thr = thr_original.copy()

    # Apply missing-value fallback and bound clipping to the calibrated
    # threshold table. These values are *_thr_bound and do not yet include the
    # observation-specific ice override.
    thr['wse_std_thr_bound'] = (
        pd.to_numeric(thr['wse_std_thr_cal'], errors='coerce')
        .fillna(wse_std_threshold_bounds[1])    # replace NaN with upper bound
        .clip(*wse_std_threshold_bounds)        # clip values to [lower, upper]
    )

    thr['wse_u_thr_bound'] = (
        pd.to_numeric(thr['wse_u_thr_cal'], errors='coerce')
        .fillna(wse_u_threshold_bounds[1])      # replace NaN with upper bound
        .clip(*wse_u_threshold_bounds)          # clip values to [lower, upper]
    )

    thr['xtrk_dist_thr_bound'] = (
        pd.to_numeric(thr['xtrk_dist_thr_cal'], errors='coerce')
        .fillna(xtrk_dist_threshold_bounds[0])  # replace NaN with lower bound
        .clip(*xtrk_dist_threshold_bounds)      # clip values to [lower, upper]
    )

    # ---------------------------------------------------------------------
    # Compute crid_scenario & row ice_condition, to align with thresholds
    # ---------------------------------------------------------------------
    df['crid_scenario'] = df['crid'].apply(_crid_suffix2)

    # Row ice condition for choosing which ruleset to use. This follows the
    # calibration convention: ice_clim_f < 2 is ice-free; >= 2 is ice-covered.
    df['ice_condition'] = np.where(
        df['ice_clim_f'] >= 2,
        'ice-covered',
        'ice-free'
    )

    # We will merge by (crid_scenario, pass_id). To avoid dtype mismatches,
    # we create string-typed merge keys on both frames.
    df['_crid_scenario_str'] = df['crid_scenario'].astype(str)
    df['_pass_id_str']       = df['pass_id'].astype(str)

    # Helper: prepare a thresholds subtable for a specific ice_condition label
    def _prepare_thr(
        thr_in: pd.DataFrame,
        ice_label: str,
        suffix: str
    ) -> pd.DataFrame:
        """
        Extract calibrated and bounded thresholds for one ice_condition and
        rename them with suffix _ifree, _icov, or _both.
        """
        t = thr_in[thr_in['ice_condition'] == ice_label][[
            'crid_scenario', 'pass_id',
            'wse_std_thr_cal', 'wse_u_thr_cal', 'xtrk_dist_thr_cal',
            'wse_std_thr_bound', 'wse_u_thr_bound', 'xtrk_dist_thr_bound'
        ]].copy()
        t['_crid_scenario_str'] = t['crid_scenario'].astype(str)
        t['_pass_id_str']       = t['pass_id'].astype(str)

        # Drop the original key columns; keep only string keys and renamed thresholds.
        t = t.drop(columns=['crid_scenario', 'pass_id'])
        t = t.rename(columns={
            'wse_std_thr_cal':       f'wse_std_thr_cal{suffix}',
            'wse_u_thr_cal':         f'wse_u_thr_cal{suffix}',
            'xtrk_dist_thr_cal':     f'xtrk_dist_thr_cal{suffix}',
            'wse_std_thr_bound':     f'wse_std_thr_bound{suffix}',
            'wse_u_thr_bound':       f'wse_u_thr_bound{suffix}',
            'xtrk_dist_thr_bound':   f'xtrk_dist_thr_bound{suffix}',
        })
        return t

    # ---------------------------------------------------------------------
    # Bring in three sets of thresholds: ice-free, ice-covered, and both
    # ---------------------------------------------------------------------
    thr_ifree = _prepare_thr(thr, 'ice-free',    '_ifree')
    thr_icov  = _prepare_thr(thr, 'ice-covered', '_icov')
    thr_both  = _prepare_thr(thr, 'both',        '_both')

    # Merge all three onto df (left joins by string keys).
    df = df.merge(
        thr_ifree,
        on=['_crid_scenario_str', '_pass_id_str'],
        how='left'
    )
    df = df.merge(
        thr_icov,
        on=['_crid_scenario_str', '_pass_id_str'],
        how='left'
    )
    df = df.merge(
        thr_both,
        on=['_crid_scenario_str', '_pass_id_str'],
        how='left'
    )

    # ---------------------------------------------------------------------
    # Select per-metric calibrated and bounded thresholds based on rules and
    # row ice state.
    # ---------------------------------------------------------------------
    is_free = df['ice_condition'].eq('ice-free')  # ice_clim_f < 2

    def _select_threshold_per_metric(
        ifree_cal_col: str,
        icov_cal_col: str,
        both_cal_col: str,
        ifree_bound_col: str,
        icov_bound_col: str,
        both_bound_col: str,
        rule_ifree: str,
        rule_icov: str,
    ):
        """
        Construct selected calibrated/bounded threshold Series, a source label,
        and an 'apply' boolean Series for one metric.

        Returns
        -------
        cal : pandas.Series
            Selected calibrated threshold per row.
        bound : pandas.Series
            Selected threshold after fallback and bounds, before ice override.
        source : pandas.Series
            Selected threshold source: ice-free, ice-covered, both, or not apply.
        apply_flag : pandas.Series
            Whether this metric gates the row.
        """
        cal = pd.Series(np.nan, index=df.index, dtype='float64')
        bound = pd.Series(np.nan, index=df.index, dtype='float64')
        source = pd.Series('not apply', index=df.index, dtype='object')

        # Row-group: ice-free
        m = is_free
        if rule_ifree == 'ice-free':
            cal.loc[m] = df.loc[m, ifree_cal_col]
            bound.loc[m] = df.loc[m, ifree_bound_col]
            source.loc[m] = 'ice-free'
            apply_free = True
        elif rule_ifree == 'ice-covered':
            cal.loc[m] = df.loc[m, icov_cal_col]
            bound.loc[m] = df.loc[m, icov_bound_col]
            source.loc[m] = 'ice-covered'
            apply_free = True
        elif rule_ifree == 'both':
            cal.loc[m] = df.loc[m, both_cal_col]
            bound.loc[m] = df.loc[m, both_bound_col]
            source.loc[m] = 'both'
            apply_free = True
        else:  # 'not apply'
            apply_free = False

        # Row-group: ice-affected under the calibration convention (ice_clim_f >= 2)
        m = ~is_free
        if rule_icov == 'ice-free':
            cal.loc[m] = df.loc[m, ifree_cal_col]
            bound.loc[m] = df.loc[m, ifree_bound_col]
            source.loc[m] = 'ice-free'
            apply_cov = True
        elif rule_icov == 'ice-covered':
            cal.loc[m] = df.loc[m, icov_cal_col]
            bound.loc[m] = df.loc[m, icov_bound_col]
            source.loc[m] = 'ice-covered'
            apply_cov = True
        elif rule_icov == 'both':
            cal.loc[m] = df.loc[m, both_cal_col]
            bound.loc[m] = df.loc[m, both_bound_col]
            source.loc[m] = 'both'
            apply_cov = True
        else:  # 'not apply'
            apply_cov = False

        apply_flag = pd.Series(False, index=df.index)
        apply_flag.loc[ is_free] = apply_free
        apply_flag.loc[~is_free] = apply_cov
        return cal, bound, source, apply_flag  # thresholds, source, and whether applicable

    # Select calibrated/bounded thresholds, source labels, and apply flags for each metric.
    wse_std_cal, wse_std_bound, wse_std_source, apply_std = _select_threshold_per_metric(
        'wse_std_thr_cal_ifree', 'wse_std_thr_cal_icov', 'wse_std_thr_cal_both',
        'wse_std_thr_bound_ifree', 'wse_std_thr_bound_icov', 'wse_std_thr_bound_both',
        rules_for_ice_free_data[0], rules_for_ice_covered_data[0]
    )
    wse_u_cal, wse_u_bound, wse_u_source, apply_u = _select_threshold_per_metric(
        'wse_u_thr_cal_ifree', 'wse_u_thr_cal_icov', 'wse_u_thr_cal_both',
        'wse_u_thr_bound_ifree', 'wse_u_thr_bound_icov', 'wse_u_thr_bound_both',
        rules_for_ice_free_data[1], rules_for_ice_covered_data[1]
    )
    xtrk_cal, xtrk_bound, xtrk_source, apply_x = _select_threshold_per_metric(
        'xtrk_dist_thr_cal_ifree', 'xtrk_dist_thr_cal_icov', 'xtrk_dist_thr_cal_both',
        'xtrk_dist_thr_bound_ifree', 'xtrk_dist_thr_bound_icov', 'xtrk_dist_thr_bound_both',
        rules_for_ice_free_data[2], rules_for_ice_covered_data[2]
    )

    # ---------------------------------------------------------------------
    # No additional calibration fallback is performed here. However, if a
    # selected bounded threshold is still NaN and the metric applies, replace it
    # with the same bounds-based default used above:
    #      - wse_std, wse_u -> max(bounds)
    #      - xtrk_dist      -> min(bounds)
    # This also protects against an externally supplied threshold table that
    # lacks a matching threshold row for an observation.
    # ---------------------------------------------------------------------
    if apply_std.any():
        wse_std_bound.loc[apply_std & wse_std_bound.isna()] = max(
            wse_std_threshold_bounds
        )
    if apply_u.any():
        wse_u_bound.loc[apply_u & wse_u_bound.isna()] = max(
            wse_u_threshold_bounds
        )
    if apply_x.any():
        xtrk_bound.loc[apply_x & xtrk_bound.isna()] = min(
            xtrk_dist_threshold_bounds
        )

    # Store the final observation-level applied thresholds. They initially equal
    # the bounded values and are adjusted below only by the ice override.
    df['wse_std_thr_app']   = pd.to_numeric(wse_std_bound, errors='coerce')
    df['wse_u_thr_app']     = pd.to_numeric(wse_u_bound, errors='coerce')
    df['xtrk_dist_thr_app'] = pd.to_numeric(xtrk_bound, errors='coerce')

    # ---------------------------------------------------------------------
    # Ice override: For ice-affected rows, bump wse_std/u thresholds up to minima
    # (if the metric applies and a value exists), using prior conservative mask (>= 1).
    # ---------------------------------------------------------------------
    ice_cov_mask = df['ice_clim_f'] >= 1  # NOTE: used >=1 to be more conservative than >=2
    bump_std_mask = ice_cov_mask & apply_std & df['wse_std_thr_app'].notna()
    bump_u_mask   = ice_cov_mask & apply_u   & df['wse_u_thr_app'].notna()
    df.loc[
        bump_std_mask & (df['wse_std_thr_app'] < wse_std_ice_min),
        'wse_std_thr_app'
    ] = wse_std_ice_min
    df.loc[
        bump_u_mask & (df['wse_u_thr_app'] < wse_u_ice_min),
        'wse_u_thr_app'
    ] = wse_u_ice_min

    # ---------------------------------------------------------------------
    # Build optional threshold-detail outputs before temporary columns are
    # discarded from the filtered DataFrame.
    # ---------------------------------------------------------------------
    threshold_summary = None
    observation_thresholds = None

    if return_threshold_details:
        # Observation-level threshold information. Inactive metrics remain NaN
        # and have source = 'not apply'.
        observation_thresholds = pd.DataFrame(index=df.index)

        # Retain useful identifiers when they are available in the input table.
        for col in [
            'lake_id', 'index_col', 'obs_id', 'time', 'datetime',
            'pass_id', 'ice_clim_f'
        ]:
            if col in df.columns:
                observation_thresholds[col] = df[col]

        observation_thresholds['crid_scenario'] = df['crid_scenario']
        observation_thresholds['row_ice_condition'] = df['ice_condition']

        observation_thresholds['wse_std_thr_source'] = wse_std_source
        observation_thresholds['wse_std_thr_cal'] = pd.to_numeric(
            wse_std_cal,
            errors='coerce'
        )
        observation_thresholds['wse_std_thr_bound'] = pd.to_numeric(
            wse_std_bound,
            errors='coerce'
        )
        observation_thresholds['wse_std_thr_app'] = df['wse_std_thr_app']

        observation_thresholds['wse_u_thr_source'] = wse_u_source
        observation_thresholds['wse_u_thr_cal'] = pd.to_numeric(
            wse_u_cal,
            errors='coerce'
        )
        observation_thresholds['wse_u_thr_bound'] = pd.to_numeric(
            wse_u_bound,
            errors='coerce'
        )
        observation_thresholds['wse_u_thr_app'] = df['wse_u_thr_app']

        observation_thresholds['xtrk_dist_thr_source'] = xtrk_source
        observation_thresholds['xtrk_dist_thr_cal'] = pd.to_numeric(
            xtrk_cal,
            errors='coerce'
        )
        observation_thresholds['xtrk_dist_thr_bound'] = pd.to_numeric(
            xtrk_bound,
            errors='coerce'
        )
        observation_thresholds['xtrk_dist_thr_app'] = df['xtrk_dist_thr_app']

        # Ensure a stable column order while retaining only identifiers that
        # actually exist in the input DataFrame.
        ordered_cols = [
            c for c in observation_detail_columns
            if c in observation_thresholds.columns
        ]
        observation_thresholds = (
            observation_thresholds[ordered_cols]
            .reset_index(drop=True)
        )

        # Threshold-summary table. Keep every calibrated threshold row, but
        # populate *_thr_bound only for threshold rows selected by at least one
        # observation and only for metrics that are active.
        threshold_summary = _empty_threshold_summary(thr_original)
        threshold_summary['_crid_scenario_str'] = (
            threshold_summary['crid_scenario'].astype(str)
        )
        threshold_summary['_pass_id_str'] = (
            threshold_summary['pass_id'].astype(str)
        )

        def _populate_summary_bound(metric_prefix: str):
            """Populate one summary *_thr_bound column from selected rows."""
            source_col = f'{metric_prefix}_thr_source'
            bound_col = f'{metric_prefix}_thr_bound'

            tmp = observation_thresholds.loc[
                observation_thresholds[source_col].isin(
                    ['ice-free', 'ice-covered', 'both']
                ),
                ['crid_scenario', 'pass_id', source_col, bound_col]
            ].copy()

            if tmp.empty:
                return

            tmp['_crid_scenario_str'] = tmp['crid_scenario'].astype(str)
            tmp['_pass_id_str'] = tmp['pass_id'].astype(str)
            tmp['ice_condition'] = tmp[source_col]

            # All observations selecting the same threshold row have the same
            # bounded value; first() produces one value per summary row.
            grouped = (
                tmp.groupby(
                    ['_crid_scenario_str', '_pass_id_str', 'ice_condition'],
                    dropna=False
                )[bound_col]
                .first()
                .reset_index()
            )

            for _, row in grouped.iterrows():
                m = (
                    threshold_summary['_crid_scenario_str'].eq(
                        row['_crid_scenario_str']
                    )
                    & threshold_summary['_pass_id_str'].eq(
                        row['_pass_id_str']
                    )
                    & threshold_summary['ice_condition'].eq(
                        row['ice_condition']
                    )
                )
                threshold_summary.loc[m, bound_col] = row[bound_col]

        _populate_summary_bound('wse_std')
        _populate_summary_bound('wse_u')
        _populate_summary_bound('xtrk_dist')

        threshold_summary.drop(
            columns=['_crid_scenario_str', '_pass_id_str'],
            inplace=True,
            errors='ignore'
        )

    # ---------------------------------------------------------------------
    # Build final gating conditions
    # (coerce numeric for safety; NaNs in applied metrics -> row fails)
    # ---------------------------------------------------------------------
    for c in [
        'wse_std', 'wse_u', 'xtrk_dist',
        'wse_std_thr_app', 'wse_u_thr_app', 'xtrk_dist_thr_app'
    ]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Per-metric pass/fail (if a metric does not apply, it is treated as True).
    cond_std  = (~apply_std) | (df['wse_std'] <= df['wse_std_thr_app'])
    cond_u    = (~apply_u)   | (df['wse_u']   <= df['wse_u_thr_app'])
    cond_xtrk = (~apply_x)   | (
        df['xtrk_dist'].abs() >= df['xtrk_dist_thr_app']
    )

    keep_mask = cond_std & cond_u & cond_xtrk

    # ---------------------------------------------------------------------
    # Cleanup & return only original df columns for rows that pass. Optional
    # threshold details are returned separately and therefore do not alter the
    # historical filtered-DataFrame schema.
    # ---------------------------------------------------------------------
    filtered_df = df.loc[keep_mask, original_columns]

    if return_threshold_details:
        return filtered_df, threshold_summary, observation_thresholds
    return filtered_df

def filter_ice_outliers(df, remove_tukey_outliers, by_pass=True, by_crid_scenario=True,
                        multiplier=3, lower_q=0.25, upper_q=0.75, used_q='upper',
                        filter_by='both'):
    """
    Removes ice-covered/freeze-up observations where area_total or wse is a Tukey outlier
    relative to comparable ice-free observations, with optional grouping and directional outlier filtering.
    Output preserves the original row order and original columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing lake observations
        remove_tukey_outliers (function): Function that returns (filtered_df, lower_bound, upper_bound)
                                          for a specified column using Tukey’s method
        by_pass (bool): Whether to match ice-covered points with ice-free observations from the same pass_id
        by_crid_scenario (bool): Whether to match by CRID scenario (based on the last two digits in crid, e.g., C0, C2, and D0)
        multiplier (float): Multiplier for IQR in Tukey outlier filtering
        lower_q (float): Lower quantile to compute IQR
        upper_q (float): Upper quantile to compute IQR
        used_q (str): Which bound(s) to use: 'upper', 'lower', or 'both'
        filter_by (str): Which variable(s) to filter by: 'area', 'wse', or 'both'

    Returns:
        pd.DataFrame: Filtered DataFrame with ice-covered outliers removed,
                      preserving original row order and all original columns.

    Logic: 
        LakeSP observations tend to be more uncertain during freeze-up periods. This function provides
        an option to compare freeze-up observations (in area_total or WSE) with ice-free observations, and
        remove possible errors during the freeze-up period.

    Caution: 
        Some reservoirs can experience significant water level draw-downs during the freeze-up period.
        So caveats are needed when using this function to remove negative anomalies, which could be true signals.
        Therefore, we provide an option for the filtering direction ("used_q"), and we recommend using filtering
        positive anomalies as high lake water area or WSE during the freeze-up period are less likely and are probably errors.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    original_columns = df.columns.tolist()

    # Add a derived column: CRID scenario (used for grouping)
    df['crid_scenario'] = df['crid'].apply(_crid_suffix2)

    # Split into ice-covered (freeze-up or fully frozen) and ice-free observations
    df_ice = df[df['ice_clim_f'] > 1].copy()     # Typically more uncertain
    df_noice = df[df['ice_clim_f'] <= 1].copy()  # Used as reference group

    # If there are no ice-free records to use for comparison, return original
    if df_noice.empty:
        return df[original_columns]

    # Initialize a list to track indices of ice-covered rows to retain
    rows_to_keep = []

    # Iterate through each ice-covered observation
    for idx, row in df_ice.iterrows():
        # Start with all ice-free rows, then narrow down based on grouping options
        condition = pd.Series(True, index=df_noice.index)

        # Restrict to the same pass_id if requested
        if by_pass and row['pass_id'] in df_noice['pass_id'].values:
            condition &= (df_noice['pass_id'] == row['pass_id'])

        # Restrict to the same CRID scenario if requested
        if by_crid_scenario and row['crid_scenario'] in df_noice['crid_scenario'].values:
            condition &= (df_noice['crid_scenario'] == row['crid_scenario'])

        # Final reference group of comparable ice-free records
        reference_group = df_noice[condition]
        if reference_group.empty:
            # Fallback to all ice-free records if no match was found
            reference_group = df_noice

        # Drop NaNs for outlier calculation
        valid_area = reference_group['area_total'].dropna()
        valid_wse = reference_group['wse'].dropna()

        # If not enough data for Tukey bounds, keep the row by default
        if len(valid_area) < 2 and len(valid_wse) < 2:
            rows_to_keep.append(idx)
            continue

        # Initialize bounds
        area_lb, area_ub = None, None
        wse_lb, wse_ub = None, None

        # Calculate Tukey bounds for area_total if needed
        if filter_by in ['area', 'both'] and len(valid_area) >= 2:
            _, area_lb, area_ub = remove_tukey_outliers(reference_group, col='area_total',
                                                        multiplier=multiplier,
                                                        lower_q=lower_q, upper_q=upper_q)

        # Calculate Tukey bounds for wse if needed
        if filter_by in ['wse', 'both'] and len(valid_wse) >= 2:
            _, wse_lb, wse_ub = remove_tukey_outliers(reference_group, col='wse',
                                                      multiplier=multiplier,
                                                      lower_q=lower_q, upper_q=upper_q)

        # Retrieve the current observation's area and wse
        val_area = row['area_total']
        val_wse = row['wse']

        # Evaluate whether the current row is within Tukey bounds for area_total
        is_area_ok = True
        if filter_by in ['area', 'both'] and pd.notna(val_area) and area_lb is not None:
            if used_q == 'both':
                is_area_ok = area_lb <= val_area <= area_ub
            elif used_q == 'upper':
                is_area_ok = val_area <= area_ub
            elif used_q == 'lower':
                is_area_ok = val_area >= area_lb

        # Evaluate whether the current row is within Tukey bounds for wse
        is_wse_ok = True
        if filter_by in ['wse', 'both'] and pd.notna(val_wse) and wse_lb is not None:
            if used_q == 'both':
                is_wse_ok = wse_lb <= val_wse <= wse_ub
            elif used_q == 'upper':
                is_wse_ok = val_wse <= wse_ub
            elif used_q == 'lower':
                is_wse_ok = val_wse >= wse_lb

        # Keep the row only if it passes the outlier check
        if is_area_ok and is_wse_ok:
            rows_to_keep.append(idx)

    # Recombine filtered ice-covered rows with original ice-free rows
    df_ice_filtered = df_ice.loc[rows_to_keep]
    df_combined = pd.concat([df_ice_filtered, df_noice])

    # Ensure original row order and return only original columns
    common = df.index.intersection(df_combined.index, sort=False)  # keep df’s order
    return df_combined.loc[common, original_columns]

def convert_to_daily_series(
    df, gauge_df,
    time_col='datetime',
    gauge_time_col='gauge_datetime',
    wse_col='wse',
    wse_filtered_col='wse_adjusted',
    gauge_wse_col='gauge_wse',
    interp_method='linear',
    major_gap_days=90  # threshold (in days) for “major gap” in original gauge data
):
    """
    Compute daily-interpolated WSE time series from SWOT (raw and adjusted) and
    gauge data over their overlapping date range.

    Over large gaps in the ORIGINAL gauge series (consecutive gap >= major_gap_days),
    the interior dates of those gaps are EXCLUDED from the returned daily series, so
    interpolation will not bridge across major gauge gaps.

    Note: The overlapping time range is determined based on interpolated wse_col (not wse_filtered_col)
    and the original gauge_wse_col. If wse_filtered_col is empty, the corresponding output
    will be NaN, but the function can still return valid unfiltered and gauge outputs.

    Returns NaN for all outputs if either df or gauge_df is empty.

    Updated: 08/30/2025 from "compute_daily_variability"
    """

    import numpy as np
    import pandas as pd

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

    # Compute daily means
    wse_daily = df.groupby('date')[wse_col].mean()
    gauge_daily = gauge_df.groupby('date')[gauge_wse_col].mean()

    # Optional: check for filtered column existence
    wse_filtered_daily = pd.Series(dtype='float64')
    if wse_filtered_col in df.columns:
        wse_filtered_daily = df.groupby('date')[wse_filtered_col].mean()

    # Ensure datetime index
    wse_daily.index = pd.to_datetime(wse_daily.index)
    gauge_daily.index = pd.to_datetime(gauge_daily.index)
    wse_filtered_daily.index = pd.to_datetime(wse_filtered_daily.index)

    # Interpolate wse_daily and wse_filtered_daily within the full range of wse_daily
    if wse_daily.empty:
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    full_range = pd.date_range(start=wse_daily.index.min(), end=wse_daily.index.max(), freq='D')

    def safe_interp(series, full_index):
        return (series.reindex(full_index)
                      .interpolate(method=interp_method, limit_direction='both')
                      .bfill()
                      .ffill()) #Extrapolates flatly using the edge values
        # Flat edge extrapolation (bfill/ffill) is safer for lake WSE unless
        # strong justification exists for other trends.

    wse_interp_full = safe_interp(wse_daily, full_range)
    wse_filtered_interp_full = safe_interp(wse_filtered_daily, full_range) if not wse_filtered_daily.empty else np.nan

    # Now determine overlap between interpolated wse and original gauge_daily
    if gauge_daily.empty or wse_interp_full.empty:
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    start_date = max(wse_interp_full.index.min(), gauge_daily.index.min())
    end_date = min(wse_interp_full.index.max(), gauge_daily.index.max())

    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    # Base overlapping date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # pd.date_range(...) always returns a DatetimeIndex

    # Exclude interior days of major gaps in the ORIGINAL gauge data
    # Identify consecutive gaps >= major_gap_days between *observed* gauge dates.
    gauge_dates = gauge_daily.sort_index().index
    # Since gauge_daily is a Series indexed by dates (daily means of the gauge),
    # we use .sort_index() to ensure those dates are in chronological (ascending) order,
    # and .index then extracts just the DatetimeIndex (the list of observation dates).
    if len(gauge_dates) >= 2:
        # Measure gaps between consecutive observed dates
        diffs = gauge_dates.to_series().diff()  # first is NaT

        # Find the ends of big gaps
        # Indices i where gap between gauge_dates[i-1] and gauge_dates[i] is large
        large_gap_idx = diffs[diffs >= pd.Timedelta(days=major_gap_days)].index

        # Build a boolean mask over the base date_range, then drop large-gap interiors
        included = pd.Series(True, index=date_range)
        # Loop through each large gap
        for gap_end in large_gap_idx: # gap_end is the later observed date in a large gap.
            # previous observed date:
            prev_date = gauge_dates[gauge_dates.get_loc(gap_end) - 1] #observed date immediately before the gap.
            next_date = gap_end
            # So the gap is prev_date → next_date (say 2020-01-01 → 2020-04-15).

            # Compute the interior days of the gap
            # Exclude the interior days only (keep endpoints where observations exist)
            gap_start_interior = prev_date + pd.Timedelta(days=1)
            gap_end_interior = next_date - pd.Timedelta(days=1)
            if gap_start_interior <= gap_end_interior:
                # Mark those interior days as False (excluded)
                # Slice is safe even if out of bounds; pandas aligns by index labels
                included.loc[gap_start_interior:gap_end_interior] = False

        # Apply the mask to produce a filtered date_range that has major gaps removed
        kept_dates = included.index[included.values]
    else:
        kept_dates = date_range

    # If everything is excluded, return NaNs
    if len(kept_dates) == 0:
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    # Interpolate gauge onto the kept dates only (won’t bridge removed gaps)
    gauge_interp = safe_interp(gauge_daily, kept_dates)
    # Slice interpolated WSE and filtered WSE into the same kept dates
    wse_interp = wse_interp_full.reindex(kept_dates)
    if isinstance(wse_filtered_interp_full, pd.Series):
        wse_filtered_interp = wse_filtered_interp_full.reindex(kept_dates)
    else:
        wse_filtered_interp = np.nan

    return {
        'daily_wse': wse_interp,
        'daily_wse_filtered': wse_filtered_interp,
        'daily_gauge': gauge_interp
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
    data = data.sort_values(time_col, kind='mergesort')

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
    data = data.sort_values(time_col, kind='mergesort')

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
        interpolated_values = np.interp(x_regular, x_orig, y_orig) #flat extrapolation.
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
                return np.interp(x_eval, x_regular, smoothed) #flat extrapolation
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

    if pywt is None:
        raise ImportError(
            "filter_wavelet() requires PyWavelets. Install it with `pip install PyWavelets`."
        )

    # Prepare and sort data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col, kind='mergesort')

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
    data = data.sort_values(time_col, kind='mergesort')

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
    data = data.sort_values(time_col, kind='mergesort')

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
    data = data.sort_values(time_col, kind='mergesort')

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

    if KalmanFilter is None:
        raise ImportError(
            "filter_kalman() requires pykalman. Install it with `pip install pykalman`."
        )

    # Prepare and sort data
    data = data.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col, kind='mergesort')

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

def drop_eval_in_apply_gaps(
    df_eval,
    df_apply,
    max_temporal_gap,
    datetime_col,
):
    """
    Remove rows from df_eval whose timestamps fall inside the large gaps
    between consecutive timestamps in df_apply.

    A "large gap" is defined as any interval between two consecutive, unique,
    non-null df_apply[datetime_col] values whose length exceeds
    max_temporal_gap days. For each df_eval timestamp, we find the previous
    and next df_apply timestamps via vectorized as-of merges; rows that lie
    inside any large gap are dropped.

    Parameters
    ----------
    df_eval : pandas.DataFrame
        The evaluation dataframe that you want to filter. Must contain a
        datetime-like column named by `datetime_col`. May include rows that
        are not present in `df_apply`. Original row order and index are preserved.
    df_apply : pandas.DataFrame
        A dataframe (often a subset of `df_eval`) used to define gaps.
        Must contain the same datetime-like column. Only unique, non-null
        times in `df_apply` are used to form gaps.
    max_temporal_gap : int
        Gap threshold in **days**. Any consecutive pair of `df_apply` times
        with a separation strictly greater than this threshold defines a
        “large gap.” Example: 90 means gaps > 90 days.
    datetime_col : str, default "datetime"
        Name of the timestamp column in both dataframes.
        The column should be timezone-consistent across both frames.

    Returns
    -------
    filtered : pandas.DataFrame
        `df_eval` with rows removed that fall in any large `df_apply` gap.
        Preserves dtypes, original order, and original index.

    Notes
    -----
    - If `df_apply` has fewer than two unique, non-null timestamps, no gaps
      can be formed → nothing is dropped.
    - `NaT` values in `df_eval[datetime_col]` are *kept* (they cannot be placed
      inside a gap).
    - Duplicates in `df_apply[datetime_col]` are ignored when forming gaps.
    - Timezone handling: make sure both columns are either tz-naive or share
      the same timezone. Mixed tz-naive/aware data will error in merges.

    Complexity
    ----------
    O(N log N + M log M) for sorting `N = len(df_eval)` and `M = len(df_apply)`,
    plus O(N) merging, all vectorized.

    Example
    -------
    >>> # df_apply times: [Jan 1, Jan 10, Mar 20] → gap Jan10→Mar20 is ~70 days (>30)
    >>> # Any df_eval times strictly between Jan 10 and Mar 20 will be removed.
    >>> filtered = drop_eval_in_apply_gaps(df_eval, df_apply, max_temporal_gap=30)
    """
    # Construct the threshold as a Timedelta in days
    thr = pd.Timedelta(days=max_temporal_gap)

    # Build a sorted view of eval times, KEEPING original index via 'orig_idx'
    # We exclude NaT here for the matching step only; NaT rows cannot be inside a gap,
    # and we will keep them by default (i.e., they won't be dropped).
    eval_times = (
        df_eval[[datetime_col]]
        .loc[df_eval[datetime_col].notna()]
        .sort_values(datetime_col)
        .reset_index()
        .rename(columns={"index": "orig_idx", datetime_col: "t"})
    )

    # Build a sorted, unique list of apply times; these define the gap endpoints
    apply_times = (
        df_apply[[datetime_col]]
        .loc[df_apply[datetime_col].notna()]
        .drop_duplicates()
        .sort_values(datetime_col)
    )

    # If there are fewer than 2 apply times, there are no gaps → return unchanged
    if apply_times.shape[0] < 2 or eval_times.shape[0] == 0:
        return df_eval

    # Prepare two copies of apply times with distinct column names
    ap_prev = apply_times.rename(columns={datetime_col: "a"})  # for "previous" merge
    ap_next = apply_times.rename(columns={datetime_col: "b"})  # for "next" merge

    # For each eval timestamp t, find the previous (<= t) apply time 'a'
    prev = pd.merge_asof(
        eval_times, ap_prev, left_on="t", right_on="a", direction="backward"
    )

    # For each eval timestamp t, find the next (>= t) apply time 'b'
    nxt = pd.merge_asof(
        eval_times, ap_next, left_on="t", right_on="b", direction="forward"
    )

    # Join to align previous and next apply times for each eval row
    tmp = prev[["orig_idx", "t", "a"]].merge(nxt[["orig_idx", "b"]], on="orig_idx", how="inner")
    tmp = tmp.rename(columns={"a": "prev_time", "b": "next_time"})

    # Define "inside a large gap" for each eval timestamp row:
    # 1) Both neighbors exist (not at the ends of the apply timeline),
    # 2) The gap between them exceeds the threshold,
    # 3) The eval time lies inside that interval (strictly within, excluding edge).
    has_neighbors = tmp["prev_time"].notna() & tmp["next_time"].notna()
    gap_is_large = (tmp["next_time"] - tmp["prev_time"]) > thr
    in_gap = (
            has_neighbors
            & gap_is_large
            & (tmp["t"] > tmp["prev_time"])
            & (tmp["t"] < tmp["next_time"])
    )

    # Map identified eval rows back to original df_eval index
    to_drop_idx = tmp.loc[in_gap, "orig_idx"]

    # Build a keep-mask aligned to df_eval.index (True = keep, False = drop)
    keep_mask = pd.Series(True, index=df_eval.index)
    if not to_drop_idx.empty:
        keep_mask.loc[to_drop_idx.values] = False

    filtered = df_eval.loc[keep_mask]

    return filtered

def apply_customized_filter(
    df_eval,
    df_heuristic_thresholds,

    # Bound overrides for applied thresholds_df
    wse_std_threshold_bounds = [0, 3],
    wse_u_threshold_bounds   = [0, 0.5],
    xtrk_dist_threshold_bounds = [0, 75000],

    # Ice overrides for applied thresholds on ice-affected rows
    wse_std_ice_min=3,
    wse_u_ice_min=0.1,

    allow_major_gap = 'no', # 'yes' or 'no'; controls whether major gaps are allowed in the retained series.
    max_temporal_gap = 90, # Maximum temporal gap (days) for filtering
    min_temporal_range = 365, # Minimum temporal range (days) for filtering

    # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
    # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
    rules_for_ice_free_data=['ice-free', 'ice-free', 'not apply'],
    rules_for_ice_covered_data=['ice-free', 'ice-free', 'not apply'],

    gauge_df = None, # enter gauge_df; None if no gauge data is available.
    plot_period = ["2023-07-21T00:00:00Z", "2025-07-01T00:00:00Z"], # Start and end time for diagnostic plotting.

    apply_low_pass_filter = 'yes', #'yes' strongly recommended
    evaluating_at_full_data = 'no', #'no' recommended
    r2_filter = 'yes', #'yes' recommended
    filter_type = 'savgol', #lowess, wavelet, savgol, kalman, spline, median, hampel.
    z_score_thresholds = [2.576, 3.5], #z-score thresholds for 1st and 2nd rounds of low-pass filters, respectively.
                                       #2.576(99% for two tails), 2.807(99.5%), 2.967(99.7%), 3.291(99.9%), 3.5(99.95%)
    maximum_residual_spreads = [0.08, 0.06], #max residual spreads for 1st and 2nd rounds of low-pass filters, respectively.
    show_filtering_evolution = 'no', #for visualization only; caution: 'yes' may load many figures at the end of the script execution.
):   
    """
    Apply the full HALF filtering workflow to one LakeSP WSE time series.
    
    
    Procedure
    ---------    
    1. Apply calibrated heuristic thresholds to obtain an initial heuristic baseline.
    2. Fit a configurable low-pass filter to the baseline.
    3. Evaluate residuals against the selected evaluation set.
    4. Iteratively remove outliers until convergence criteria are met.
    5. Optionally run a second, more permissive filtering round to reduce over-rejection.
    
    
    Parameters
    ---------
    df_eval : pandas.DataFrame
        Initial LakeSP time series for one lake.
    
    df_heuristic_thresholds : pandas.DataFrame
        Calibrated heuristic thresholds returned by calibrate_heuristic_thresholds().
    
    wse_std_threshold_bounds : list
        [min, max] bounds applied to the wse_std threshold.
    
    wse_u_threshold_bounds : list
        [min, max] bounds applied to the wse_u threshold.
    
    xtrk_dist_threshold_bounds : list
        [min, max] bounds applied to the abs(xtrk_dist) threshold.
    
    wse_std_ice_min, wse_u_ice_min : float
        Minimum effective thresholds for wse_std and wse_u under possible ice
        influence. These values relax the initial threshold screening for
        ice-affected observations.
    
    allow_major_gap : {'yes', 'no'}
        Controls how major temporal gaps in the heuristic baseline are handled.
        'no':
            The filtering attempt fails if the heuristic baseline contains a gap
            longer than max_temporal_gap or does not span min_temporal_range.
        'yes':
            Major gaps in the heuristic baseline are allowed, but candidate
            observations falling inside those large baseline gaps are excluded from
            residual-based outlier testing.
    
    max_temporal_gap : int or float
        Major-gap threshold in days. When allow_major_gap='no', gaps longer than
        this value cause the filtering attempt to fail. 
        When allow_major_gap='yes', this value defines the baseline gaps inside 
        which candidate observations are excluded from residual testing.
    
    min_temporal_range : int or float
        Minimum required temporal span, in days, for the baseline time series. 
        This value is also used with max_temporal_gap to determine the minimum 
        number of observations needed for reliable filtering.
    
    rules_for_ice_free_data, rules_for_ice_covered_data : list of str
        Per-metric threshold-application rules for [wse_std, wse_u, xtrk_dist].
        rules_for_ice_free_data:
            Rules used when an observation is ice-free, based on ice_clim_f < 2.
        rules_for_ice_covered_data:
            Rules used when an observation is ice-covered, based on ice_clim_f >= 2.
    
    gauge_df : pandas.DataFrame or None
        Gauge time series used only for optional diagnostic plotting when
        show_filtering_evolution='yes'. 
        It does not affect the filtering result.
        If gauge time series is not available, just enter None. 
    
    plot_period : list
        Start and end times used for diagnostic plotting.    
        plot_period[0]: start time, formatted as yyyy-mm-ddThh:mm:ssZ
        plot_period[1]: end time, formatted as yyyy-mm-ddThh:mm:ssZ
        
    apply_low_pass_filter : {'yes', 'no'}
        'yes': 
            run both heuristic-baseline filtering and iterative low-pass
            residual filtering.
        'no': 
            return only the heuristic-baseline subset.
    
    evaluating_at_full_data : {'yes', 'no'}
        Used only when apply_low_pass_filter='yes'.
        'yes': 
            evaluate residual outlier removal against the full LakeSP record.
        'no': 
            evaluate residual outlier removal only against the selected
            candidate observations.
    
    r2_filter : {'yes', 'no'}
        Used only when apply_low_pass_filter='yes'.
        'yes':
            run an optional second filtering round after recovering selected
            high-quality observations.
        'no': 
            use only the first low-pass filtering round.
    
    filter_type : str
        Low-pass filter type. Supported options include:
            'lowess', 'wavelet', 'savgol', 'kalman', 'spline', 'median', and 'hampel'.
    
    z_score_thresholds : list
        Z-score thresholds for residual-based outlier removal.
        z_score_thresholds[0]: round-1 threshold, generally more aggressive.
        z_score_thresholds[1]: round-2 threshold, generally more permissive.
        
    maximum_residual_spreads : list
        Relative residual-spread tolerances for residual-based outlier removal.
        A residual whose magnitude is small relative to the WSE range can be
        retained even if its z-score is large.
        maximum_residual_spreads[0]: round-1 tolerance.
        maximum_residual_spreads[1]: round-2 tolerance.
        
    show_filtering_evolution : {'yes', 'no'}
        'yes': 
            generate diagnostic plots showing how filtering evolves through iterations.
        'no': 
            do not generate iteration-level diagnostic plots.
        Use 'yes' only for debugging or visual inspection, because it can generate
        many figures during batch processing.
    
    
    Returns
    ---------
    df_eval : pandas.DataFrame
        Filtered LakeSP time series. This may be empty if filtering fails.
    
    [n_while, n_while_r2] : list of int
        Iteration counts for round-1 and round-2 filtering.
        
        n_while:
            Number of iterations for round-1 low-pass filtering.
        n_while_r2:
            Number of iterations for round-2 low-pass filtering.
    
        For both values, the following status codes may occur:
            -9: original LakeSP input is empty; filtering is not applicable.
            -2: this filtering round was disabled or not applicable.
            -1: filtering started but was abandoned.
            0:  filtering did not complete successfully, often because the 
                candidate time series became empty or insufficient.
            >0: number of completed filtering iterations.
        
    filter_status : str
        Filtering outcome. Possible values are:
            
        'no data':
            No valid LakeSP observations were available.
    
        'heuristic baseline':
            Low-pass filtering was turned off, and only the heuristic-baseline
            subset was returned.
    
        'fail':
            The filtering attempt failed to produce a valid retained time series.
    
        'success':
            The filtering attempt produced a valid retained time series.

    threshold_summary, observation_thresholds : pandas.DataFrame
        The summary table contains calibrated and bounded threshold rows; 
        the observation table contains the final row-level thresholds actually applied.
    """
    
    # In case the input time series DataFrame is empty.
    if df_eval.empty:
        return df_eval.copy(), [-9, -9], 'no data', pd.DataFrame(), pd.DataFrame()


    # Freeze the initial df_eval to df
    df = df_eval.copy()

    # Initialize df_eval for filter update (safety measure applied)
    df_eval = df_eval.copy() # This will be updated through filtering.
    start_time = plot_period[0]
    end_time = plot_period[1]
    
    # First return threshold-detail outputs: 
    # Threshold-detail outputs document the heuristic-threshold application for this
    # HALF attempt. They are computed from the original input observations and the
    # attempt-specific bounds/rules. Later low-pass filtering, high-quality recovery,
    # and optional round-2 filtering may change the final retained df_eval, but they
    # do not change these threshold-detail tables because the same threshold settings
    # apply throughout the attempt.
    _, threshold_summary, observation_thresholds = apply_heuristic_thresholds(
        df,
        df_heuristic_thresholds,
        wse_std_threshold_bounds=wse_std_threshold_bounds,
        wse_u_threshold_bounds=wse_u_threshold_bounds,
        xtrk_dist_threshold_bounds=xtrk_dist_threshold_bounds,
        wse_std_ice_min=wse_std_ice_min,
        wse_u_ice_min=wse_u_ice_min,
        rules_for_ice_free_data=rules_for_ice_free_data,
        rules_for_ice_covered_data=rules_for_ice_covered_data,
        return_threshold_details=True,
    )
    # Note: By default, apply_heuristic_thresholds() returns only the filtered DataFrame.
    # The optional threshold-detail outputs are returned only when
    # return_threshold_details=True is explicitly passed.
    

    # By default: turn off n_while and n_while_r2 (-2), if apply_low_pass_filter == 'no'.
    n_while    = -2
    n_while_r2 = -2
    # Check if we would like to execute both baseline filtering (Step 2.1) and low-pass filtering (Step 2.2) or just Step 1
    if apply_low_pass_filter == 'no':
        # Apply heuristic thresholds to generate the heuristic baseline.
        df_eval = apply_heuristic_thresholds(df_eval, df_heuristic_thresholds,
                                             wse_std_threshold_bounds = wse_std_threshold_bounds,
                                             wse_u_threshold_bounds   = wse_u_threshold_bounds,
                                             xtrk_dist_threshold_bounds = xtrk_dist_threshold_bounds,
                                             wse_std_ice_min = wse_std_ice_min, wse_u_ice_min = wse_u_ice_min,
                                             rules_for_ice_free_data   = rules_for_ice_free_data, #(per-metric: [wse_std, wse_u, xtrk_dist])
                                             rules_for_ice_covered_data= rules_for_ice_covered_data)
        # Note: in the built-in function, wse_std/u threshold for freeze-up/ice-covered period is relaxed to increase data availability.
        # Based on our testing, this more lenient condition for ice-covered periods seems necessary.
        # Also see function apply_heuristic_thresholds for more details.

        return df_eval, [n_while, n_while_r2], 'heuristic baseline', threshold_summary, observation_thresholds

    else:
        # Execute both steps
        # If preferred, first constrain df_eval to the heuristic baseline before executing the low-pass filtering.
        if evaluating_at_full_data == 'no': # Evaluate outlier removal (z-score clipping) only on the selected heuristic baseline.
            # Apply heuristic thresholds to generate the heuristic baseline.
            df_eval = apply_heuristic_thresholds(df_eval, df_heuristic_thresholds,
                                                 wse_std_threshold_bounds = wse_std_threshold_bounds,
                                                 wse_u_threshold_bounds   = wse_u_threshold_bounds,
                                                 xtrk_dist_threshold_bounds = xtrk_dist_threshold_bounds,
                                                 wse_std_ice_min = wse_std_ice_min, wse_u_ice_min = wse_u_ice_min,
                                                 rules_for_ice_free_data   = rules_for_ice_free_data, #(per-metric: [wse_std, wse_u, xtrk_dist])
                                                 rules_for_ice_covered_data= rules_for_ice_covered_data)
        # Otherwise, if evaluating_at_full_data == 'yes', evaluate z-score cliping on the full df_eval data.

        """
        Start round-1 (mandatory) low-pass filtering: results will be stored in df_eval (a selected subset of LakeSP after filtering)
        To avoid confusion:
            "Filter application" refers to applying the chosen filter method to generate a smoothing curve. This is done on df_apply (baseline).
            "Filter evaluation" refers to using the smoothing curve as the reference for z-score clipping. This is done on df_eval.
        This "while" loop:
            - Starts by selecting high-quality LakeSP baseline (df_apply) from df_eval for filter application (i.e., generating smoothing curve)
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
        n_while = 0  # Initialize r1 loop/iteration times (turned on)
        initial_length = len(df_eval) # In case this is zero (after applying first baseline subsetting), the "while" statement will not run.
        updated_length = 0 # Initialize the length of the updated df_eval
        minimum_data_n = float(min_temporal_range/max_temporal_gap)+1 # the minimum number of data points considered to be acceptable.
        lowess_QA_check = 'check' # Initialize a QA check for the lowess filter. This is only relevant if filter_type is set to 'lowess'.
        
        while (updated_length < initial_length) and (n_while < 40): # All conditions must be satisfied
            initial_length = len(df_eval) # Note: df_eval is updated per iteration.

            # Apply heuristic thresholds to generate the "heuristic baseline" (i.e., high-quality observations for filter application).
            df_apply = apply_heuristic_thresholds(df_eval, df_heuristic_thresholds,
                                                  wse_std_threshold_bounds = wse_std_threshold_bounds,
                                                  wse_u_threshold_bounds   = wse_u_threshold_bounds,
                                                  xtrk_dist_threshold_bounds = xtrk_dist_threshold_bounds,
                                                  wse_std_ice_min = wse_std_ice_min, wse_u_ice_min = wse_u_ice_min,
                                                  rules_for_ice_free_data   = rules_for_ice_free_data, # per-metric: [wse_std, wse_u, xtrk_dist]
                                                  rules_for_ice_covered_data= rules_for_ice_covered_data)
            
            # Remove bad crossover calibration, although this is redundant for PIC2 and PID0 as quality_f < 3 precludes xovr_cal_q = 2 (see bitwise definition)
            df_apply = df_apply[df_apply['xovr_cal_q'] < 2]
            # Remove bad observations flagged in PIC2 and PID0: specular_rining_bad, xovr_cal_bad, and low_coh_bad.
            df_apply = df_apply[df_apply['quality_f'] < 3]

            # Truncate df_eval to the same time range of df_apply to avoid extrapolation
            tmin = df_apply['time'].min()
            tmax = df_apply['time'].max()
            df_eval = df_eval.loc[df_eval['time'].between(tmin, tmax, inclusive='both')]

            # Check whether the current filtering iteration has enough temporal support.
            # - df_apply is the heuristic baseline used to fit the low-pass curve.
            # - df_eval is the candidate observation set evaluated against that curve.
            #            
            # This lake is abandoned if any of the following situations occurs.
            #
            # If the heuristic baseline contains too few observations to support a 
            # reliable smoothing curve.
            #
            # If allow_major_gap == "no", strict mode:
            # - df_apply must not contain any temporal gap > max_temporal_gap 
            #   (e.g., 3-4 months or a hydrological season).
            # - df_apply must span at least min_temporal_range days
            #   (e.g., 1 year)
            # - If either condition fails, the filtering attempt is abandoned.
            #
            # If allow_major_gap == "yes", lenient mode:
            # - Large gaps in df_apply do not immediately fail the lake.
            # - Instead, df_eval observations that fall inside those large df_apply gaps
            #   are removed before residual-based outlier testing.
            # - This avoids evaluating observations against a smoothing curve that is
            #   poorly constrained across unsupported temporal gaps.
            # - The no-large-gap and minimum-span requirements are not enforced in this
            #   lenient mode, but the candidate series must still contain enough
            #   observations for filtering. 
            #   A post-filtering temporal-coverage check (as in Trudel et al. (2026), 
            #   min span 1 year and max gap 120 days) is recommended before using 
            #   lenient results for validation or phenology.

            if len(df_apply) < minimum_data_n: # Data points too sparse to yield reliable pattern.
                df_eval = df_eval.iloc[0:0] # Clear up df_eval
                n_while = -1 # -1 indicates this lake is abandoned.
                break # break the while loop
            else:
                exceeds_limit = (df_apply['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
                if allow_major_gap == 'no': # Do not allow major gaps in df_apply or in df_eval
                    if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                        df_eval = df_eval.iloc[0:0] # Clear up df_eval
                        n_while = -1 # -1 indicates this lake is abandoned.
                        break
                    else: # Check if the time range of df_apply is too short (e.g., 1 year)
                        if (df_apply['datetime'].max() - df_apply['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
                            df_eval = df_eval.iloc[0:0] # Clear up df_eval
                            n_while = -1 # -1 indicates this lake is abandoned.
                            break
                else: # Remove observations in df_eval that fall within any gap longer than max_temporal_gap in df_apply
                    if exceeds_limit.any(): # Check if any time difference exceeds max_temporal_gap
                        df_eval = drop_eval_in_apply_gaps(df_eval, df_apply, max_temporal_gap, 'datetime')
                        # Since df_apply size >= minimum_data_n, df_eval after gap removal still >= minimum_data_n.


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
                    n_jobs=-1) # No need to interpolate unequal time
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
                    ) # No need to interpolate unequal time
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

            # Preserve the evaluated time before df_eval is updated. 
            # Time_eval is only used if show_filtering_evolution is set to 'yes'.
            time_eval = df_eval['datetime']

            # Compute the z-score
            # Assign residuals to df_eval; the column is created on the first iteration.
            df_eval['residuals'] = residuals
            if np.nansum(np.abs(residuals)) == 0: # Note sometimes all residuals are 0 due to overfitting.
                z_scores = (residuals - np.nanmean(residuals))/1.0 # Force it to be 0, so there will be no outliers.
            else:
                z_scores = (residuals - np.nanmean(residuals))/np.nanstd(residuals)

            # Z-score clipping
            # Check whether residuals need to be removed or not based on how spread the residuals are.
            abs_residual_p = np.abs(df_eval['residuals']) / ( np.max(df_apply['wse'])  - np.min(df_apply['wse']) )
            # Define mask based on combined conditions
            mask = (np.abs(z_scores) < z_score_thresholds[0]) | (abs_residual_p < maximum_residual_spreads[0])
            # Apply mask to filter df
            df_eval = df_eval[mask] # Update df_eval by removing outliers for the next iteration.

            # Remove positive anomalies during freeze-up/ice-covered period
            # First by area_total and/or wse. Set "by_pass" to be True because area_total is pass dependent.
            # The multiplier is set higher to de-risk over-rejection due to limited observations per pass.
            df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=True, by_crid_scenario=False,
                                    multiplier=0.3, lower_q=0, upper_q=1, used_q='upper', filter_by='both')  #or filter_by='area'
            # Second by wse. Set "by_pass" to be False to make the removal more general if possible (to avoid over-rejection)
            # This second removal may be necessary as pass-specific outliers may remain if there's no ice-free observation for that pass.
            df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=False, by_crid_scenario=False,
                                    multiplier=0.2, lower_q=0, upper_q=1, used_q='upper', filter_by='wse') #area, wse, or both
            #Note: Users can optimize their "filter_by" and "pass_by" parameters.

            # Remove remaining isolated extreme outliers using Tukey method (IQR method)
            # Here, we use 10th and 90th percentile.
            df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse', multiplier=3, lower_q=0.1, upper_q=0.9)

            # Further remove WSE observations that are still 150 m higher than the median WSE of this lake
            # Typical seasonal range for large reservoirs: 10–60 meters (e.g., the Three Gorges Reservoir ranges in 145-175 m).
            # Very large reservoirs (e.g., hydropower or multipurpose dams): can exceed 100 meters
            # A few massive reservoirs may approach or even exceed 150–300 meters in water level fluctuation:
            # e.g., Jinping-I Dam, 305 m; lake_id [4610062383, 4610049903];
            # According to the Global Dam Watch (GDW) and GeoDAR/ICOLD datasets, quite a few dams are higher than 200-300 m.
            # So, we here use a 150-m threshold to be safe, but this may retain residual outliers. 
            df_eval = df_eval[ np.abs(df_eval['wse'] - np.median(df_eval['wse'])) <= 150 ]

            # Plot filter evolution if preferred. 
            # Caution: this will generate a series of plots (one per iteration). Set show_filtering_evolution to 'no' unless necessary. 
            if show_filtering_evolution == 'yes': # Show how outlier removal evolves through iteration.
                plt.rcParams["font.family"] = "Arial"
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.grid(True, linewidth=0.5, zorder=1)

                # Plot gauge measurements if the lake has gauge data
                if gauge_df is not None:
                    # Compute a preliminary datum bias between SWOT and gauge measurements.
                    # Note this bias correction is preliminary and is only intended here for visualization.
                    bias_swot_gauge_prelim = np.nanmedian(gauge_df['gauge_wse']) - np.nanmedian(df['wse'])
                    ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse'] - bias_swot_gauge_prelim, \
                            label='gauge', color='green', marker = 'o', markersize=6, linestyle='--') 
                        
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
                ax.set_title('Lake ID ' + str(df["lake_id"].unique()[0]) + ' WSE Plot: ' + filter_type)
                ax.legend()

            # Update the length of df_eval (evaluated data after outlier removal)
            updated_length = len(df_eval)
            n_while += 1

        # Double check r1 low-pass result
        if len(df_eval) < minimum_data_n: # data points too sparse to yield reliable pattern.
            df_eval = df_eval.iloc[0:0] # Clear up df_eval
            n_while = -1 # -1 indicates this lake is abandoned.
        else:
            exceeds_limit = (df_eval['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
            if allow_major_gap == 'no': # Do not allow major gaps in df_eval
                if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                    df_eval = df_eval.iloc[0:0] # Clear up df_eval
                    n_while = -1 # -1 indicates this lake is abandoned.
                else: # Check if the time range of df_eval is too short (1 year)
                    if (df_eval['datetime'].max() - df_eval['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
                        df_eval = df_eval.iloc[0:0] # Clear up df_eval
                        n_while = -1 # -1 indicates this lake is abandoned.



        """
        Optional: High-quality data recovery and round-2 filtering

        High-quality LakeSP observations may have been unintentionally removed in round 1 above when the filter struggled to eliminate extreme outliers.
        The following section provides an option (i.e., if recovering_observations == "yes"), to reintroduce those removed high-quality observations.

        After high-quality observations are reintroduced, it is recommended to run another round (round 2) filtering, which is less aggressive than round 1,
        to ensure the elimination of very extreme outliers.
        """
        if r2_filter == 'yes' and n_while >0: # If this option is turned on, and round-1 is valid.
            n_while_r2 = 0 # Initialize the iteration times for round-2 filtering (turn on).

            # -----Reintroduce/recover high-quality observations-----
            # Initialize df_good_quality as a subset of df.
            # Apply stricter quality control: retain only observations flagged as "good" by built-in quality flags
            df_good_quality = df[(df['xovr_cal_q'] == 0) & (df['quality_f'] == 0) & (df['ice_clim_f'] == 0)]

            # Further apply heuristic thresholds (no ice period this time)
            df_good_quality = apply_heuristic_thresholds(df_good_quality, df_heuristic_thresholds,
                                                         wse_std_threshold_bounds = wse_std_threshold_bounds,
                                                         wse_u_threshold_bounds   = wse_u_threshold_bounds,
                                                         xtrk_dist_threshold_bounds = xtrk_dist_threshold_bounds,
                                                         wse_std_ice_min = wse_std_ice_min, wse_u_ice_min = wse_u_ice_min,
                                                         rules_for_ice_free_data   = rules_for_ice_free_data,  #(per-metric: [wse_std, wse_u, xtrk_dist])
                                                         rules_for_ice_covered_data= rules_for_ice_covered_data)

            # Identify high-quality observations not already present in df_eval based on index_col
            df_to_recover = df_good_quality[~df_good_quality['index_col'].isin(df_eval['index_col'])]

            # Append the recovered observations to df_eval
            df_eval_locked = df_eval.copy() # Locked round-1 result used as the starting point for recovery.
            df_eval = pd.concat([df_eval, df_to_recover], ignore_index=True)

            # Sort df_eval by high-precision datetime to maintain chronological order
            df_eval = df_eval.sort_values('datetime', kind='mergesort').reset_index(drop=True) # mergesort keeps relative order for ties.
            # Note the code above is safe even when df_good_quality is empty.


            # -----Run a round-2, less aggressive filtering-----
            # The logic is consistent with round 1, except that the filter is applied and evaluated on the same data: df_eval.
            initial_length = len(df_eval) # In case this is zero, the "while" statement will not run.
            updated_length = 0  # Initialize the length of the updated df_eval
            while (updated_length < initial_length) and (n_while_r2 < 5): # A max of 5 iteration times to avoid over-rejection in round-2 filtering.
                initial_length = len(df_eval) # Note: df_eval is updated per iteration.


                # Check whether the round-2 candidate series has enough temporal support.
                #
                # Different from round 1, the low-pass filter is applied to and evaluated on
                # the same dataframe: df_eval. There is no separate df_apply baseline in this
                # round.
                #
                # This lake is abandoned if any of the follwoing situations occurs.
                #
                # If df_eval contains too few observations to support a reliable smoothing curve.
                #
                # If allow_major_gap == "no", strict mode:
                # - df_eval must not contain any temporal gap > max_temporal_gap.
                # - df_eval must span at least min_temporal_range days.
                # - If either condition fails, round 2 is abandoned.
                #
                # If allow_major_gap == "yes", lenient mode:
                # - Large gaps in df_eval are allowed.
                # - The gap and minimum-span requirements are not enforced in round 2.
                # - The candidate series must still contain enough observations for filtering.
                # - A post-filtering temporal-coverage check is recommended before using
                #   lenient results for validation or phenology.
                
                if len(df_eval) < minimum_data_n: #data points too sparse to yield reliable pattern.
                    df_eval = df_eval.iloc[0:0] # Clear up df_eval
                    n_while_r2 = -1 # -1 indicates this lake is abandoned.
                    break # break the while loop
                else:
                    exceeds_limit = (df_eval['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
                    if allow_major_gap == 'no': # do not allow major gaps in df_eval or in df_eval
                        if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                            df_eval = df_eval.iloc[0:0] # Clear up df_eval
                            n_while_r2 = -1 # -1 indicates this lake is abandoned.
                            break # break the while loop
                        else: # Check if the time range of df_eval is too short (1 year)
                            if (df_eval['datetime'].max() - df_eval['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
                                df_eval = df_eval.iloc[0:0] # Clear up df_eval
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
                        n_jobs=-1) # No need to interpolate unequal time
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


                # Preserve the original df_eval before it is modified. 
                # This is only used if show_filtering_evolution is set to 'yes'.
                df_eval_original = df_eval.copy()

                # Compute the z-score
                # Assign residuals to df_eval; the column is created on the first iteration.
                df_eval['residuals'] = residuals
                if np.nansum(np.abs(residuals)) == 0: # Note sometimes all residuals are 0 due to overfitting.
                    z_scores = (residuals - np.nanmean(residuals))/1.0 # Force it to be 0, so there will be no outliers.
                else:
                    z_scores = (residuals - np.nanmean(residuals))/np.nanstd(residuals)

                # Z-score clipping
                # Check whether residuals need to be removed or not based on how spread the residuals are.
                # This is evaluated by maximum_residual_spread, which computes the maximum residual as a proportion of df_eval range.
                abs_residual_p = np.abs(df_eval['residuals']) / ( np.max(df_eval['wse']) - np.min(df_eval['wse']) )
                # Define mask based on combined conditions
                mask = (np.abs(z_scores) < z_score_thresholds[1]) | (abs_residual_p < maximum_residual_spreads[1])
                # Apply mask to filter df
                df_eval = df_eval[mask] # Update df_eval by removing outliers for the next iteration.

                # Remove positive anomalies during the freeze-up/ice-covered period
                # First by area_total and/or wse. Set "by_pass" to be True because area_total is pass dependent.
                # The multiplier is set higher to de-risk over-rejection due to limited observations per pass.
                df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=True, by_crid_scenario=False,
                                        multiplier=0.3, lower_q=0, upper_q=1, used_q='upper', filter_by='both')  #or filter_by='area'
                # Second by wse. Set "by_pass" to be False to make the removal more general if possible (to avoid over-rejection)
                # This second removal may be necessary as pass-specific outliers may remain if there's no ice-free observation for that pass.
                df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=False, by_crid_scenario=False,
                                        multiplier=0.2, lower_q=0, upper_q=1, used_q='upper', filter_by='wse') #area, wse, or both
                #Note: Users can optimize their "filter_by" and "pass_by" parameters.

                # Remove remaining isolated outliers using Tukey method (IQR method)
                # Use 10th and 90th percentile.
                df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse', multiplier=3, lower_q=0.1, upper_q=0.9)

                # Further remove observations that are still 150 m higher than the median WSE
                # See reasoning in round-1. 
                df_eval = df_eval[ np.abs(df_eval['wse'] - np.median(df_eval['wse'])) <= 150 ]
                
                # Apply lock scheme for round-1 result
                # Recover observations from df_eval_locked that are removed from df_eval (based on index_col).
                df_to_recover_locked = df_eval_locked[~df_eval_locked['index_col'].isin(df_eval['index_col'])]
                df_eval = pd.concat([df_eval, df_to_recover_locked], ignore_index=True)
                df_eval = df_eval.sort_values('datetime', kind='mergesort').reset_index(drop=True) #mergesor


                # Plot filter evolution if preferred. Caution: this will generate a series of plots (one per iteration)
                if show_filtering_evolution == 'yes': # Show how outlier removal evolves through iteration.
                    plt.rcParams["font.family"] = "Arial"
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.grid(True, linewidth=0.5, zorder=1)

                    # Plot gauge measurements if the lake has gauge data
                    if gauge_df is not None:
                        # Compute a preliminary datum bias between SWOT and gauge measurements.
                        # Note this bias correction is preliminary and is only intended here for visualization.
                        bias_swot_gauge_prelim = np.nanmedian(gauge_df['gauge_wse']) - np.nanmedian(df['wse'])
                        ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse'] - bias_swot_gauge_prelim, \
                                label='gauge', color='green', marker = 'o', markersize=6, linestyle='--') # Shift gauge to SWOT datum.

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
                    ax.set_title('Lake ID ' + str(df["lake_id"].unique()[0]) + ' WSE Plot: ' + filter_type + ' (round 2)')
                    ax.legend()

                # Update the length of df_eval (evaluated data after outlier removal)
                updated_length = len(df_eval)
                n_while_r2 += 1

            # Double check r2 low-pass result
            if len(df_eval) < minimum_data_n: #data points too sparse to yield reliable pattern.
                df_eval = df_eval.iloc[0:0] # Clear up df_eval
                n_while_r2 = -1 # -1 indicates this lake is abandoned.
            else:
                exceeds_limit = (df_eval['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
                if allow_major_gap == 'no': # do not allow major gaps in df_eval or in df_eval
                    if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                        df_eval = df_eval.iloc[0:0] # Clear up df_eval
                        n_while_r2 = -1 # -1 indicates this lake is abandoned.
                    else: # Check if the time range of df_eval is too short (1 year)
                        if (df_eval['datetime'].max() - df_eval['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
                            df_eval = df_eval.iloc[0:0] # Clear up df_eval
                            n_while_r2 = -1 # -1 indicates this lake is abandoned.

        # Unpack filter summary:
        if r2_filter == 'yes': # filter_r2 is turned on:
            n_while_use = n_while_r2 # use r2 result. This can be -2 is n_while <= 0 (failed)
        else:
            n_while_use = n_while # use r1 result

        if n_while_use <= 0:
            filter_status = 'fail'
        else: # n_while_use > 0:
            filter_status = 'success'

        return df_eval, [n_while, n_while_r2], filter_status, threshold_summary, observation_thresholds


def apply_baseline_tukey_filter(
    df_eval,
    baseline_SQL,
    multiplier=3,
    lower_q=0.1,
    upper_q=0.9,
    iteration_n=5
):
    """
    This function enables a simple filter based on a baseline time series defined by baseline_SQL,
    then filtered by a Tukey IQR removal (remove_tukey_outliers function).

    Parameters:
        df_eval (DataFrame): Initial LakeSP time series
        baseline_SQL (str):
            pandas .query expression; rows meeting this are used as the baseline time series.
        multiplier, lower_q, upper_q: Inputs for the remove_tukey_outliers function (see detailed in remove_tukey_outliers)
        iteration_n (Integer): Maximum number of iterations for Tukey outlier removal

    Returns:
        df_eval (DataFrame): Filtered LakeSP time series
        n_while (Integer): Number of iterations for Tukey noise removal
            -9: Indicates the original LakeSP input (df_eval) is empty and the filter is not applicable
            0:  Indicates df_eval became empty after baseline subsetting (no good observations to initiate the low-pass filtering)
            -1: Indicates the iteration started but was abandoned
            >0: Indicates the number of regular iterations
        filter_status [text]: scenarios of filtered result
            - no data
            - fail
            - success

    """
    # In case the input time series DataFrame is empty.
    if df_eval.empty:
        return df_eval.copy(), -9, 'no data'

    df_eval = df_eval.copy() # Security measure
    n_while = 0 # Initialize the iteration times
    # Generate the baseline time series
    df_eval = df_eval.query(baseline_SQL) #engine='python' not needed

    # Remove remaining isolated extreme outliers using Tukey method (IQR method)
    initial_length = len(df_eval) # In case this is zero, the "while" statement won't run.
    updated_length = 0  # Initialize the length of the updated df_eval
    while (updated_length < initial_length) and (n_while < iteration_n):
        initial_length = len(df_eval) # Note: df_eval is update
        # Use 10th and 90th percentile.
        df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse', \
                                                  multiplier=multiplier, lower_q=lower_q, upper_q=upper_q)

        # Further remove observations that are still 150 m higher than the median WSE
        # Typical range for large reservoirs: 10–60 meters (e.g., the Three Gorges Reservoir ranges in 145-175 m).
        # Very large reservoirs (e.g., hydropower or multipurpose dams): can exceed 100 meters
        # A few massive reservoirs may approach or even exceed 150–200 meters in water level fluctuation.
        df_eval = df_eval[ np.abs(df_eval['wse'] - np.median(df_eval['wse'])) <= 150 ]
        # Note: this works if df_eval is empty.

        # Update the length of df_eval (evaluated data after outlier removal)
        updated_length = len(df_eval)
        n_while += 1

    # Double check result
    if len(df_eval) == 0:
        df_eval = df_eval.iloc[0:0] # Clear up df_eval
        n_while = -1 # -1 indicates this lake is abandoned.

    if n_while <= 0:
        filter_status = 'fail'
    else: # n_while > 0:
        filter_status = 'success'

    return df_eval, n_while, filter_status

def sp_cycle_adjustment(df_eval):
    """
    Reduce intra-cycle cross-pass WSE inconsistencies caused by multiple orbit passes.
    For lakes spanning multiple SWOT orbit passes, WSE values within the same orbit cycle may show substantial
    inconsistencies (e.g., zig-zag patterns) across different passes.

    Logic: The following three options are provided to mitigate this issue:
        - Option 1: Compute a cycle-averaged WSE time series.
                    Averaging all WSE values within each cycle can help eliminate intra-cycle inconsistencies.
        - Option 2: Retain only observations from the pass that captures the largest observed lake area (area_total).
                    The representative pass is identified based on the highest median area_total across the time series.
                    Note: Both Option 1 and Option 2 yield one WSE value per cycle.
        - Option 3 (recommended): Adjust each WSE value by removing its pass-specific bias relative to the overall WSE
                    average across the time series. This approach preserves the original number of observations and has
                    been shown to produce more reliable results.
    Note that option 2 and option 3 will not run if intra-cycle WSE inconsistency is insignificant.

    Parameters:
        df_eval (DataFrame): Initial LakeSP time series

    Returns:
        df_option1, df_option2, df_option3: cycle-adjusted time series for each of the three options.
    """
    # Copy the original input for protection measure
    df_eval = df_eval.copy() # The script below handles the case of empty dataframe.

    # Duplicate "wse" values to a new column "wse_adjusted" in df_eval (results after filtering).
    # If cycle-adjustment is needed, wse_adjusted will be updated to be the cycle-adjusted WSEs for option 3.
    # Otherwise, wse_adjusted will remain a duplicate of wse (after filtering).
    df_eval['wse_adjusted'] = df_eval['wse']

    # Option 1: Cycle-averaged WSE time series. Note that cycle_id will be sorted in ascending order.
    df_cycle_avg = df_eval.groupby('cycle_id')['wse'].mean().rename('wse_cycle_avg').reset_index()
    # Compute the middle observation date per cycle
    cycle_dates = df_eval.groupby('cycle_id')['datetime'].median().rename('mid_date').reset_index()
    # Merge with df_cycle_avg based on cycle_id. Merged dataframe contains mid_date and wse_cycle_avg columns
    df_option1 = pd.merge(df_cycle_avg, cycle_dates, on='cycle_id')

    # Compare intra-cycle vs inter-cycle WSE variability
    intra_cycle_std = df_eval.groupby('cycle_id')['wse'].std().median() # Computed as the median of cycle-level WSE standard deviations.
    inter_cycle_std = df_option1['wse_cycle_avg'].std() # Computed as the standard deviation of cycle-averaged WSEs

    # Check if options 2 and 3 cycle adjustment is needed: intra-cycle variability must exceed inter-cycle variability
    if intra_cycle_std > inter_cycle_std:
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

        # Since wse values are adjusted, we run another Tukey outlier removal on df_eval.wse_adjusted, just on the safe side. 
        df_option3, _, _ = remove_tukey_outliers(df_eval, col='wse_adjusted', multiplier=3, lower_q=0.1, upper_q=0.9)

    else: # Return option 2 and option 3 both as df_eval as neither option is applied.
        df_option2 = df_eval.copy()
        df_option3 = df_eval.copy()

    return df_option1, df_option2, df_option3