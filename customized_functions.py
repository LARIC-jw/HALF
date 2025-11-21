"""
Customized function module
Initialized: 04/20/2025
Last updated: 09/15/2025
Authors: 
    Jida Wang (jidaw@illinois.edu); 
    Melanie Trudel (melanie.trudel@usherbrooke.ca)
"""

import pywt
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d, PchipInterpolator, UnivariateSpline
#from io import StringIO
from joblib import Parallel, delayed
from scipy.signal import savgol_filter, medfilt
from pykalman import KalmanFilter
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#import seaborn as sns

"""
Functions: Do not change the functions unless necessary. 
    Basic functions:
        compute_rmse:              Computes root mean squared error (RMSE), np.nan robust. 
        compute_correlation:       Computes Pearson or Spearman correlation coefficient
        remove_tukey_outliers:     Removes outliers using a generalized Tukey method (IQR-based).
        calibrate_heuristic_thresholds: Calibrate heuristic thresholds (max wse_std, max wse_u, and min xtrk_dist) before SP filtering.
        apply_heuristic_thresholds: Applies heuristic thresholds to subset the benmark for SP filtering.
        filter_ice_outliers:       Removes anomalies ice-covered/freeze-up observations.
        convert_to_daily_series:   Compute daily-interpolated WSEs from SWOT and gauge data over their overlapping time range.
        drop_eval_in_apply_gaps:   Remove data from the evaluated time series whose timestamps fall in large gaps in the baseline time series.
        apply_customized_filter:   Apply heuristic thresholds to filter the SP time series (where apply_heuristic_thresholds will be executed as well).
        apply_baseline_tukey_filter: A simple filter based on a baseline time series defined by baseline_SQL and Tukey IQR removal. 
        sp_cycle_adjustment:       Reduce intra-cycle WSE inconsistencies in the SP time series caused by multiple orbit passes.        
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

def compute_correlation(y, y_hat, method='pearson'):
    """
    Computes correlation coefficient between two arrays using Spearman or Pearson method.
    
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

# Modified on 08/14/2025 to include the heuristic threshold for xtrk_dist and ice condition. 
def calibrate_heuristic_thresholds(
    df, 
    conservative_SQL,
    by_crid_scenario=[False, False, False],
    by_pass_id=[False, False, True],
    by_ice=[True, True, True]
):
    """
    Calibrate heuristic thresholds (wse_std, wse_u, xtrk_dist) for SWOT lake filtering.
    Overview
    --------
    This function takes a DataFrame of SWOT water surface elevation (WSE) quality
    metrics, computes heuristic thresholds, and applies a structured set of "fallout rules"
    to fill missing thresholds. It enforces consistency by ensuring that for each
    (crid_scenario, pass_id) pair, both ice states and a synthetic "both" state exist.
    
    Parameters
    ----------
         df (pd.DataFrame): Input dataframe containing at least the following columns:
             ['lake_id', 'crid', 'pass_id', 'ice_clim_f', 'wse_std', 'wse_u', 'xtrk_dist']
         conservative_SQL : str
             pandas .query expression; rows meeting this are used to calibrate thresholds.
         by_crid_scenario, by_pass_id, by_ice : list[bool] of length 3
             Per-metric grouping toggles for [wse_std, wse_u, xtrk_dist].
            • by_crid_scenario (list of bool): If True, use crid_scenario for grouping.
            • by_pass_id (list of bool): If True, use pass_id for grouping
            • by_ice (list of bool):  If True, use ice flag (ice_clim_f >=2 or < 2) for grouping

         note: xtrk_dist (in m, valid max: 75000 m): Distance of the lake polygon centroid from the spacecraft nadir track;
             this value is computed using a local spherical Earth approximation.
             A negative value indicates that the lake is on the left side of the swath, relative to the spacecraft velocity vector.
             A positive value indicates that the lake is on the right side of the swath.
             So, absolute value is used for simplicity.

    Key steps
    ---------
    1) Base rows construction:
       - Extract unique combinations of (crid_scenario, pass_id, ice_condition) from df.
       - If only one ice_condition is present for a given (crid_scenario, pass_id), add
         the missing one and mark it with grouping_scheme = 3.
       - Add an additional "both" row (representing both ice conditions) for every (crid_scenario, pass_id).
       - Preserve the order of (crid_scenario, pass_id) pairs as they first appear in df.

    2) Grouping scheme marking (before fallout):
       - For ice rows:
           * grouping_scheme = 1 → key combo exists in conservative_SQL subset
           * grouping_scheme = 2 → key combo exists in df but not conservative_SQL
           * grouping_scheme = 3 → synthetic ice row added in step 1
       - For "both" rows:
           * grouping_scheme = 1 → (crid_scenario, pass_id) exists in conservative subset
           * grouping_scheme = 2 → otherwise

    3) Threshold calibration from conservative subset:
       - Always compute from df.query(conservative_SQL).
       - Ice rows: use per-metric toggles (by_crid_scenario, by_pass_id, by_ice).
       - "Both" rows: use the *same* per-metric toggles for crid/pass
         (by_crid_scenario, by_pass_id) but **ignore by_ice**.
       Aggregation rule per metric:
         * wse_std_threshold, wse_u_threshold = max
         * xtrk_dist_threshold = min

    4) Fallout rules for missing thresholds:
       - Ice rows (ice-covered / ice-free):
           * wse_std & wse_u (ice-condition-centric):
               L1: same ice + pass_id → max
               L2: same ice + crid_scenario → max
               L3: same ice → max
               L4: NaN, awaiting hard default (max of bounds) in apply_customized_filter
           * xtrk_dist (pass-centric):
               L1: same pass + ice → min
               L2: same pass + crid_scenario → min
               L3: same pass → min
               L4: NaN, awaiting hard default (min of bounds) in apply_customized_filter

       - "Both" rows (consider ONLY other "both" rows):
           * wse_std & wse_u:
               L1: same pass_id → max
               L2: same crid_scenario → max
               L3: NaN, awaiting hard default (max of bounds) in apply_customized_filter
           * xtrk_dist:
               L1: same pass_id → min
               L2: same crid_scenario → min
               L3: NaN, awaiting hard default (min of bounds) in apply_customized_filter

    5) Finalization:
       - Rows sorted by the order of (crid_scenario, pass_id) as first seen in df,
         then by ice_condition.

    Returns
    -------
    The returned DataFrame ALWAYS contains 'crid_scenario', 'pass_id',
             and 'ice_condition' columns (plus 'lake_id' and the three thresholds),
             even if the grouping for a particular metric does not use all keys.
             > lake_id: PLD lake id of the input df.
             > crid_scenario: "PIC2_or_PID0" or "early_versions" (e.g., PIC0, PGC0)
             > pass_id: SWOT orbit pass
             > ice_condition: ice-covered or ice-free
                 - ice-covered: ice_clim_f >= 2
                 - ice-free:    ice_clim_f < 2
                 - both:        either case, not distinguishing between ice-covered or -free. 
             > wse_std_threshold: the maximum wse_std threshold under this pass_id, crid_scenario, and ice_condition combination
             > wse_u_threshold: the maximum wse_u threshold under this pass_id, crid_scenario, and ice_condition combination
             > xtrk_dist_threshold: the minimum abs(xtrk_dist) threshold under this pass_id, crid_scenario, and ice_condition combination

    Note:
           • Returned rows in base_df are all unique combinations of
             (crid_scenario, pass_id, ice_condition) present in the input df,
             as well as the synthetic rows with the opposite ice_condition (if not present in df), and both ice conditions. 
           • grouping_scheme encodes the origin of each row for traceability.
           • However, thresholds are computed only from rows that satisfy `conservative_SQL`.
           • For rows not satisfying `conservative_SQL`, the fallout rules described above fill the NaN values.
           • ice_condition values: 'ice-covered', 'ice-free', and synthetic 'both'.    
    """
    # -------------------------
    # Handle empty df
    # -------------------------
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                'lake_id','crid_scenario','pass_id','ice_condition',
                'wse_std_threshold','wse_u_threshold','xtrk_dist_threshold','grouping_scheme'
            ]
        )
    lake_id = df['lake_id'].iloc[0] if ('lake_id' in df.columns) else 'unknown'

    # -------------------------
    # Preprocess
    # -------------------------
    df = df.copy()
    df['crid_scenario'] = df['crid'].apply(
        lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions'
    )
    df['xtrk_dist_abs'] = df['xtrk_dist'].abs()
    df['ice_condition'] = np.where(df['ice_clim_f'] >= 2, 'ice-covered', 'ice-free')

    # Conservative subset used for threshold calibration
    try:
        df_cons = df.query(conservative_SQL) #engine='python' not needed
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
    # Mark grouping_scheme BEFORE fallout
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
        """Grouping keys for ICE rows (may include ice_condition)."""
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
        ('wse_std','wse_std_threshold',0,'max'),
        ('wse_u','wse_u_threshold',1,'max'),
        ('xtrk_dist_abs','xtrk_dist_threshold',2,'min'),
    ]

    # 1) ICE rows calibration
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
    # Fallout rules (fill remaining NaNs)
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
    
    # A) ICE-condition-centric fallout for wse_std_threshold and wse_u_threshold (only ice rows)
    for col in ['wse_std_threshold', 'wse_u_threshold']:
        tgt = base_df['ice_condition'].isin(['ice-covered','ice-free'])
        # Level 1: same ice_condition + pass_id → max
        base_df = _fill_by_group(base_df, tgt, ['ice_condition','pass_id'], col, 'max')          
        # Level 2: same ice_condition + crid_scenario → max
        base_df = _fill_by_group(base_df, tgt, ['ice_condition','crid_scenario'], col, 'max')    
        # Level 3: same ice_condition → max
        base_df = _fill_by_group(base_df, tgt, ['ice_condition'], col, 'max')                    
        ## Level 4: hard default = max(bounds) 
        #base_df.loc[tgt & base_df[col].isna(), col] = max(bounds) # Leave it as NaN for now                               

    # B) Pass-centric fallout for xtrk_dist_threshold (only ice rows)
    col = 'xtrk_dist_threshold'
    tgt = base_df['ice_condition'].isin(['ice-covered','ice-free'])
    # Level 1: same pass_id + ice_condition → min
    base_df = _fill_by_group(base_df, tgt, ['pass_id','ice_condition'], col, 'min')              
    # Level 2: same pass_id + crid_scenario → min
    base_df = _fill_by_group(base_df, tgt, ['pass_id','crid_scenario'], col, 'min')              
    # Level 3: same pass_id → min
    base_df = _fill_by_group(base_df, tgt, ['pass_id'], col, 'min')                              
    ## Level 4: hard default = min(bounds)
    #base_df.loc[tgt & base_df[col].isna(), col] = min(bounds) # Leave it as NaN for now

    # C) BOTH rows fallout (only among "both" rows)
    tgt_both = base_df['ice_condition'].eq('both')
    # Level 1: same pass_id → max/max/min
    for col, red in [('wse_std_threshold','max'), ('wse_u_threshold','max'), ('xtrk_dist_threshold','min')]:
        base_df = _fill_by_group(base_df, tgt_both, ['pass_id'], col, red)                       
    # Level 2: same crid_scenario → max/max/min
    for col, red in [('wse_std_threshold','max'), ('wse_u_threshold','max'), ('xtrk_dist_threshold','min')]:
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
                    'wse_std_threshold','wse_u_threshold','xtrk_dist_threshold','grouping_scheme']]

# Modified on 08/15/2025 to include the heuristic threshold for xtrk_dist and ice condition. 
def apply_heuristic_thresholds(
    df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    # NEW: Bound overrides for applied thresholds_df
    wse_std_threshold_bounds = [0, 3],
    wse_u_threshold_bounds   = [0, 0.5],
    xtrk_dist_threshold_bounds = [0, 75000],
    
    # Ice overrides for applied thresholds on ice-affected rows
    wse_std_ice_min: float = 3.0,
    wse_u_ice_min: float   = 0.5,    
    
    # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
    # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
    rules_for_ice_free_data    = ['ice-free', 'ice-free', 'not apply'],
    rules_for_ice_covered_data = ['ice-free', 'ice-free', 'ice-covered']
):
    """
    Apply calibrated heuristic thresholds (from calibrate_heuristic_thresholds) to df.

    Matching keys and inputs
    ------------------------
    - df rows must include at least: ['crid', 'pass_id', 'ice_clim_f', 'wse_std', 'wse_u', 'xtrk_dist'].
    - thresholds_df is the output of calibrate_heuristic_thresholds, which contains rows
      for each unique (crid_scenario, pass_id) with three ice_condition values:
        'ice-free', 'ice-covered', and 'both'.
      Each has the three thresholds:
        ['wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold'].
    - wse_std_threshold_bounds (list): [min, max] bounds when applying wse_std threshold
    - wse_u_threshold_bounds (list): [min, max] bounds when applying wse_u threshold
    - xtrk_dist_threshold_bounds (list): [min, max] bounds when applying abs(xtrk_dist) threshold

    Rule system
    -----------
    We import all three sets of thresholds (ice-free / ice-covered / both) and, 
    for each row, select which one to apply **per metric** based on:
      - rules_for_ice_free_data    : used when the row is actually ice-free (ice_clim_f < 2)
      - rules_for_ice_covered_data : used when the row is actually ice-covered (ice_clim_f = 2)

    Each rules_* list has 3 strings (for [wse_std, wse_u, xtrk_dist], respectively), 
    where each string is one of:
      'ice-free' | 'ice-covered' | 'both' | 'not apply'
      ice-free: means apply the threshold calibrated or configured when the lake is ice free
      ice-covered: means apply the threshold calibrated or configured when the lake is affected by ice
      both: means apply the threshold based on other keys regardless of the ice condition
      not apply: means do not apply this metric for thresholding when filtering df. 

    More interpretation
    --------------
    - As described above, if a metric rule is 'not apply', that metric does NOT 
      gate the row (i.e., ignored).
    - Otherwise, we compare:
        wse_std        <= selected wse_std_threshold
        wse_u          <= selected wse_u_threshold
        abs(xtrk_dist) >= selected xtrk_dist_threshold
      All **applied** metrics must pass for the row to be kept.

    No fallout, but bounds as last resort
    -------------------------------------
    To allow flexibility, bound-clipping and fallout will first be applied to thresholds_df. 
    For security, if a selected threshold is still NaN for a metric
    that **applies**, we replace it with a bounds-based default:
      - wse_std, wse_u: use max(bounds)
      - xtrk_dist     : use min(bounds)

    Ice override
    --------------------------------------
    For rows with ice influence (we use ice_clim_f >= 1 here for conservatism),
    if a metric applies and its threshold exists, we bump it to the minimum:
      wse_std_threshold_eff = max(wse_std_threshold_eff, wse_std_ice_min)
      wse_u_threshold_eff   = max(wse_u_threshold_eff,   wse_u_ice_min)

    Returns
    -------
    Filtered subset of df that passes all **applied** metric checks. 
    Only the original columns in the filtered df are returned. 
    """

    # ---------------------------------------------------------------------
    # Quick guards & rule validation
    # ---------------------------------------------------------------------
    if thresholds_df.empty or df.empty:
        #print("[apply_heuristic_thresholds] ALERT: df and thresholds_df are empty. "
        #      "No rows can be validated. Returning empty DataFrame.")
        return df.iloc[0:0]  # preserve schema
    
    # Make sure rule list has no typo. 
    def _ok_rule_list(lst):
        # Each list must be length-3 with per-metric entries,
        # each entry is one of the allowed strings:
        allowed = {'ice-free', 'ice-covered', 'both', 'not apply'}
        return isinstance(lst, (list, tuple)) and len(lst) == 3 and all(x in allowed for x in lst)

    if not _ok_rule_list(rules_for_ice_free_data):
        raise ValueError("rules_for_ice_free_data must be a 3-item list of "
                         "['ice-free'|'ice-covered'|'both'|'not apply'].")
    if not _ok_rule_list(rules_for_ice_covered_data):
        raise ValueError("rules_for_ice_covered_data must be a 3-item list of "
                         "['ice-free'|'ice-covered'|'both'|'not apply'].")

    # Keep only original df columns for the return value
    original_columns = df.columns.tolist()
    df = df.copy()
    thr = thresholds_df.copy()
    
    # Apply bound clipping and fallout for thr
    thr['wse_std_threshold'] = (
        thr['wse_std_threshold']
        .fillna(wse_std_threshold_bounds[1])    # replace NaN with upper bound
        .clip(*wse_std_threshold_bounds)        # clip values to [lower, upper]       
    )
    
    thr['wse_u_threshold'] = (
        thr['wse_u_threshold']
        .fillna(wse_u_threshold_bounds[1])      # replace NaN with upper bound
        .clip(*wse_u_threshold_bounds)          # clip values to [lower, upper]       
    )
    
    thr['xtrk_dist_threshold'] = (
        thr['xtrk_dist_threshold']
        .fillna(xtrk_dist_threshold_bounds[0])  # replace NaN with lower bound
        .clip(*xtrk_dist_threshold_bounds)      # clip values to [lower, upper]       
    )
    
    # ---------------------------------------------------------------------
    # Compute crid_scenario & row ice_condition, to align with thresholds
    # ---------------------------------------------------------------------
    df['crid_scenario'] = df['crid'].apply(
        lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions'
    )
    # Row ice condition (for choosing which ruleset to use)
    df['ice_condition'] = np.where(df['ice_clim_f'] >= 2, 'ice-covered', 'ice-free')

    # We will merge by (crid_scenario, pass_id). To avoid dtype mismatches,
    # we create string-typed merge keys on both frames.
    df['_crid_scenario_str'] = df['crid_scenario'].astype(str)
    df['_pass_id_str']       = df['pass_id'].astype(str)

    # Helper: prepare a thresholds subtable for a specific ice_condition label
    def _prepare_thr(thr_in: pd.DataFrame, ice_label: str,
                     suffix: str) -> pd.DataFrame:
        """
        Extract thresholds where thr_in['ice_condition'] == ice_label
        and return a frame with string keys and renamed columns:
          wse_std_threshold{suffix}
          wse_u_threshold{suffix}
          xtrk_dist_threshold{suffix}
          suffix: _ifree, _icov, or _both
        """
        t = thr_in[thr_in['ice_condition'] == ice_label][[
            'crid_scenario', 'pass_id',
            'wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold'
        ]].copy()
        t['_crid_scenario_str'] = t['crid_scenario'].astype(str)
        t['_pass_id_str']       = t['pass_id'].astype(str)
        # Drop the original key columns; keep only string keys and renamed thresholds
        t = t.drop(columns=['crid_scenario', 'pass_id'])
        t = t.rename(columns={
            'wse_std_threshold':   f'wse_std_threshold{suffix}',
            'wse_u_threshold':     f'wse_u_threshold{suffix}',
            'xtrk_dist_threshold': f'xtrk_dist_threshold{suffix}',
        })
        return t

    # ---------------------------------------------------------------------
    # Bring in three sets of thresholds: ice-free, ice-covered, and both
    # ---------------------------------------------------------------------
    thr_ifree = _prepare_thr(thr, 'ice-free',   '_ifree')
    thr_icov  = _prepare_thr(thr, 'ice-covered','_icov')
    thr_both  = _prepare_thr(thr, 'both',       '_both')

    # Merge all three onto df (left-joins by string keys)
    df = df.merge(thr_ifree, on=['_crid_scenario_str', '_pass_id_str'], how='left')
    df = df.merge(thr_icov,  on=['_crid_scenario_str', '_pass_id_str'], how='left')
    df = df.merge(thr_both,  on=['_crid_scenario_str', '_pass_id_str'], how='left')

    # ---------------------------------------------------------------------
    # Select per-metric effective thresholds based on rules and row ice state
    # ---------------------------------------------------------------------
    is_free = df['ice_condition'].eq('ice-free') #ice_clim_f < 2

    def _select_threshold_per_metric(ifree_col: str, icov_col: str, both_col: str,
                                     rule_ifree: str, rule_icov: str):
        """
        Construct the effective threshold Series and an 'apply' boolean Series
        for one metric, given the columns for ifree/icov/both and the rules
        for ice-free rows and ice-affected rows.

        Returns
        -------
        eff : pd.Series (float)
            Selected threshold per row (if NaN, will be handled later).
        apply_flag : pd.Series (bool)
            Whether this metric gates the row or not.
        """
        eff = pd.Series(np.nan, index=df.index, dtype='float64')

        # Row-group: ice-free
        m = is_free
        if rule_ifree == 'ice-free':
            eff.loc[m] = df.loc[m, ifree_col]
            apply_free = True
        elif rule_ifree == 'ice-covered':
            eff.loc[m] = df.loc[m, icov_col]
            apply_free = True
        elif rule_ifree == 'both':
            eff.loc[m] = df.loc[m, both_col]
            apply_free = True
        else:  # 'not apply'
            apply_free = False

        # Row-group: ice-affected (ice_clim_f >= 2)
        m = ~is_free
        if rule_icov == 'ice-free':
            eff.loc[m] = df.loc[m, ifree_col]
            apply_cov = True
        elif rule_icov == 'ice-covered':
            eff.loc[m] = df.loc[m, icov_col]
            apply_cov = True
        elif rule_icov == 'both':
            eff.loc[m] = df.loc[m, both_col]
            apply_cov = True
        else:  # 'not apply'
            apply_cov = False

        apply_flag = pd.Series(False, index=df.index)
        apply_flag.loc[ is_free] = apply_free
        apply_flag.loc[~is_free] = apply_cov
        return eff, apply_flag #thresholds, and whether applicable

    # Select effective thresholds + apply flags for each metric
    wse_std_eff, apply_std = _select_threshold_per_metric(
        'wse_std_threshold_ifree', 'wse_std_threshold_icov', 'wse_std_threshold_both',
        rules_for_ice_free_data[0], rules_for_ice_covered_data[0]
    )
    wse_u_eff, apply_u = _select_threshold_per_metric(
        'wse_u_threshold_ifree', 'wse_u_threshold_icov', 'wse_u_threshold_both',
        rules_for_ice_free_data[1], rules_for_ice_covered_data[1]
    )
    xtrk_eff, apply_x = _select_threshold_per_metric(
        'xtrk_dist_threshold_ifree', 'xtrk_dist_threshold_icov', 'xtrk_dist_threshold_both',
        rules_for_ice_free_data[2], rules_for_ice_covered_data[2]
    )

    # Store “effective” thresholds (we will adjust them below for NaNs and ice overrides)
    df['wse_std_threshold_eff']   = pd.to_numeric(wse_std_eff, errors='coerce')
    df['wse_u_threshold_eff']     = pd.to_numeric(wse_u_eff,   errors='coerce')
    df['xtrk_dist_threshold_eff'] = pd.to_numeric(xtrk_eff,    errors='coerce')

    # ---------------------------------------------------------------------
    # No fallout here, but if a selected threshold is NaN AND the metric applies,
    # replace with a bounds-based default:
    #      - wse_std, wse_u -> max(bounds)
    #      - xtrk_dist      -> min(bounds)
    # ---------------------------------------------------------------------
    if apply_std.any():
        df.loc[apply_std & df['wse_std_threshold_eff'].isna(), 'wse_std_threshold_eff'] = max(wse_std_threshold_bounds)
    if apply_u.any():
        df.loc[apply_u & df['wse_u_threshold_eff'].isna(), 'wse_u_threshold_eff'] = max(wse_u_threshold_bounds)
    if apply_x.any():
        df.loc[apply_x & df['xtrk_dist_threshold_eff'].isna(),'xtrk_dist_threshold_eff'] = min(xtrk_dist_threshold_bounds)

    # ---------------------------------------------------------------------
    # Ice override: For ice-affected rows, bump wse_std/u thresholds up to minima
    # (if the metric applies and a value exists), using prior conservative mask (>= 1).
    # ---------------------------------------------------------------------
    ice_cov_mask = df['ice_clim_f'] >= 1  # NOTE: used >=1 to be more conservative than >=2
    bump_std_mask = ice_cov_mask & apply_std & df['wse_std_threshold_eff'].notna()
    bump_u_mask   = ice_cov_mask & apply_u   & df['wse_u_threshold_eff'].notna()
    df.loc[bump_std_mask & (df['wse_std_threshold_eff'] < wse_std_ice_min), 'wse_std_threshold_eff'] = wse_std_ice_min
    df.loc[bump_u_mask   & (df['wse_u_threshold_eff']   < wse_u_ice_min),   'wse_u_threshold_eff']   = wse_u_ice_min

    # ---------------------------------------------------------------------
    # Build final gating conditions
    # (coerce numeric for safety; NaNs in applied metrics → row fails)
    # ---------------------------------------------------------------------
    for c in ['wse_std', 'wse_u', 'xtrk_dist',
              'wse_std_threshold_eff', 'wse_u_threshold_eff', 'xtrk_dist_threshold_eff']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Per-metric pass/fail (if a metric doesn't apply, it's treated as True)
    cond_std  = (~apply_std) | (df['wse_std'] <= df['wse_std_threshold_eff'])
    cond_u    = (~apply_u)   | (df['wse_u']   <= df['wse_u_threshold_eff'])
    cond_xtrk = (~apply_x)   | (df['xtrk_dist'].abs() >= df['xtrk_dist_threshold_eff'])

    keep_mask = cond_std & cond_u & cond_xtrk

    # ---------------------------------------------------------------------
    # Cleanup & return only original df columns for rows that pass
    # ---------------------------------------------------------------------
    df.drop(columns=[
        '_crid_scenario_str', '_pass_id_str',
        'wse_std_threshold_ifree', 'wse_u_threshold_ifree', 'xtrk_dist_threshold_ifree',
        'wse_std_threshold_icov',  'wse_u_threshold_icov',  'xtrk_dist_threshold_icov',
        'wse_std_threshold_both',  'wse_u_threshold_both',  'xtrk_dist_threshold_both',
        'wse_std_threshold_eff', 'wse_u_threshold_eff', 'xtrk_dist_threshold_eff'
    ], inplace=True, errors='ignore')
    # Only return columns in the original df. 
    return df.loc[keep_mask, original_columns]

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
        by_crid_scenario (bool): Whether to match by CRID scenario (e.g., PIC2_or_PID0 vs early_versions)
        multiplier (float): Multiplier for IQR in Tukey outlier filtering
        lower_q (float): Lower quantile to compute IQR
        upper_q (float): Upper quantile to compute IQR
        used_q (str): Which bound(s) to use: 'upper', 'lower', or 'both'
        filter_by (str): Which variable(s) to filter by: 'area', 'wse', or 'both'

    Returns:
        pd.DataFrame: Filtered DataFrame with ice-covered outliers removed,
                      preserving original row order and all original columns.
                      
    Logic: LakeSP observations tend to be more uncertain during freeze-up periods. This function provides
    an option to compare freeze-up observations (in area_total or WSE) with ice-free observations, and 
    remove possible errors during the freeze-up period. 
    
    Caution: Some reservoirs can experience significant water level draw-downs during the freeze-up period. 
    So caveats are needed when using this function to remove negative anomalies, which could be true signals.
    Therefore, we provide an option for the filtering direction ("used_q"), and we recommend using filtering
    positive anomalies as high lake water area or WSE during the freeze-up period are less likely and are probably errors. 
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    original_columns = df.columns.tolist()

    # Add a derived column: CRID scenario (used for grouping)
    df['crid_scenario'] = df['crid'].apply(lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions')

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

    # NEW on 08/30/2025: Exclude interior days of major gaps in the ORIGINAL gauge data
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
    Remove rows from df_eval whose timestamps fall inside the *large* gaps
    between consecutive timestamps in df_apply.

    A “large gap” is defined as any interval between two consecutive, unique,
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
    
    # NEW: Bound overrides for applied thresholds_df
    wse_std_threshold_bounds = [0, 3],
    wse_u_threshold_bounds   = [0, 0.5],
    xtrk_dist_threshold_bounds = [0, 75000],
    
    # Ice overrides for applied thresholds on ice-affected rows
    wse_std_ice_min=3,
    wse_u_ice_min=0.1,
    
    allow_major_gap = 'no', # 'yes/no', to include whether we allow data gap in the filtered time series.
    max_temporal_gap = 95, #Maximum temporal gap (days) for filtering
    min_temporal_range = 365, # Minimum temporal range (days) for filtering    
    
    # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
    # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
    rules_for_ice_free_data=['ice-free', 'ice-free', 'not apply'],
    rules_for_ice_covered_data=['ice-free', 'ice-free', 'not apply'],       
    
    gauge_df = None, # enter gauge_df; None if no gauge data is available. 
    plot_period = ["2023-07-21T00:00:00Z", "2025-07-01T00:00:00Z"], #Defining start and end time for plotting. 
    
    apply_low_pass_filter = 'yes', #'yes' strongly recommended
    evaluating_at_full_data = 'no', #'no' recommended
    r2_filter = 'yes', #'yes' recommended    
    filter_type = 'savgol', #lowess, wavelet, savgol, kalman, spline, median, hampel.
    z_score_thresholds = [2.576, 3.5], #z-score thresholds for 1st and 2nd rounds of low-pass filters, respectively. 
                                       #2.576(99% for two tails), 2.807(99.5%), 2.967(99.7%), 3.291(99.9%), 3.5(99.95%)
    maximum_residual_spreads = [0.08, 0.06], #max residual spreads for 1st and 2nd rounds of low-pass filters, respectively.
    show_filtering_evolution = 'no' #for visualization only; caution: 'yes' may load many figures at the end of the script execution.
):  
    """
    This function applies the previously calibrated heuristic thresholds to iteratively clean up the input SP time series.
    
    Review of the procedure for time series filtering (refer back to "Step 2" in the main script):        
        2.1: Baseline filtering:
            > The calibrated heuristic thresholds are applied to the initial SP time series to retrieve the baseline subset.
        2.2: Iterative low-pass filtering:
            > A low-pass filter (e.g., LOWESS or Savitzky–Golay) is then fitted to the baseline, but evaluated against 
             the initial SP time series to identify and remove noises.
            > This procedure is repeated iteratively until convergence criteria are satisfied.
        Flexible parameter settings are supported throughout the process.
        
    Parameters:
        df_eval (DataFrame): Initial SP time series
        df_heuristic_thresholds (DataFrame): Calibrated heuristic thresholds, output from calibrate_heuristic_thresholds
        wse_std_threshold_bounds (list): [min, max] bounds when applying wse_std threshold
        wse_u_threshold_bounds (list): [min, max] bounds when applying wse_u threshold
        xtrk_dist_threshold_bounds (list): [min, max] bounds when applying abs(xtrk_dist) threshold
        wse_std_ice_min, wse_u_ice_min (float): Ice overrides for applied thresholds on ice-affected rows
        allow_major_gap: 'yes/'no', to include whether we allow data gap in the filtered time series.
        max_temporal_gap: Maximum allowable temporal gap (in days) in the baseline time series for filtering; 
            The filtering process is abandoned if this gap is exceeded. 
            This parameter only applies if allow_major_gap is set to 'no'. 
        min_temporal_range: Minimum required temporal range (in days) in the baseline time series for filtering; 
            The filtering process is abandoned if this range is not met.  
            This parameter only applies if allow_major_gap is set to 'no'. 
        rules_for_ice_free_data, rules_for_ice_covered_data: Rules of threshold application per metric:
            - rules_for_ice_free_data    : used when the row is actually ice-free (ice_clim_f < 2)
            - rules_for_ice_covered_data : used when the row is actually ice-covered (ice_clim_f >= 2)
        gauge_df (DataFrame): Gauge time series, if available; None if not.
        plot_period: Starting and ending time for plotting
            - [0] start_time (yyyy-mm-ddThh:mm:ssZ): Starting time
            - [1] end_time (yyyy-mm-ddThh:mm:ssZ):   Ending time                       
        apply_low_pass_filter (text):
            - "yes" = both baseline (Step 2.1) and low-pass (Step 2.2) filters will be executed;
            - "no" = only bechmark filter (Step 2.1) will be executed. 
        
        The following parameters only matter if apply_low_pass_filter is set to "yes":
        evaluating_at_full_data (text):    "yes" = evaluate outlier removal (z-score clipping) on full LakeSP data; 
                                           "no" = evaluate only on selected observations (the benmark)       
        r2_filter (text):                  "yes" = perform another round (round-2) of filtering to remove remaining noise; "no" = otherwise                                     
        filter_type (text):                Low-pass filter type: lowess, wavelet, savgol, kalman, spline, median, and hampel.
        z_score_thresholds:                Z-score threshols
            [0](float):                       For round-1 (more aggressive) filtering
            [1] (float):                      For round-2 (less aggressive) filtering
        maximum_residual_spreads:          A tolerance of maximum relative residual that is not considered to be an outlier
            [0] (float):                      For round-1 filtering 
            [1] (float):                      For round-2 filtering 
        show_filtering_evolution (text):   "yes" = plot how outlier filtering evolves through iteration; "no" = otherwise

    Returns: 
        df_eval (DataFrame): Filtered SP time series 
        [n_while, n_while_r2]
           - n_while (Integer): Number of iterations for round-1 low-pass filtering           
           - n_while_r2 (Integer): Number of iterations for round-2 low-pass filtering
                For both n_while and n_while_r2, the following discrete values apply:
                  -9: Indicates the original SP input (df_eval) is empty and the filter is not applicable
                  -2: This round of filter is turned off. 
                  0:  Indicates df_eval became empty after baseline subsetting (no good observations to initiate the low-pass filtering)
                  -1: Indicates the iteration started but was abandoned
                  >0: Indicates the number of regular iterations
        filter_status [text]: scenarios of filtered result
            - no data
            - heuristic baseline: low-pass filter turned off
            - fail
            - success
    """    
    # In case the input time series DataFrame is empty. 
    if df_eval.empty:
        return df_eval.copy(), [-9, -9], 'no data'
    
    # Freeze the initial df_eval to df
    df = df_eval.copy()
    
    # Initialize df_eval for filter update (safety measure applied)
    df_eval = df_eval.copy() # This will be updated through filtering. 
    start_time = plot_period[0]
    end_time = plot_period[1]
    
    
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
        
        return df_eval, [n_while, n_while_r2], 'heuristic baseline'   
    
    else:
        # execute both steps     
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
        n_while = 0  # Initialize r1 loop/iteration times (turned on)    
        initial_length = len(df_eval) # In case this is zero (after applying first baseline subsetting), the "while" statement won't run. 
        updated_length = 0 # Initialize the length of the updated df_eval    
        minimum_data_n = float(min_temporal_range/max_temporal_gap)+1 # the minimum number of data points considered to be acceptable.  
        lowess_QA_check = 'check' # Initialize a QA check for the lowess filter. This is only relevant if filter_type is set to 'lowess'. 
        while (updated_length < initial_length) and (n_while < 40): # All conditions must be satisfied        
            initial_length = len(df_eval) # Note: df_eval is updated per iteration. 
                   
            # Apply heuristic thresholds to generate the "heuristic baseline" (i.e., good-quality observations for filter application). 
            df_apply = apply_heuristic_thresholds(df_eval, df_heuristic_thresholds, 
                                                  wse_std_threshold_bounds = wse_std_threshold_bounds,
                                                  wse_u_threshold_bounds   = wse_u_threshold_bounds,
                                                  xtrk_dist_threshold_bounds = xtrk_dist_threshold_bounds,
                                                  wse_std_ice_min = wse_std_ice_min, wse_u_ice_min = wse_u_ice_min, 
                                                  rules_for_ice_free_data   = rules_for_ice_free_data, #(per-metric: [wse_std, wse_u, xtrk_dist])
                                                  rules_for_ice_covered_data= rules_for_ice_covered_data)         
            # Remove bad crossover calibration, although this is redundant for PIC2 and PID0 as quality_f < 3 precludes xovr_cal_q = 2 (see bitwise definition)
            df_apply = df_apply[df_apply['xovr_cal_q'] < 2]  
            # Remove bad observations flagged in PIC2 and PID0: specular_rining_bad, xovr_cal_bad, and low_coh_bad. 
            df_apply = df_apply[df_apply['quality_f'] < 3] 
            
            # Truncate df_eval to the same time range of df_apply to avoid extrapolation
            tmin = df_apply['time'].min()
            tmax = df_apply['time'].max()
            df_eval = df_eval.loc[df_eval['time'].between(tmin, tmax, inclusive='both')] #works if df_apply is empty. 
            
            ####
            #mask = (df_apply["xtrk_dist"].abs() >= 10000) & (df_apply["xtrk_dist"].abs() <= 60000)
            #df_apply = df_apply[mask]   
            ####      
           
            # This lake is abandoned if any of the scenarios occurs any time:
            # 1. The number of data points in the baseline time series (df_apply) is too limited to yield reliable pattern (i.e., size < minimum_data_n)
            # If allow_major_gap == 'no', meaning we do not allow major gaps in df_apply. 
            #    2. The time series in df_apply (not df_eval!) has major temporal gaps (i.e., > max_temporal_gap, such as 3-4 months or a hydroclimate season).
            #    3. The time range of df_apply (equivalent to that of df_eval after truncation above) is too short (e.g., < 1 year)     
            # If allow_major_gap == 'yes', meaning we allow major gaps in df_apply. 
            #    This is a lenient scenario; yet for uncertainty control, we discard all df_eval data failling in the major gaps, and 
            #    but do not enforce the time range (although the number of data points in the resulting df_eval must be sufficient).
            if len(df_apply) < minimum_data_n: #data points too sparse to yield reliable pattern. 
                df_eval = df_eval.iloc[0:0] # Clear up de_eval
                n_while = -1 # -1 indicates this lake is abandoned.             
                break # break the while loop            
            else:
                exceeds_limit = (df_apply['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
                if allow_major_gap == 'no': # do not allow major gaps in df_apply or in df_eval   
                    if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                        df_eval = df_eval.iloc[0:0] # Clear up de_eval
                        n_while = -1 # -1 indicates this lake is abandoned.             
                        break # break the while loop
                    else: # Check if the time range of df_apply is too short (1 year)
                        if (df_apply['datetime'].max() - df_apply['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
                            df_eval = df_eval.iloc[0:0] # Clear up de_eval  
                            n_while = -1 # -1 indicates this lake is abandoned. 
                            break # break the while loop
                else: # Remove observations in df_eval that fall within any gap longer than max_temporal_gap in df_apply
                    if exceeds_limit.any(): # Check if any time difference exceeds max_temporal_gap
                        df_eval = drop_eval_in_apply_gaps(df_eval, df_apply, max_temporal_gap, 'datetime') 
                        # since df_apply size >= minimum_data_n, df_eval after gap removal still >= minimum_data_n. 
            
            
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
            abs_residual_p = np.abs(df_eval['residuals']) / ( np.max(df_apply['wse'])  - np.min(df_apply['wse']) )
            # Define mask based on combined conditions
            mask = (np.abs(z_scores) < z_score_thresholds[0]) | (abs_residual_p < maximum_residual_spreads[0])
            # Apply mask to filter df
            df_eval = df_eval[mask] # Update df_eval by removing outliers for the next iteration. 
                    
            # Remove positive anomalies during the freeze-up/ice-covered period 
            # First by area_total and/or wse. Set "by_pass" to be True because area_total is pass dependent.
            # The multiplier is set higher to de-risk over-rejection due to limited observations per pass. 
            df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=True, by_crid_scenario=False,
                                    multiplier=0.3, lower_q=0, upper_q=1, used_q='upper', filter_by='both')  #filter_by='area'   
            # Second by wse. Set "by_pass" to be False to make the removal more general if possible (to avoid over-rejection)
            # This second removal may be necessary as pass-specific outliers may remain if there's no ice-free observation for that pass.
            df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=False, by_crid_scenario=False,
                                    multiplier=0.2, lower_q=0, upper_q=1, used_q='upper', filter_by='wse') #area, wse, or both
            #Note: Users can optimize their "filter_by" and "pass_by" parameters. 
            
            # Remove remaining isolated extreme outliers using Tukey method (IQR method) 
            # Use 10th and 90th percentile.
            df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse', multiplier=3, lower_q=0.1, upper_q=0.9)
            
            # Further remove observations that are still 150 m higher than the median WSE
            # Typical range for large reservoirs: 10–60 meters (e.g., the Three Gorges Reservoir ranges in 145-175 m). 
            # Very large reservoirs (e.g., hydropower or multipurpose dams): can exceed 100 meters
            # A few massive reservoirs may approach or even exceed 150–300 meters in water level fluctuation:
            # e.g., China’s Jinping-I Dam, 305 m; [4610062383, 4610049903]; 
            # According to GDW, quite a few dams are higher than 200-300 m. 
            df_eval = df_eval[ np.abs(df_eval['wse'] - np.median(df_eval['wse'])) <= 150 ]
            # Note: this works if df_eval is empty.
            
            # Plot filter evolution if preferred. Caution: this will generate a series of plot (one per iteration)
            if show_filtering_evolution == 'yes': # Show how the outlier removal evolves through iteraction
                plt.rcParams["font.family"] = "Arial"
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.grid(True, linewidth=0.5, zorder=1)
                
                # Plot gauge measurements if the lake has gauge data
                if gauge_df is not None:
                    # Compute a preliminary datum bias between SWOT and gauge measurements.
                    # Note this bias correction is preliminary and is only intended here for visualization                
                    bias_swot_gauge_prelim = np.nanmedian(gauge_df['gauge_wse']) - np.nanmedian(df['wse'])               
                    ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse'] - bias_swot_gauge_prelim, \
                            label='gauge', color='green', marker = 'o', markersize=6, linestyle='--') # Shift gauge to SWOT datum. 
                
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
        if len(df_eval) < minimum_data_n: #data points too sparse to yield reliable pattern. 
            df_eval = df_eval.iloc[0:0] # Clear up de_eval
            n_while = -1 # -1 indicates this lake is abandoned.                         
        else:
            exceeds_limit = (df_eval['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
            if allow_major_gap == 'no': # do not allow major gaps in df_eval  
                if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                    df_eval = df_eval.iloc[0:0] # Clear up de_eval
                    n_while = -1 # -1 indicates this lake is abandoned.             
                else: # Check if the time range of df_eval is too short (1 year)
                    if (df_eval['datetime'].max() - df_eval['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
                        df_eval = df_eval.iloc[0:0] # Clear up de_eval  
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
            # Apply stricter quality control: retain only observations flagged as "good" by CNES baseline quality flags
            df_good_quality = df[(df['xovr_cal_q'] == 0) & (df['quality_f'] == 0) & (df['ice_clim_f'] == 0)] 
            
            ####
            #mask = (df_good_quality["xtrk_dist"].abs() >= 10000) & (df_good_quality["xtrk_dist"].abs() <= 60000)
            #df_good_quality = df_good_quality[mask]   
            ####
            
            # Further apply heuristic thresholds (no ice period this time)
            df_good_quality = apply_heuristic_thresholds(df_good_quality, df_heuristic_thresholds, 
                                                         wse_std_threshold_bounds = wse_std_threshold_bounds,
                                                         wse_u_threshold_bounds   = wse_u_threshold_bounds,
                                                         xtrk_dist_threshold_bounds = xtrk_dist_threshold_bounds,
                                                         wse_std_ice_min = wse_std_ice_min, wse_u_ice_min = wse_u_ice_min, 
                                                         rules_for_ice_free_data   = rules_for_ice_free_data,  #(per-metric: [wse_std, wse_u, xtrk_dist])
                                                         rules_for_ice_covered_data= rules_for_ice_covered_data)         
       
            # Identify high-quality observations not already present in df_eval based on the index_col
            df_to_recover = df_good_quality[~df_good_quality['index_col'].isin(df_eval['index_col'])]
            
            # Append these recovered observations to df_eval
            df_eval_locked = df_eval.copy() # Lcoecked df_eval from round 1 -----------------------------
            df_eval = pd.concat([df_eval, df_to_recover], ignore_index=True)
            
            # Sort df_eval by high-precision datetime to maintain chronological order        
            df_eval = df_eval.sort_values('datetime', kind='mergesort').reset_index(drop=True) #mergesort keeps relative order for ties. 
            # Line below was deprecated because index_col is based on the original lakeSP time series which may have been time-sorted.
            #     df_eval = df_eval.sort_values(by='index_col').reset_index(drop=True)        
            # Note the code above is safe even when df_good_quality is empty.
        
        
            # -----Run a round-2, less aggressive filtering-----
            # The logic is consistent with round 1, except that the filter is applied and evaluated on the same data: df_eval.
            initial_length = len(df_eval) # In case this is zero, the "while" statement won't run. 
            updated_length = 0  # Initialize the length of the updated df_eval                
            while (updated_length < initial_length) and (n_while_r2 < 5): # A max of 10 iteration times to avoid over-rejection in round-2 filtering. 
                initial_length = len(df_eval) # Note: df_eval is updated per iteration.   
                
                # This lake is abandoned if any of the scenarios occurs any time:
                # 1. The number of data points in the baseline time series (df_eval) is too limited to yield reliable pattern (i.e., size < minimum_data_n)
                # If allow_major_gap == 'no', meaning we do not allow major gaps in df_eval. 
                #    2. The time series in df_eval has major temporal gaps (i.e., > max_temporal_gap, such as 3-4 months or a hydroclimate season).
                #    3. The time range of df_eval is too short (e.g., < 1 year)     
                # If allow_major_gap == 'yes', meaning we allow major gaps in df_eval. 
                #    This is a lenient scenario; we do not enforce the time range (but the number of data points in df_eval must be sufficient).        
                if len(df_eval) < minimum_data_n: #data points too sparse to yield reliable pattern. 
                    df_eval = df_eval.iloc[0:0] # Clear up de_eval
                    n_while_r2 = -1 # -1 indicates this lake is abandoned.             
                    break # break the while loop            
                else:
                    exceeds_limit = (df_eval['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
                    if allow_major_gap == 'no': # do not allow major gaps in df_eval or in df_eval   
                        if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                            df_eval = df_eval.iloc[0:0] # Clear up de_eval
                            n_while_r2 = -1 # -1 indicates this lake is abandoned.             
                            break # break the while loop
                        else: # Check if the time range of df_eval is too short (1 year)
                            if (df_eval['datetime'].max() - df_eval['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
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
                abs_residual_p = np.abs(df_eval['residuals']) / ( np.max(df_eval['wse']) - np.min(df_eval['wse']) )
                # Define mask based on combined conditions
                mask = (np.abs(z_scores) < z_score_thresholds[1]) | (abs_residual_p < maximum_residual_spreads[1])                
                # Apply mask to filter df
                df_eval = df_eval[mask] # Update df_eval by removing outliers for the next iteration.                 
                    
                # Remove positive anomalies during the freeze-up/ice-covered period
                # First by area_total and/or wse. Set "by_pass" to be True because area_total is pass dependent.
                # The multiplier is set higher to de-risk over-rejection due to limited observations per pass.
                df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=True, by_crid_scenario=False,
                                        multiplier=0.3, lower_q=0, upper_q=1, used_q='upper', filter_by='both')  #filter_by='area'   
                # Second by wse. Set "by_pass" to be False to make the removal more general if possible (to avoid over-rejection)
                # This second removal may be necessary as pass-specific outliers may remain if there's no ice-free observation for that pass.
                df_eval = filter_ice_outliers(df_eval, remove_tukey_outliers, by_pass=False, by_crid_scenario=False,
                                        multiplier=0.2, lower_q=0, upper_q=1, used_q='upper', filter_by='wse') #area, wse, or both
                #Note: Users can optimize their "filter_by" and "pass_by" parameters. 
                
                # Remove remaining isolated outliers using Tukey method (IQR method)    
                # Use 10th and 90th percentile.
                df_eval, _, _ = remove_tukey_outliers(df_eval, col='wse', multiplier=3, lower_q=0.1, upper_q=0.9)
                
                # Further remove observations that are still 150 m higher than the median WSE
                # Typical range for large reservoirs: 10–60 meters (e.g., the Three Gorges Reservoir ranges in 145-175 m). 
                # Very large reservoirs (e.g., hydropower or multipurpose dams): can exceed 100 meters
                # A few massive reservoirs may approach or even exceed 150–200 meters in water level fluctuation.
                df_eval = df_eval[ np.abs(df_eval['wse'] - np.median(df_eval['wse'])) <= 150 ]
                # Note: this works if df_eval is empty.
                
                # Apply lock scheme for round-1 result ------------------------------------
                # Recover observations from df_eval_locked that are removed from df_eval (based on index_col). 
                df_to_recover_locked = df_eval_locked[~df_eval_locked['index_col'].isin(df_eval['index_col'])]               
                df_eval = pd.concat([df_eval, df_to_recover_locked], ignore_index=True)      
                df_eval = df_eval.sort_values('datetime', kind='mergesort').reset_index(drop=True) #mergesor        
                
                
                # Plot filter evolution if preferred. Caution: this will generate a series of plot (one per iteration)
                if show_filtering_evolution == 'yes': # Show how the outlier removal evolves through iteraction
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
                df_eval = df_eval.iloc[0:0] # Clear up de_eval
                n_while_r2 = -1 # -1 indicates this lake is abandoned.                       
            else:
                exceeds_limit = (df_eval['datetime'].diff()) > pd.Timedelta(days=max_temporal_gap) # Check if any time difference exceeds max_temporal_gap
                if allow_major_gap == 'no': # do not allow major gaps in df_eval or in df_eval   
                    if exceeds_limit.any(): # If any gap exceeds max_temporal_gap
                        df_eval = df_eval.iloc[0:0] # Clear up de_eval
                        n_while_r2 = -1 # -1 indicates this lake is abandoned.             
                    else: # Check if the time range of df_eval is too short (1 year)
                        if (df_eval['datetime'].max() - df_eval['datetime'].min()) < pd.Timedelta(days=min_temporal_range):
                            df_eval = df_eval.iloc[0:0] # Clear up de_eval  
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
        
        return df_eval, [n_while, n_while_r2], filter_status

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
        df_eval (DataFrame): Initial SP time series
        baseline_SQL (str): 
            pandas .query expression; rows meeting this are used as the baseline time series.
        multiplier, lower_q, upper_q: Inputs for the remove_tukey_outliers function (see detailed in remove_tukey_outliers)
        iteration_n (Integer): Maximum number of iterations for Tukey outlier removal         

    Returns: 
        df_eval (DataFrame): Filtered SP time series 
        n_while (Integer): Number of iterations for Tukey noise removal           
            -9: Indicates the original SP input (df_eval) is empty and the filter is not applicable 
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
        df_eval = df_eval.iloc[0:0] # Clear up de_eval  
        n_while = -1 # -1 indicates this lake is abandoned. 
    
    if n_while <= 0: 
        filter_status = 'fail'
    else: # n_while > 0:
        filter_status = 'success'
    
    return df_eval, n_while, filter_status   
    
def sp_cycle_adjustment(df_eval):
    """
    This function is to reduce intra-cycle WSE inconsistencies caused by multiple orbit passes.
    For large lakes spanning multiple SWOT orbit passes, WSE values within the same orbit cycle may show substantial 
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
        df_eval (DataFrame): Initial SP time series    
        
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
    inter_cycle_std = df_option1['wse_cycle_avg'].std() # Computed as the standard devaition of cycle-averaged WSEs
    
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
        
        # Run another tukey outlier removal on df_eval.wse_adjusted
        df_option3, _, _ = remove_tukey_outliers(df_eval, col='wse_adjusted', multiplier=3, lower_q=0.1, upper_q=0.9)
        
    else: # Return option 2 and option 3 both as df_eval as neither option is applied. 
        df_option2 = df_eval.copy()
        df_option3 = df_eval.copy()
    
    return df_option1, df_option2, df_option3
        
        
    