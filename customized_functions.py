# -*- coding: utf-8 -*-
"""
Customized function module
Created: 04/20/2025
Last updated: 08/11/2025
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
#import seaborn as sns

"""
Functions: Do not change the functions unless necessary. 
    Basic functions:
        compute_rmse:              Computes root mean squared error (RMSE), np.nan robust. 
        compute_correlation:       Computes Pearson or Spearman correlation coefficient
        remove_tukey_outliers:     Removes outliers using a generalized Tukey method (IQR-based).
        calibrate_heuristic_thresholds: Calibrate heuristic thresholds (max wse_std, max wse_u, and min xtrk_dist) before SP filtering.
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

# Modified on 08/04/2025 to include the heuristic threshold for xtrk_dist. 
def calibrate_heuristic_thresholds(df,
                                   by_crid_scenario=[True, True, True],
                                   by_pass_id=[True, True, True],
                                   wse_std_threshold_minmax=[0, 3],
                                   wse_u_threshold_minmax=[0.1, 0.5],
                                   xtrk_dist_threshold_minmax=[0, 75000]):
    """
    Calibrates heuristic thresholds for wse_std, wse_u, and xtrk_dist,
    optionally grouped by CRID scenario and/or pass_id for each variable.
    The returned DataFrame ALWAYS contains both 'crid_scenario'
    and 'pass_id' columns (plus 'lake_id' and the three thresholds), even if the
    grouping for a particular metric does not use one or both keys.
    
    Note: 
        • fill_value (-999999999999) in each metric has been replaced by NaN.
        • If a metric is *not* grouped by a key, its threshold is computed at the
            level of the remaining keys (or at global), and then broadcast to all
            (crid_scenario, pass_id) rows via a left-merge.
        • Pandas aggregations ignore NaNs by default (e.g., max/min skip NaN).
        • xtrk_dist thresholds are computed on abs(xtrk_dist).

    Parameters:
        df (pd.DataFrame): Input dataframe containing at least the following columns:
            ['lake_id', 'crid', 'pass_id', 'wse_std', 'wse_u', 'xtrk_dist']
        by_crid_scenario (list of bool): Controls grouping for each metric [wse_std, wse_u, xtrk_dist].
            If True, use crid_scenario for grouping.
        by_pass_id (list of bool): Controls grouping for each metric [wse_std, wse_u, xtrk_dist]
            If True, use pass_id for grouping
        wse_std_threshold_minmax (list): [min, max] bounds for wse_std threshold
        wse_u_threshold_minmax (list): [min, max] bounds for wse_u threshold
        xtrk_dist_threshold_minmax (list): [min, max] bounds for abs(xtrk_dist) threshold
        
        note: xtrk_dist (in m, valid max: 75000 m): Distance of the lake polygon centroid from the spacecraft nadir track; 
            this value is computed using a local spherical Earth approximation. 
            A negative value indicates that the lake is on the left side of the swath, relative to the spacecraft velocity vector. 
            A positive value indicates that the lake is on the right side of the swath.
            So, absolute value is used for simplicity. 
    
    Returns:
        pd.DataFrame: A threshold table including:
            ['lake_id', 'crid_scenario', 'pass_id', 'wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold'].
            > lake_id: PLD lake id of the input df. 
            > crid_scenario: "PIC2_or_PID0" or "early_versions" (e.g., PIC0, PGC0); or 'global' if df is empty
            > pass_id: SWOT orbit pass; or 'global' if df is empty
            > wse_std_threshold: the maximum wse_std threshold under this pass_id and crid_scenario combination
            > wse_u_threshold: the maximum wse_u threshold under this pass_id and crid_scenario combination   
            > xtrk_dist_threshold: the minimum abs(xtrk_dist) threshold under this pass_id and crid_scenario combination 
    """
    # Determine lake_id (assumed consistent within the input df)
    lake_id = df['lake_id'].iloc[0] if 'lake_id' in df.columns and not df.empty else 'unknown' #Uses 'unknown' if df is empty

    # Handle empty input
    # If there's no data, return a fallback threshold DataFrame using the global default thresholds.
    if df.empty:
        return pd.DataFrame([{
            'lake_id': lake_id,
            'crid_scenario': 'global', #still keep crid_scenario and pass_id attributes, but filled values by "global"
            'pass_id': 'global',
            'wse_std_threshold': max(wse_std_threshold_minmax),
            'wse_u_threshold': max(wse_u_threshold_minmax),
            'xtrk_dist_threshold': min(xtrk_dist_threshold_minmax) #use min for abs(xtrk_dist)
        }])

    df = df.copy() # Avoid modifying the original DataFrame.

    # Define crid_scenario values for grouping
    df['crid_scenario'] = df['crid'].apply(lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions')

    # Compute absolute cross-track distance for thresholding
    df['xtrk_dist_abs'] = df['xtrk_dist'].abs()

    # Helper function to decide grouping:
    # For each evaluated metric (wse_std, wse_u, xtrk_dist_abs, as indexed 0, 1, 2), determine which columns to group by.
    # If neither crid_scenario nor pass_id is chosen, it uses a fallback group called "global" (i.e., no grouping).
    def get_group_keys(metric_index):
        keys = []
        if by_crid_scenario[metric_index]: #if Ture
            keys.append('crid_scenario')
        if by_pass_id[metric_index]:
            keys.append('pass_id')
        return keys if keys else ['global']
    # Example: by_crid_scenario = [True, False, False]
    #          by_pass_id = [True, True, False]
    # when index = 0, the evaluated metric is wse_std, grouping keys returned: ['crid_scenario', 'pass_id']
    # when index = 1, the evaluated metric is wse_u, grouping keys teruend: just ['pass_id]
    # when index = 2, the evaluated metric is xtrk_dist, grouping keys returned: ['global]: no grouping will be applied. 

    # Prepare to compute thresholds separately for each metric
    thresholds = []
    # Threshold aggregation loop: handles all three metrics (wse_std, wse_u, xtrk_dist_abs) in a unified way.
    for metric_name, agg_col, minmax, idx, agg_func in zip(
        ['wse_std', 'wse_u', 'xtrk_dist_abs'],
        ['wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold'],
        [wse_std_threshold_minmax, wse_u_threshold_minmax, xtrk_dist_threshold_minmax],
        [0, 1, 2],
        ['max', 'max', 'min']  # NOTE: xtrk_dist uses min(abs())
    ):
        # Determine grouping strategy for this metric.
        group_keys = get_group_keys(idx)

        # If no grouping, create a dummy grouping key "global" so we can still .groupby().
        if group_keys == ['global']:
            df['global'] = 'global' # Attribute and its value are both "global" here. 

        # Perform aggregation: get the max (or min) value per group for the metric.
        grouped = df.groupby(group_keys).agg({metric_name: agg_func}).reset_index() # Pandas automatically ignores NaN. 
        # Clip values to user-defined min/max range
        grouped[agg_col] = grouped[metric_name].clip(lower=minmax[0], upper=minmax[1])
        # Drop the original unbounded column.
        grouped = grouped.drop(columns=[metric_name]) 
        # Save the group keys and threshold values for later merging.
        thresholds.append((group_keys, grouped))
    
    # Build base_df that ALAYS has both keys: using the data's actual combinations.
    base_df = df[['crid_scenario', 'pass_id']].drop_duplicates().copy()

    # Merge thresholds for each metric
    for group_keys, th_df in thresholds:
        if group_keys == ['global']:
            # Broadcast the global value(s) to ALL rows:
            #   align on a temp 'global' column present in both frames.
            base_df['global'] = 'global' # An intermediate column to be deleted later. 
            base_df = base_df.merge(th_df, on='global', how='left') #in this case, th_df already has a 'global' column.
            base_df.drop(columns=['global'], inplace=True)  #drop the "global" intermediate column.
        else:
            # Merge only on the keys that metric actually used
            base_df = base_df.merge(th_df, on=group_keys, how='left')

    # Fill any missing threshold cells with a "global fallback"
    # If a particular (crid_scenario, pass_id) had no data for a metric (e.g., all nan)
    # fill with the metric-level worst-case fallback (max for wse_*, min for xtrk).
    base_df['wse_std_threshold'] = base_df['wse_std_threshold'].fillna(max(wse_std_threshold_minmax))
    base_df['wse_u_threshold']   = base_df['wse_u_threshold'].fillna(max(wse_u_threshold_minmax))
    base_df['xtrk_dist_threshold'] = base_df['xtrk_dist_threshold'].fillna(min(xtrk_dist_threshold_minmax))

    # Insert lake_id & final column order (ALWAYS keep both keys)
    base_df.insert(0, 'lake_id', lake_id)
    return base_df[['lake_id', 'crid_scenario', 'pass_id',
                    'wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold']]

# Modified on 08/04/2025 to include the heuristic threshold for xtrk_dist. 
def apply_heuristic_thresholds(df, thresholds_df, wse_std_ice_min=3, wse_u_ice_min=0.5):
    """
    Apply heuristic thresholds (from calibrate_heuristic_thresholds) to rows in df.

    A row is kept if ALL are true:
      - wse_std              <= wse_std_threshold
      - wse_u                <= wse_u_threshold
      - abs(xtrk_dist)       >= xtrk_dist_threshold

    Threshold matching hierarchy (row-wise):
      > Full match on (crid_scenario, pass_id)
      > Fallback by crid_scenario only:
           - take max of wse_std_threshold, wse_u_threshold
           - take min of xtrk_dist_threshold
      > Fallback by pass_id only (same aggregation rules as above)
      > Global fallback:
           - wse_std_threshold, wse_u_threshold  -> global max
           - xtrk_dist_threshold                 -> global min
          
    In other words, Tries full match: crid_scenario + pass_id
    If unmatched:
        > Tries fallback by crid_scenario only
        > Or by pass_id only
    If still unmatched:
        > Applies global maximum of thresholds
        
    Special case (ice covered):
      If ice_clim_f > 0, cap the min thresholds to be wse_std_ice_min and wse_u_ice_min.

    Parameters
    df : pd.DataFrame
        Must include: ['crid', 'pass_id', 'wse_std', 'wse_u', 'xtrk_dist'].
        If present, 'ice_clim_f' is used to raise the minimum thresholds.
    thresholds_df : pd.DataFrame
        Output of calibrate_heuristic_thresholds, with columns including:
        ['crid_scenario', 'pass_id', 'wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold'].
        May also include a 'global' column if some metrics were ungrouped there.
    wse_std_ice_min : float
        Minimum wse_std threshold applied when ice_clim_f > 0.
    wse_u_ice_min : float
        Minimum wse_u threshold applied when ice_clim_f > 0.

    Returns: pd.DataFrame
        Filtered subset of df that meet the heuristic threshold criteria, keeping only the original columns.
    """   
    # Prevent modifying the original inputs. 
    original_columns = df.columns.tolist()
    df = df.copy()
    thr = thresholds_df.copy()

    # Compute crid_scenario in df (must match calibrate_heuristic_thresholds)
    df['crid_scenario'] = df['crid'].apply(lambda x: 'PIC2_or_PID0' if x in ['PIC2', 'PID0'] else 'early_versions')
   
    # Create temporary string keys for type-safe merge: this is needed as thresholds_df.pass_id can be "global", 
    # which does not match the integer type of df.pass_id, leading to a potential error. 
    # The temporary string keys will later be deleted. 
    df['_pass_id_str'] = df['pass_id'].astype(str)
    df['_crid_scenario_str'] = df['crid_scenario'].astype(str)

    # Keep only necessary fields from thresholds_df and build its string keys
    keep_cols = [c for c in thr.columns if c in
                 ['crid_scenario', 'pass_id', 'global',
                  'wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold']]
    thr = thr[keep_cols].copy()
    thr['_pass_id_str'] = thr['pass_id'].astype(str) if 'pass_id' in thr.columns else ''
    thr['_crid_scenario_str'] = thr['crid_scenario'].astype(str) if 'crid_scenario' in thr.columns else ''

    # First attempt: full match on (crid_scenario, pass_id)
    df = df.merge(
        thr,
        left_on=['_crid_scenario_str', '_pass_id_str'],
        right_on=['_crid_scenario_str', '_pass_id_str'],
        how='left',
        suffixes=('', '_thr') #This only matters if there are duplicate fields, which should not happen. 
    )

    # Identify rows needing fallback for ANY missing threshold
    need_fallback = (
        df['wse_std_threshold'].isna() |
        df['wse_u_threshold'].isna() |
        df['xtrk_dist_threshold'].isna()
    )
    if need_fallback.any():        
        # Fallback by crid_scenario:
        #     wse_* use max; xtrk_dist uses min (consistent with calibration semantics)
        by_crid = thr.groupby('crid_scenario', dropna=False).agg({
            'wse_std_threshold': 'max',
            'wse_u_threshold': 'max',
            'xtrk_dist_threshold': 'min'
        }).reset_index()

        # Fallback by pass_id (same aggregation rules)
        by_pass = thr.groupby('pass_id', dropna=False).agg({
            'wse_std_threshold': 'max',
            'wse_u_threshold': 'max',
            'xtrk_dist_threshold': 'min'
        }).reset_index()

        # Global fallback (max for wse_*, min for xtrk)
        global_max_std  = thr['wse_std_threshold'].max(skipna=True)
        global_max_u    = thr['wse_u_threshold'].max(skipna=True)
        global_min_xtrk = thr['xtrk_dist_threshold'].min(skipna=True)

        # Row-wise application of hierarchical fallbacks
        for idx in df.index[need_fallback]:
            r = df.loc[idx]
            std_val  = r.get('wse_std_threshold', np.nan)
            u_val    = r.get('wse_u_threshold', np.nan)
            x_val    = r.get('xtrk_dist_threshold', np.nan)

            # Fallback 1: by crid_scenario
            if pd.isna(std_val) or pd.isna(u_val) or pd.isna(x_val):
                m_crid = by_crid[by_crid['crid_scenario'] == r['crid_scenario']]
                if not m_crid.empty:
                    if pd.isna(std_val): std_val = m_crid['wse_std_threshold'].values[0]
                    if pd.isna(u_val):   u_val   = m_crid['wse_u_threshold'].values[0]
                    if pd.isna(x_val):   x_val   = m_crid['xtrk_dist_threshold'].values[0]

            # Fallback 2: by pass_id
            if pd.isna(std_val) or pd.isna(u_val) or pd.isna(x_val):
                m_pass = by_pass[by_pass['pass_id'] == r['pass_id']]
                if not m_pass.empty:
                    if pd.isna(std_val): std_val = m_pass['wse_std_threshold'].values[0]
                    if pd.isna(u_val):   u_val   = m_pass['wse_u_threshold'].values[0]
                    if pd.isna(x_val):   x_val   = m_pass['xtrk_dist_threshold'].values[0]

            # Fallback 3: global
            if pd.isna(std_val): std_val = global_max_std
            if pd.isna(u_val):   u_val   = global_max_u
            if pd.isna(x_val):   x_val   = global_min_xtrk

            df.at[idx, 'wse_std_threshold']   = std_val
            df.at[idx, 'wse_u_threshold']     = u_val
            df.at[idx, 'xtrk_dist_threshold'] = x_val

    # Special ice override (raise minimum wse thresholds where ice_clim_f > 0)
    if 'ice_clim_f' in df.columns:
        ice_mask = df['ice_clim_f'] > 0
        # Raise to at least the ice minimums
        df.loc[ice_mask & (df['wse_std_threshold'] < wse_std_ice_min), 'wse_std_threshold'] = wse_std_ice_min
        df.loc[ice_mask & (df['wse_u_threshold']   < wse_u_ice_min),   'wse_u_threshold']   = wse_u_ice_min

    # Clean temp keys
    df.drop(columns=['_pass_id_str', '_crid_scenario_str'], inplace=True, errors='ignore')

    # Ensure numeric types (robustness to strings, infinities, etc.)
    for c in ['wse_std', 'wse_u', 'xtrk_dist',
              'wse_std_threshold', 'wse_u_threshold', 'xtrk_dist_threshold']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Final filtering mask
    mask = (
        (df['wse_std']              <= df['wse_std_threshold']) &
        (df['wse_u']                <= df['wse_u_threshold']) &
        (df['xtrk_dist'].abs()      >= df['xtrk_dist_threshold'])
    )

    # Return only the original columns
    return df.loc[mask, original_columns]

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
    return df_combined.loc[df.index.intersection(df_combined.index)].sort_index()[original_columns]

def convert_to_daily_series(df, gauge_df, 
                            time_col='datetime',
                            gauge_time_col='gauge_datetime',
                            wse_col='wse',
                            wse_filtered_col='wse_adjusted',
                            gauge_wse_col='gauge_wse',
                            interp_method='linear'):
    """
    Compute daily-interpolated WSE time series from SWOT (raw and adjusted) and
    gauge data over their overlapping date range.

    Note: The overlapping time range is determined based on interpolated wse_col (not wse_filtered_col) 
    and the original gauge_wse_col. If wse_filtered_col is empty, the corresponding output 
    will be NaN, but the function can still return valid unfiltered and gauge outputs.

    Returns NaN for all outputs if either df or gauge_df is empty.
    
    Updated: 07/01/2025 from "compute_daily_variability"
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
    full_range = pd.date_range(start=wse_daily.index.min(), end=wse_daily.index.max(), freq='D')

    def safe_interp(series, full_index):
        return (series.reindex(full_index)
                      .interpolate(method=interp_method, limit_direction='both')
                      .bfill()
                      .ffill()) #Extrapolates flatly using the edge values
    #Flat extrapolation is safer and more appropriate for lake WSE time series unless we have 
    #strong hydrological or operational justification for assuming a linear or quadratic trend.

    wse_interp_full = safe_interp(wse_daily, full_range)
    wse_filtered_interp_full = safe_interp(wse_filtered_daily, full_range) if not wse_filtered_daily.empty else np.nan

    # Now determine overlap between interpolated wse and original gauge_daily
    if gauge_daily.empty or wse_interp_full.empty: #Invalid
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    start_date = max(wse_interp_full.index.min(), gauge_daily.index.min())
    end_date = min(wse_interp_full.index.max(), gauge_daily.index.max())

    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date: #Invalid
        return {
            'daily_wse': np.nan,
            'daily_wse_filtered': np.nan,
            'daily_gauge': np.nan
        }

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Interpolate gauge onto the overlapping date range
    gauge_interp = safe_interp(gauge_daily, date_range)

    # Slice interpolated WSE and filtered WSE into the same range
    wse_interp = wse_interp_full[date_range]

    if isinstance(wse_filtered_interp_full, pd.Series):
        wse_filtered_interp = wse_filtered_interp_full[date_range]
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