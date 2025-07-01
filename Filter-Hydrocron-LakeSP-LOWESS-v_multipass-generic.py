# -*- coding: utf-8 -*-
"""
Created on April 20, 2025

LakeSP filter based on iterative LOWESS filtering method
For other lakes without gauge observations (except the Three Gorges Reservoir)
This version was improved based on Melanie's script shared in March, 2025.

@author: Melanie Trudel, Jida Wang
"""


import requests, datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import StringIO
import numpy as np
import statsmodels.api as sm #from Melanie. 

def nearest_ind(items, pivot):
    time_diff = np.abs([date - pivot for date in items])
    return time_diff.argmin(0), items[time_diff.argmin(0)]


#from statsmodels.nonparametric.smoothers_lowess import lowess #Note statsmodels package is needed here: pip install statsmodels

# Function to generate confidence-like bounds for a LOWESS smoothing curve 
# by running the smoothing multiple times using a range of smoothing parameters (frac) 
# and iteration counts (it), then taking the min and max of the results.
# Variables: 
# x, y: Input data points.
# eval_x: The x-values at which the LOWESS is evaluated (can be denser/sparser than original x).
# minfrac, maxfrac: The lower and upper bounds for the frac parameter (controls the smoothing window).
# it_v: A list of values for the it parameter (number of robustness iterations).
def lowess_with_confidence_bounds(x, y, eval_x, minfrac, maxfrac, frac_step, it_v):        
        
    #create values from minfrac to maxfrac in 0.01 increments. These control how "smooth" the LOWESS curve is.
    frac_v = np.linspace(minfrac,maxfrac,int(np.ceil((maxfrac-minfrac)/frac_step)+1))
    # frac_v = np.linspace(minfrac, maxfrac, 10)  # Or fix it at 10 times to save time (not always though).
    
    # Preallocate a matrix to hold smoothed results
    p=0
    # Each row corresponds to a LOWESS run using a specific combination of frac and it; each column is a point in eval_x.
    smoothed_values = np.empty((len(frac_v)*len(it_v), len(eval_x)))
    #residual_values = np.empty((len(frac_v)*len(it_v), len(eval_x)))
    # Run LOWESS for each combination of frac and it.
    for j in range(len(it_v)):
        for i in range(len(frac_v)):
            # Store each smoothed curve as a row in smoothed_values
            # The xvals argument tells the LOWESS function: 
            #     “do not just smooth at the original x; instead, evaluate the smoothed curve at xvals.”
            smoothed_values[p] = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, it=it_v[j], frac=frac_v[i]) #from Melanie.
            p += 1
    # Calculate envelope bounds
    # Extracts the minimum and maximum smoothed values at each point in eval_x, across all the different LOWESS runs 
    # — basically returning a sort of empirical envelope or uncertainty band.
    # Some time lowess produce nan, so I used nanmin and nanmax to obtain the top and bottom    
    bottom=np.nanmin(smoothed_values, axis=0)
    top=np.nanmax(smoothed_values, axis=0) 
    return bottom, top, smoothed_values
    # note that when frac is too small, smoothed_values can lead to nan values. 

# Function to calculate residuals based on the absolute minimum residual (with the sign kept) for each data point. 
# A: all smoothed curves based on different [frac, it] argument combinations: 
#      basically the output smoothed_values from function lowess_with_confidence_bounds 
# B: Original observations to evaluate residuials: eval_x for function lowess_with_confidence_bounds 
def signed_min_abs_residual(A, B): 
    # Subtract B from each row of A
    B = np.array(B)
    E = A - np.array(B)  # Residual time series for each smoothing
    # Mask NaNs in E for computation
    abs_E = np.abs(E)
    abs_E[np.isnan(E)] = np.inf  # So NaNs won't be selected as min
    # Retrieve indices of minimum absolute residual per column (observation per timestep)
    idx = np.argmin(abs_E, axis=0)
    # Use indices to retrieve original signed residuals
    result = E[idx, np.arange(E.shape[1])]
    return result

# Function to calculate residuals based on the median residual for each data point. 
# A: all smoothed curves based on different [frac, it] argument combinations: 
#      basically the output smoothed_values from function lowess_with_confidence_bounds 
# B: Original observations to evaluate residuials: eval_x for function lowess_with_confidence_bounds 
def median_residual(A, B):
    # Subtract B from each row of A
    B = np.array(B)
    E = A - np.array(B)  # Residual time series for each smoothing
    result = np.nanmedian(E, axis=0) # Median residual for each data point. 
    return result
    
# #==========PARALLEL PROCESSING==========
# # Since each smoothing run is independent, parallel processing makes a big impact. Here's a version using joblib.
# #   n_jobs=-1 uses all available CPU cores.
# #   Adjust frac resolution (0.02 here) for speed/smoothness balance.
# from joblib import Parallel, delayed

# def lowess_single_run(x, y, eval_x, it, frac):
#     return lowess(endog=y, exog=x, xvals=eval_x, it=it, frac=frac)

# def lowess_with_confidence_bounds_fast(x, y, eval_x, minfrac, maxfrac, frac_step, it_v, n_jobs=-1):
#     frac_v = np.linspace(minfrac, maxfrac, int(np.ceil((maxfrac - minfrac) / frac_step)) + 1)
    
#     combos = [(it, frac) for it in it_v for frac in frac_v]
    
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(lowess_single_run)(x, y, eval_x, it, frac) for it, frac in combos
#     )
    
#     smoothed_values = np.array(results)
#     bottom = np.nanmin(smoothed_values, axis=0)
#     top = np.nanmax(smoothed_values, axis=0)
    
#     return bottom, top, smoothed_values
# #===================================


#lakeID_list = [7110570263]
lakeID_list = [8221430182]
#LakeID 4340980733 is the Three Gorges Reservoir (TGR) ---------------------------------------------------------------------------------------

# Define lowess parameters
minfrac = 0.07 # 0.07 Tricky: We may want to avoid overfitting otherwise selection may be biased towards the overfitted curve. 
frac_step = 0.01 
maxfrac = 0.3 #0.4
it = [1,2,3,4,5] #Default value 3 usually does well. 
 
# Define fill values depending on variable type. 
fill_text = 'no_data'
fill_float = -999999999999

for n in range(0, len(lakeID_list)):     
    
    #####################################
    #read Hydrocron
    # Define parameters for Hydrocron
    feature_id = lakeID_list[n]
    feature = "PriorLake"
    start_time = "2023-03-01T00:00:00Z"  
    end_time = "2025-06-06T00:00:00Z"
    output =  "csv" #"geojson"
    fields  = 'lake_id,pass_id,obs_id,overlap,n_overlap,time,time_str,wse,wse_u,wse_r_u,wse_std,area_total,area_tot_u,area_detct,area_det_u,layovr_val,xtrk_dist,quality_f,dark_frac,ice_clim_f,partial_f,xovr_cal_q,geoid_hght,solid_tide,load_tidef,load_tideg,pole_tide,dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,p_ref_area,crid'
    
    enquiry_input =  "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=" + \
                    feature + "&feature_id=" + str(feature_id) + "&start_time=" + start_time + "&end_time=" + end_time + "&output=" + output + "&fields=" + fields
    hydrocron_response = requests.get(enquiry_input).json()
    extracted_data = hydrocron_response['results'][output] # extract just the csv
    df = pd.read_csv(StringIO(extracted_data))
    ########################################
  
    print(feature_id)
    #Clean up the data
    df = df.loc[df.time_str != fill_text] #Drop measurements where time_str is no_data. 
    df.wse = df.wse.mask(df.wse == fill_float, np.nan) #Replace values in the wse column, where value = fill, to nan.
    df.wse_u = df.wse_u.mask(df.wse_u == fill_float, np.nan) #Replace values in the wse_u column, where value = fill, to nan.
    df['index_col'] = range(0, len(df)) #add an index attribute to later join the removed outliers. 
    # Convert 'time_str' to datetime format.
    df['time_str'] = pd.to_datetime(df['time_str'], format='%Y-%m-%dT%H:%M:%SZ') 
    df['wse_unbias'] = range(0, len(df)) 
    
    #####################################
    #unbiais between pass
    
    n_pass = df.pass_id.unique()
    # n_pass = [1]  # to test without the unbiais between pass
    if len(n_pass) > 1:
     AREA = []
     for nn in range(len(n_pass)):
       AREA_pass = np.nanmedian(df.area_detct[(df['ice_clim_f']<1) & (df['wse_u']<0.1) & (df['wse_std']<1.0)  & (df['pass_id']==n_pass[nn])   ])
       AREA.append(AREA_pass)
    
    # Do we remove pass with a lot of difference or very small portion of the lake ??
     good_pass = n_pass[AREA>np.nanmax(AREA)*0.1]
     REF_pass= n_pass[AREA==np.nanmax(AREA)]

     df = df[df['pass_id'].isin(good_pass)] # drop bad pass
     B_REF= np.median(df.wse[(df['ice_clim_f']<1) & (df['wse_u']<0.1) & (df['wse_std']<1.0) & (df['pass_id']==REF_pass[0]) ])
#     B_REF= np.median(df['geoid_hght']) 
    
    
     B=[]
     for nn in range(len(good_pass)):
       B_pass = np.median(df.wse[(df['ice_clim_f']<1) & (df['wse_u']<0.1) & (df['wse_std']<1.0)  & (df['pass_id']==good_pass[nn])])
#       B_pass = np.median(df['geoid_hght'])
       B.append(B_pass-B_REF)
       df.wse_unbias[df['pass_id']==good_pass[nn]] = df.wse[df['pass_id']==good_pass[nn]] - B[nn]

     df.wse=df.wse_unbias


    #################################################### 
    # Start filtering --- output will be df_cleaned (data after filtering)
    # This "while" loop: 
    #   1. Starts with a copy of df for the evaluation purpose: df_eval
    #   2. Clean up df for the filtering purpose: df_cleaned
    #   2. Iteratively:
    #         LOWESS-smooths df_cleaned (evaluated on df_eval) based on argument combinations              
    #         Compute residuals of each point from all smooth lines.
    #         Remove outliers based on the residual z-scores.
    #   3. It stops when:
    #         The spread (standard deviation) of the residuals are small enough (< lim), or
    #         No more outlier is removed, or
    #         It has already looped n_while times.
    lim = 1.0 
    n_while = 0    
    #Initiate the lowess-based statistical filter, which will be updated in the added attribute "stats_filter"
    df['stats_filter'] = 1 #Initiation: 1 means good; 0 means statistical outlier (to be written)
    
    # Remove very bad quality data
    df_cleaned = df.dropna(subset=['wse']) #drop nan wse values for smoothing purpose. 
    df_cleaned = df_cleaned[df_cleaned['wse_std'] < 5.0] # Caution: some lake surface (large fluvial lakes or reservoirs) can have a major gradient. 
    #df_cleaned = df_cleaned[df_cleaned['xovr_cal_q'] < 2] #Remove bad crossover calibration.
    
    initial_length = len(df_cleaned) #In case this could be zero, the "while" statement won't run. 
    updated_length = 0 #Initiate updated_length
    while (lim > 0.1) & (updated_length < initial_length) & (n_while < 8):
        initial_length = len(df_cleaned) # initialize/update initial_length
        
        df_eval = df_cleaned.copy() #df_eval is for evaluation purpose.  
        
        # Remove poor-quality data based on quality flag for smoothing purpose. 
        #df_cleaned = df_cleaned[df_cleaned['wse_std'] < 3.0] # Caution: some lake surface (large fluvial lakes or reservoirs) can have a major gradient. 
        # ... Eliminating them may remove climate extreme points that are critical for characterizing lake change signal, e.g., Lake Tefe (ID 6220321573). 
        df_cleaned = df_cleaned[((df_cleaned['wse_std'] < 2) | (df_cleaned['ice_clim_f']>0))] # Allow bad wse_std if observations are ice-affected.
        df_cleaned = df_cleaned[df_cleaned['xovr_cal_q'] < 2] #Remove bad crossover calibration. 
        df_cleaned = df_cleaned[df_cleaned['wse_u'] < 0.1] #originally: 0.5 
 
        # Compute filter base on different (frac, it) combinations and apply the filter on all data (df_eval)
        bottom, top, smoothed_values = lowess_with_confidence_bounds(df_cleaned['time'], df_cleaned['wse'], df_eval['time'], \
                                                                     minfrac, maxfrac, frac_step, it)
     
        #calculate residuals
        residuals = signed_min_abs_residual(smoothed_values, df_eval['wse']) #based on the absolute minimum residual (with the sign kept)
        #residuals = median_residual(smoothed_values, df_eval['wse']) #another option: based on the median residual. 
        #----------------------------
        ## Original code from Melanie:
        #dmin = np.abs(df4['wse'] - bottom)
        #dmax =np.abs(df4['wse'] - top)
        #fil = np.min([dmin,dmax],axis=0)
        #df = df4[fil<np.std(fil)*3]
        #lim = np.std(fil)
        #n_while = n_while+1
        #----------------------------
        
        # Remove outliers based on residual z-score
        # Note that residuals can be nan if frac is too small. 
        if np.nansum(np.abs(residuals)) == 0: # note sometimes all residuals are 0 due to overfitting.
            z_scores = (residuals - np.nanmean(residuals))/1.0 #force it to be 0, so there will be no outliers. 
        else: 
            z_scores = (residuals - np.nanmean(residuals))/np.nanstd(residuals)
        # Keep points within a x-sigma threshold and cuts out strong outliers from df_eval.
        df_cleaned = df_eval[np.abs(z_scores) < 3] #Update df_cleaned based on df_eval for the next iteration.    
        
        # Update for next iteration
        lim = np.nanstd(residuals)
        updated_length = len(df_cleaned) #eventually, df_cleaned only contains the survivals after smoothing (non-outliers).♣
        n_while += 1
        
        # Show filter evolution
        plt.figure(figsize=(15, 5))
        plt.errorbar(df_eval.time_str, df_eval.wse, yerr=df_eval.wse_u, label='raw obs.', marker='o', color=(0.6,0.6,0.6), \
                     markersize=4, capsize=3, linestyle='', zorder=2) # plot data before this round of filter
        #plt.fill_between(df_eval['time_str'], bottom, top, alpha=0.5, color="b") #
        #plot all smoothed curves instead.
        for i in range(smoothed_values.shape[0]): #number of rows
            plt.plot(df_eval['time_str'], smoothed_values[i], linewidth=0.5, color='gray', alpha=0.4)
        plt.plot(df_cleaned['time_str'], df_cleaned['wse'], marker='s', linestyle='None') # Data after filtering
        plt.ylabel('Water surface elevation (m)')
        plt.xlabel('SWOT observation date')   
        #plt.ylim(365, 368)
    #######################################
    
    # Post-processing filtering
    # Further remove data that are still 50 m higher than the mean WSE
    wse_average = np.mean(df_cleaned['wse'])
    diff_average = np.abs(df_cleaned['wse'] - wse_average)
    df_cleaned = df_cleaned[diff_average < 15]
    ## Further remove data
    # df_cleaned = df_cleaned[df_cleaned['wse_u'] < 0.5]
    # df_cleaned = df_cleaned[df_cleaned['xovr_cal_q'] < 2]    
    
    #Link survivals (non-outliers) back to df data frame through index_col: 
    #     assign df.stats_filter to be 0 when df.index_col goes beyond df_cleaned.index_col (i.e., outliers)
    #     stats_filter: 1 means good; 0 means statistical outlier (to be written)
    df.loc[~df['index_col'].isin(df_cleaned['index_col']), 'stats_filter'] = 0 #~ is a logical NOT operator.   
        
    
    # Plot final time series
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(12, 6))  # <- create Axes object
    ax.grid(True, linewidth=0.5, zorder=1)
   
    ##Three Gorges ReservoirR only ---------------------------------------------------------------------------------------------------------------------------
    if feature_id == 4340980733:
        #Read gauge data
        gauge_wse = r"D:\JW\SWOT\gauge_TGD.xlsx"
        gauge_data = pd.read_excel(gauge_wse)     
        plt.plot(gauge_data.Date, gauge_data.Water_level_m, label='gauge', color='green', marker='x', markersize=3, linestyle='', zorder=4)
        plt.ylim(130, 190)
    ##TGR only ---------------------------------------------------------------------------------------------------------------------------  
    
    # Plot raw WSE time series (with error bars)
    ax.errorbar(df.time_str, df.wse, yerr=df.wse_u, label='raw SWOT', marker='o',
            color=(0.6, 0.6, 0.6), markersize=4, capsize=3, linestyle='', zorder=2)

    # Flag measurements with CNES filters
    ax.plot(df.query('quality_f == 1').time_str, df.query('quality_f == 1').wse,
            label='quality_f = 1', marker='s', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor='red')
    ax.plot(df.query('xovr_cal_q >= 2').time_str, df.query('xovr_cal_q >= 2').wse,
            label='xovr_cal_q >= 2', marker='D', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor='orange')
    ax.plot(df.query('ice_clim_f >= 1').time_str, df.query('ice_clim_f >= 1').wse,
            label='ice_clim_f >= 1', marker='^', linestyle='', markersize=8,
            markerfacecolor='none', markeredgecolor='blue')
    ax.plot(df.query('wse_std >= 2').time_str, df.query('wse_std >= 2').wse,
            label='wse_std >= 2', marker='o', linestyle='', markersize=7,
            markerfacecolor='none', markeredgecolor='yellow')

    # Plot filtered result
    filtered = df.query('stats_filter != 0')
    ax.errorbar(filtered.time_str, filtered.wse, filtered.wse_u,
           label='flexible filter', color='black', marker='o',
           markersize=4, capsize=3, linestyle='--')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    fig.autofmt_xdate()
    ax.set_xlim(datetime.datetime(2023, 4, 1), datetime.datetime(2025, 5, 1))    

    # Format y-axis
    ax.set_ylim(np.nanmin(df_cleaned.wse), np.nanmax(df_cleaned.wse))    
    
    
    # Axis labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('WSE (m)', fontsize=12)
    ax.set_title('Lake ID ' + str(feature_id) + ' Time Series Plot (LOWESS)')
    ax.legend()
    

# Unbias with station to compare



    

    df_SPENCE = pd.read_csv(r'C:\SWOT\transfert_geoid\Baker Creek daily lake levels FEB 2023 to May 2025.csv',sep=',', encoding='iso-8859-2')
    stationSPENCE = 'Landing Lake'
       
    wse_ECCC = df_SPENCE[stationSPENCE]
        
    d_ECCC = pd.to_datetime(df_SPENCE['date'],format='mixed').reset_index(drop=True)



   
#    stationECCC = '10JC003'
#    stationECCC = '06EC003'

#    url = "https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline?stations[]="+ stationECCC + "&parameters[]=46&start_date=2022-03-01%2000:00:00&end_date=2025-06-09%2023:59:59"
#    df_eccc = pd.read_csv(url)
#    col_names = ['ID', 'Date', 'Param', 'WSE', 'Qual', 'Symb', 'Appro', 'classif','Qual']
#    df_eccc.columns = col_names

#    d_ECCC = pd.to_datetime(df_eccc['Date'],format='%Y-%m-%dT%H:%M:%SZ')
#    wse_ECCC= df_eccc["WSE"]


    [ind, date_find] = nearest_ind(d_ECCC, df['time_str'])
    # calcul l'ecart entre la date trouvée et la date SWOT
    ecart = np.array((date_find- df['time_str']).dt.days)
    ecart = np.array((date_find.reset_index(drop=True)- df['time_str'].reset_index(drop=True)).dt.days)
    # conserve la donnée juste si l'écart n'est pas de plus de 1 jours   
    OBS_STATION = wse_ECCC[ind[abs(ecart)<2]].reset_index(drop=True)
    df = df[abs(ecart)<2].reset_index(drop=True)

    delta = np.array(OBS_STATION)- np.array(df['wse'])
    B = np.nanmedian(delta[(df['ice_clim_f']<1) & (df['wse_u']<0.1) & (df['wse_std']<2.0) ])
    SWOT_to_STATION = df_cleaned['wse'] + B

    
    plt.figure(figsize=(15, 5))
    plt.plot(d_ECCC, wse_ECCC, label='gauge', color='green', marker='x', markersize=3, linestyle='', zorder=4)
    plt.plot(df_cleaned['time_str'], SWOT_to_STATION, marker='s', linestyle='None') # Data after filtering
   
    plt.show()
#    plt.savefig(r'D:\JW\SWOT\Codes\Codes_for_processing_LakeSP\Plots_others\lakeID_'+str(df['lake_id'].iloc[0])+'_lowess.png') 