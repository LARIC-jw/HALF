"""
Customized LakeSP WSE filter (v9)
Initialized: 04/20/2025
Last updated: 09/15/2025
Authors: 
    Jida Wang (jidaw@illinois.edu); 
    Melanie Trudel (melanie.trudel@usherbrooke.ca)
Goals: 
    To balance (1) WSE noise removal and (2) a good representation of the all-year lake level hydrograph.   
General Structure
    This customized filtering process consists of two major steps:
        - Step 1. Heuristic threshold calibration:
            > Heuristic thresholds are calibrated using a conservative SP subset selected using the summary quality flags.
            > These calibrated thresholds will later be applied to extract a baseline subset from the full SP time series.
            > While some noise may remain, this heuristic baseline provides an initial representation of lake phenology.
        - Step 2. Time series filtering:
            2.1: Baseline filtering:
                > The calibrated heuristic thresholds are applied to the initial SP time series to retrieve the baseline subset.
            2.2: Iterative low-pass filtering:
                > A low-pass filter (e.g., LOWESS or Savitzky–Golay) is then fitted to the baseline, but evaluated against 
                 the initial SP time series to identify and remove noises.
                > This procedure is repeated iteratively until convergence criteria are satisfied.
        Flexible parameter settings are supported throughout the process.
    See the flowchart in the PPT for more detailed steps. 
"""




"""
Define global parameters:
    start_time (yyyy-mm-ddThh:mm:ssZ): Starting time of the WSE time series
    end_time (yyyy-mm-ddThh:mm:ssZ):   Ending time of the WSE time series
    
    work_dir (text):                   Directory containing gauge data and storing outputs
    
    SP_retrieval_method (tex):         Preferred method to retrieve the SWOT LakeSP time series: 
                                            - "Hydrocron" - from the Hydrocron API
                                            - "on-premise" - from the csv file previously saved in the local disk
    
    apply_low_pass_filter (text):      "yes" = both baseline (Step 2.1) and low-pass (Step 2.2) filters will be executed;
                                       "no" = only baseline filter (Step 2.1) will be executed. 
            
    The following parameters only matter if apply_low_pass_filter is set to "yes":
    evaluating_at_full_data (text):    "yes" = evaluate filtering (z-score clipping) on full LakeSP data; 
                                       "no" = evaluate only on selected observations (the baseline time series)       
    r2_filter (text):                  "yes" = perform another round (round-2) of filtering to remove remaining noise;
                                       "no" = otherwise                                     
    filter_type (text):                Low-pass filter type: lowess, wavelet, savgol, kalman, spline, median, and hampel.
    z_score_thresholds:                Z-score threshols
        [0](float):                       For round-1 (more aggressive) filtering
        [1] (float):                      For round-2 (less aggressive) filtering
    maximum_residual_spreads:          A tolerance of maximum relative residual that is not considered to be an outlier
        [0] (float):                      For round-1 filtering 
        [1] (float):                      For round-2 filtering 
    show_filtering_evolution (text):   "yes" = plot how outlier filtering evolves through iteration;
                                       "no" = otherwise
        
        
"""
# Global parameters
start_time = "2023-07-21T00:00:00Z" #2023-07-21 is the start of the SWOT nominal orbit.
end_time = "2025-07-11T00:00:00Z"

work_dir = r'D:\D\Research\Projects\SWOT\Initial_global_lakes\Codes\Updated_codes_for_processing_LakeSP'
SP_retrieval_method = 'on-premise' #'Hydrocron' or 'on-premise'
apply_low_pass_filter = 'yes' #'yes' strongly recommended

# The following parameters only matter if apply_low_pass_filter = 'yes'
evaluating_at_full_data = 'no'          #'no' recommended
r2_filter = 'yes'                       #'yes' recommended
filter_type = 'savgol'                  #lowess, wavelet, savgol, kalman, spline, median, hampel.
z_score_thresholds = [2.576, 3.5]       #2.576(99% for two tails), 2.807(99.5%), 2.967(99.7%), 3.291(99.9%), 3.5(99.95%)
maximum_residual_spreads = [0.07, 0.05] #0.08 0.06
show_filtering_evolution = 'no'         #for visualization only; caution: 'yes' may load many figures at the end of the script execution. 

script_version ='v9'


"""
Import customized functions:
        compute_rmse:              Computes root mean squared error (RMSE), np.nan robust. 
        compute_correlation:       Computes Pearson or Spearman correlation coefficient        
        calibrate_heuristic_thresholds: Calibrate heuristic thresholds (max wse_std, max wse_u, and min xtrk_dist) before SP filtering.
        apply_customized_filter:   Apply heuristic thresholds to filter the SP time series.   
        apply_baseline_tukey_filter: Apply a simple baseline turkey IQR filter to infer WSE variability (not suitable for phenology)
        sp_cycle_adjustment:       Reduce intra-cycle WSE inconsistencies in the SP time series caused by multiple orbit passes.
        convert_to_daily_series:   Compute daily-interpolated WSEs from SWOT and gauge data over their overlapping time range.        
"""
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from io import StringIO
import seaborn as sns
from customized_functions import compute_rmse, compute_correlation, \
        calibrate_heuristic_thresholds, apply_customized_filter, apply_baseline_tukey_filter, \
        sp_cycle_adjustment, convert_to_daily_series




"""
Validation metadata setup

Summary of the validation gauge data:  
As of 08/12/2025, we have collected the following gauge data. 
Note regions/lakes may overlap among the data sources but the unique lake count is provided at the bottom. 

Region       	  lake-count   Frequency	    Sources	                       Data-providers
  Quebec	      35	       Daily	        CEHQ	                       Mélanie Trudel
  North America   586	       15 min to daily  ECCC, Quebec, USBR, USGS	   Merritt Harlan
  Canada	      277	       SWOT passes	    ECCC, CEHQ, Spence, HQ, UDES   Mélanie Trudel	
  China	          38	       Daily	        Multiple authorities           Chunqiao Song
  India	          296	       Monthly	        NWIC, APWRIMS, Gujarat, CWC    Deep Shah;Huilin Gao
  West Africa	  2	           Hourly/finer	    In situ measure	               Manuela Grippa;Félix Girard;Laurent Kergoat
  Amazon          6	           Daily	        In situ measure (MISD)	       Ayan Fleischmann
  Ceará, Brazil   8	           Every 30 min	    In situ measure (Funceme)      Rafael Oliveira;Marielle Gosset;Eduardo Sávio Rodrigues Martins
  Other Brazil    62           Daily            To confirm                     Elyssa Collins;Augusto Getirana (originally 60 reservoirs)
  
Total: 1310			
Deleted due to problematic gauge records: 13
Total unique (excluding both duplicated and deleted): 1070 PLD lakes
"""
# Initialize a dataframe for tested lakes
test_lakes = pd.DataFrame(columns=['lake_id', 'gauge_source', 'gauge_dir', 'gauge_id', 'gauge_datum']) #'gauge_datum' is no more used. 

# Read in gauge metadata. 
# Daily CEHQ records for Quebec lakes
df_CEHQ = pd.read_csv(work_dir+'/gauge_data/Canada-from-Melanie-Trudel/Daily_gauge_data_EGM08_CEHQ.csv', sep=',', encoding='iso-8859-2')
unique_pld_ids = df_CEHQ["lake_id"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'CEHQ',
        'gauge_dir': work_dir + "/gauge_data/Canada-from-Melanie-Trudel/Daily_gauge_data_EGM08_CEHQ.csv",
        'gauge_id': str(df_CEHQ[df_CEHQ["lake_id"] == unique_pld_ids[n]].iloc[0]["gauge_id"]), #Station of the first record of this lake_id
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True) 

# Lakes in North America
# Load the original CSV file
df_NA = pd.read_csv(work_dir + "/gauge_data/NA-from-Merritt-Harlan/gauge_data_08042025/NA_lake_gage_data_fulltimeseries.csv") #dense frequency
#df_NA = pd.read_csv(work_dir + "/gauge_data/NA-from-Merritt-Harlan/SWOTlake_gagedata_NorthAmerica.csv") #swot overpass frequency (deprecated)
unique_pld_ids = df_NA["lake_id"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    #if unique_pld_ids[n] not in [7320361003]: # Inconsistent gauge levels (even for the same gauge_id)
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'NA', #df_NA[df_NA["lake_id"] == unique_pld_ids[n]].iloc[0]["agency"],
        'gauge_dir': work_dir + "/gauge_data/NA-from-Merritt-Harlan/gauge_data_08042025/NA_lake_gage_data_fulltimeseries.csv",
        'gauge_id': str(df_NA[df_NA["lake_id"] == unique_pld_ids[n]].iloc[0]["gage_id"]), #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
        }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)

# Discrete records during SWOT overpass time for other Canada.
# Gage with known datum
df_Canada_GNSS = pd.read_csv(work_dir+'/gauge_data/Canada-from-Melanie-Trudel/BD_Canada_GNSS.csv', sep=',', encoding='iso-8859-2')
# Gage with unknown datum
df_Canada_UN = pd.read_csv(work_dir+'/gauge_data/Canada-from-Melanie-Trudel/BD_Canada_unknown.csv', sep=',', encoding='iso-8859-2')
df_Canada = pd.concat([df_Canada_GNSS, df_Canada_UN ]).reset_index(drop=True)                         
unique_pld_ids = df_Canada["lake_id"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Canada',
        'gauge_dir': work_dir+'/gauge_data/Canada-from-Melanie-Trudel/',
        'gauge_id': 'Canada', #string
        'gauge_datum': np.nan
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True) 
    
# Reservoirs in China
# Load the original CSV file
df_China = pd.read_csv(work_dir + "/gauge_data/China-from-Chunqiao-Song/Daily_water_level_for_Chinese_reservoirs_V2-jw-corrected.csv")
unique_pld_ids = df_China["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'China',
        'gauge_dir': work_dir + "/gauge_data/China-from-Chunqiao-Song/Daily_water_level_for_Chinese_reservoirs_V2-jw-corrected.csv",
        'gauge_id': str(df_China[df_China["PLD_Lake_ID"] == unique_pld_ids[n]].iloc[0]["Name"]), #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)

# Reservoirs in India (monthly)
# Load the metadata CSV file
df_India = pd.read_csv(work_dir + "/gauge_data/India-from-Deep-Shah/Basic_information_PLD_ID_with_WRIS_merged_deep_v2_after_manual_check.submit.csv")
unique_pld_ids = df_India["lake_id (PLD_SWOT)"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'reservoir_name': str(df_India[df_India["lake_id (PLD_SWOT)"] == unique_pld_ids[n]].iloc[0]["Reservoir Name"]),
        'reservoir_state': str(df_India[df_India["lake_id (PLD_SWOT)"] == unique_pld_ids[n]].iloc[0]["State"]),
        'gauge_source': 'India',
        'gauge_dir': work_dir + "/gauge_data/India-from-Deep-Shah/Monthly Reservoir Level & Storage Timeseries data-jw-corrected.csv",
        'gauge_id': 'India', #String
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
    'gauge_dir': work_dir + '/gauge_data/Lake-Amazon-from-Ayan/LakeTefe_WaterLevel_6220321573.xlsx',
    'gauge_id': 'Tefe',
    'gauge_datum': np.nan
}])
test_lakes = pd.concat([test_lakes, new_row], ignore_index=True) # Reindex the resulting DataFrame with a fresh, sequential index

# Other Floodplaine lakes in the Amazon
# Load the original CSV file
df_amazon = pd.read_csv(work_dir + '/gauge_data/Lake-Amazon-from-Ayan/AmazonFloodplainLakes_5_organized.csv')
unique_pld_ids = df_amazon["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Amazon_floodplain',
        'gauge_dir': work_dir + '/gauge_data/Lake-Amazon-from-Ayan/AmazonFloodplainLakes_5_organized.csv',
        'gauge_id': 'Amazon_floodplain', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)

# Reservoirs in Ceara State, Brazil
# Load the original CSV file
df_ceara = pd.read_csv(work_dir + "/gauge_data/Ceara-reservoirs-from-Rafael/Ceara_reservoirs_in_situ.csv")
unique_pld_ids = df_ceara["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Ceara_Brazil',
        'gauge_dir': work_dir + "/gauge_data/Ceara-reservoirs-from-Rafael/Ceara_reservoirs_in_situ.csv",
        'gauge_id': str(df_ceara[df_ceara["PLD_Lake_ID"] == unique_pld_ids[n]].iloc[0]["Name"]), #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field
    }])
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)
    
# Other reservoirs in Brazil
# Load the original CSV file
df_Brazil = pd.read_csv(work_dir + "/gauge_data/Brazil-reservoirs-from-Elyssa/Brazil_reservoirs_in_situ.csv")
unique_pld_ids = df_Brazil["PLD_Lake_ID"].unique() # Retrieve unique lake (PLD) IDs. 
for n in range(len(unique_pld_ids)):
    new_row = pd.DataFrame([{
        'lake_id': unique_pld_ids[n], #integer
        'gauge_source': 'Other_Brazil',
        'gauge_dir': work_dir + "/gauge_data/Brazil-reservoirs-from-Elyssa/Brazil_reservoirs_in_situ.csv",
        'gauge_id': 'Other_Brazil', #String
        'gauge_datum': np.nan # use NaN for numeric field and None for object (string) field○
    }])
    #if new_row['lake_id'].iloc[0] != 6420530103: #gauge values are all the same
    test_lakes = pd.concat([test_lakes, new_row], ignore_index=True)  

# Retrieve unique lake_ids
test_lakeIDs = np.array(test_lakes['lake_id']) 
# If needed, remove any lakes that may have gauge data issues. 
# Pattern suspiciously different: 4530045882, 4530047122 (?), 4530055932 (?), 4530064112 (caused by time frequency difference?)
#                                 4530182732, 4530360803 (?), 4530368712 (?), 4530388743 (?),  4530389213 (?)
#                                 6420530103, 7421035423, 7520006182 (?), 7520021732 (?),
#                                 7120849393
# Remove the following lakes completely, where gauge records appear problematic. 
lakes_to_exclude = [4530045882, 4530047122, 4530055932, 4530182732, 4530360803, 4530368712, 4530388743,  4530389213, \
                    6420530103, 7421035423, 7520006182, 7520021732, 7120849393]
#lakes_to_exclude = []
test_lakeIDs =  [x for x in test_lakeIDs if x not in list(set(lakes_to_exclude))] #remove lakes_to_exclude

# ----------------If needed, read other lakes without gauge data for visual validation------------------
#test_lakeIDs = [6220306802,6220306852,6220306892,6220307392,6220307412,6220307442,6220307582,6220307872,6220307922,6220307942,\
#                6220308052,6220308072,6220308442,6220308472,6220308482,6220308792,6220308802,6220308812,6220308822,6220308942,\
#                6220309302,6220309332,6220309372,6220309462,6220309582,6220309882,6220310082,6220310172,6220310292,6220359712,\
#                6220359762,6220359792,6220359872,6220359892,6220361012,6220361022,6220361092,6220361122,6220362172,6220362192,\
#                6220362202,6220362212,6220362232,6220362252,6220362282,6220362292,6220362302,6220363322,6220363392,6220363402,\
#                6220363412,6220364382,6220364482,6220364502,6220364522,6220364532,6220364562,6220364572,6220364592,6220364602,\
#                6220364642,6220364662,6220364672,6220364682,6220364692,6220364722,6220364732,6220365932,6220365972,6220366002,\
#                6220366012,6220366032,6220366052,6220366062,6220366072,6220366092,6220366102,6220366542,6220366602,6220367452,6220367512]
#test_lakeIDs = [7251006483] #[6220306892]#[6220308942] [6220364532] 
#test_lakeIDs = [4610062383, 4610049903, 2160053363] #[4530746652]#[6220306892] #[4340980733]
##test_lakeIDs = [7240054132] #[4340980733] #[4520076683] #,4550066742]
# Note: 4520076683, wse_std max = 3 may not be sufficient.  
# --------------------------------------------------------------------------------------------------- 

# Retrieve unique values in test_lakeIDs while preserving their original order
test_lakeIDs = pd.unique(test_lakeIDs)
# Note: In case a PLD lake IDs is redundant among different gauge sources, we will prefer 
# the first gauge source (the above gauge sources have been ranked in an decreasing order of preference).
print('total numnber of unique PLD lakes with gauge data: ' + str(len(test_lakeIDs)))



"""
Main script.

MAIN OUTPUTS:

1. df_lake_time_series (DataFrame)
    Stacked LakeSP time series for all evaluated lakes, with the following attributes:
    • Original LakeSP attributes  
    • wse_adjusted: Retained (good-quality) observations after filtering, after cycle adjustment as well; 
                    values equal to original WSE when no adjustment is needed.  
    • gauge_datetime: The closest timestamp in the gauge data, if available
    • gauge_wse: Original gauge WSE values corresponding to gauge_datetime, if available  
    • gauge_wse_bias_corrected: Bias-corrected gauge WSE values, if available
    • filter_flag: Flag indicating results of the customized filter. 
        - 1 indicates a retained (good-quality) observation
        - 0 indicates an outlier removed by the filter  

2. df_lake_heuristic_thresholds (DataFrame)
    Heuristic thresholds for SP filtering, calibrated for each lake, pass_id, data version (crid), 
    or ice condition grouping.
    
    Each row represents a unique combination of:
    - lake_id
    - crid_scenario (version group)
    - pass_id (SWOT orbit pass)
    - ice_condition (ice-covered or ice-free)

    Attributes:
    • lake_id: PLD lake_id
    • crid_scenario: Scenario grouping based on data version (CRID).Two possible values:
         - "PIC2_or_PID0": newer versions (e.g., PIC2, PID0)
         - "early_versions": older versions (e.g., PIC0, PGC0)
    • pass_id: Integer ID of the SWOT orbit pass
    • ice_condition: ice-covered or ice-free
    • wse_std_threshold: Heuristically calibrated upper threshold for the standard
                         deviation of WSE (wse_std), used to exclude noisy or
                         unstable measurements. Typically computed as a max value
                         for that pass and version grouping, capped between
                         a defined min and max if defined.
    • wse_u_threshold: Heuristically calibrated upper threshold for the uncertainty
                       of WSE (wse_u), used similarly to exclude unreliable
                       observations. Also computed per lake, pass, and version group.
    • wse_u_th xtrk_dist_threshold: Minimum abs(xtrk_dist) threshold                  

3. df_lake_stats (DataFrame)
    Summary statistics for each lake, including validation metrics and retention rates:

    • lake_id: Lake PLD identifier  

    Comparison of custom-filtered and gauge-based WSE time series:
    • rmse: Root Mean Square Error between filtered LakeSP and gauge WSEs  
    • correlation: Correlation coefficient between filtered LakeSP and gauge WSEs  
    • var_swot: Variability (in standard deviation) of WSE in the filtered LakeSP time series  
    • var_gauge: Variability of gauge WSEs corresponding to timestamps in the filtered LakeSP series 
    
    NOTES: 
        > rmse informs absolute error scales (not the focus of our SP filter)
        > rather, our filter focuses on the effectiveness of representing lake WSE shape characteristics: 
            > standard deviation informs magnitude of fluctuation
            > while correlation informs synchronous shape. 
            > Both aspects are necessary: One can be closer to the gauge in std dev, while the other is closer in correlation. 

    Comparison between custom-filtered and gauge-based daily-resolution WSE hydrographs
    • var_swot_daily: Variability of daily WSE interpolated from the filtered LakeSP time series  
    • var_gauge_daily:Variability of daily WSE from the full gauge time series 
    
    Comparison of CNES-baseline1[/2]-filtered and gauge-based WSE time series:
    • rmse_baseline1[/2], following the same definition described above       
    • correlation_baseline1[/2]          
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
    • rmse_unfiltered, correlation_unfiltered  
    • var_swot_unfiltered, var_gauge_unfiltered  
    • var_swot_daily_unfiltered  
    
    Iteration times for each filter execution
    An execution can contain two rounds, and each round may include several iterations:
    • n_while: Number of round-1 iterations
    • n_while_r2: Number of round-2 interactions. For both numbers of iterations: 
        - -9 indicates this lake cannot be run due to no SWOT observations at all
        - -2: This round of filter is turned off (e.g., setting r2_filter = 'no'). 
        - -1 indicates this lake is abandoned (not meeting criteria)
        - other integers indicating iteration times (0 if apply_low_pass_filter = 'no')
    Ranking of filter execution attempts
    Not every filter execution can be successful. If one attempt failed (criteria 
    not satisfied), a lower-ranking attempt is executed. 
    • filter_attempt:
        - 1_strict:  Strict criteria applied to the customized filter, so that
                     retained time series must be sufficiently long (e.g., 1 year) and 
                     must not contain major gaps (e.g., 3 months or 1 season)
        - 2_lenient: Less strict criteria applied to the customized filter, so 
                     there is no constraint on temporal span or gap, 
                     but retained time series must contain >=5 observations
        - 3_tukey:   If attempt 1 or 2 still failed, apply only a Tukey (IQR) method on 
                     the baseline time series, and the retained time series must contain
                     at least one observation.
        - 4_none:    If none of the first three attempts was successful, this lake
                     is left without any filtering. 
        
    Ice phenology
    'ice_duration': the number of records where ice_clim_f == 2 as a proportion of the full time series  
"""
# Initialize the three major outputs (described above) as empty dataframe
df_lake_time_series = pd.DataFrame()
df_lake_heuristic_thresholds = pd.DataFrame()
df_lake_stats = pd.DataFrame()

# Define fill values depending on variable type. 
fill_text = 'no_data'
fill_float = -999999999999

# Retrieve LakeSP time series
if SP_retrieval_method == 'on-premise': # retrieve all lakes from the previously saved 'on-premise' file
    df_Hydrocron = pd.read_csv(work_dir+'/df_Hydrocron.csv')
else: # otherwise, read from Hydrocron. 
    df_Hydrocron = pd.DataFrame() # Initialize an empty dataframe for Hydrocron reading

# Loop through each unique test lake
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
    if gauge_source == 'CEHQ': # Lakes in Quebec
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["lake_id"] == feature_id] # Filter for lake_id based on the current feature_id        
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime'])
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.  
        time_tolerance = '24h'
    if gauge_source == 'Canada': # Other Canadian lakes
        gauge_df_GNSS = pd.read_csv(gauge_dir + 'BD_Canada_GNSS.csv')
        gauge_df_unknow = pd.read_csv(gauge_dir + 'BD_Canada_unknown.csv')
        gauge_df= pd.concat([gauge_df_GNSS, gauge_df_unknow ]).reset_index(drop=True)     
        gauge_df = gauge_df[gauge_df["lake_id"] == feature_id]
        diff_geoid = np.abs(gauge_df['geoid_station']- gauge_df['geoid_hght'])
        gauge_df['diff_geoid']=diff_geoid 
        best_gage_id = (
           gauge_df.groupby("OBS_ID")["diff_geoid"]
           .mean()
           .idxmin()
           )
        # Filter for best gage_id with the closest geoid and format output
        gauge_df = gauge_df[gauge_df["OBS_ID"] == best_gage_id]
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['time_str'])
        gauge_df = gauge_df.rename(columns={"OBS_WSE": "gauge_wse"})
        gauge_df = gauge_df.sort_values('gauge_datetime')
        time_tolerance = '24h'
    if gauge_source == 'NA': # Lakes in North America                          
        #gauge_df = pd.read_csv(gauge_dir) #This is slow
        gauge_df = df_NA
        gauge_df = gauge_df[gauge_df["lake_id"] == feature_id] # Filter for lake_id based on the current feature_id
        gauge_df["gauge_datetime"] = pd.to_datetime(gauge_df["dateTime_UTC"], format='%Y-%m-%dT%H:%M:%SZ') # Convert time
        # Find the gage_id with the longest time span for this lake (there could be multiple gauges for the same lake)
        # Merritt: does the way I handled it make sense?         
        def time_range(gdf):
            s = gdf['gauge_datetime'].dropna()
            return (s.max() - s.min()).days if not s.empty else float('nan')
        longest_gage_id = (
            gauge_df.groupby("gage_id", dropna=False)
            .apply(time_range, include_groups=False) # silence the warning
            .idxmax()
            )       
        # Filter for best gage_id and format output
        gauge_df = gauge_df[gauge_df["gage_id"] == longest_gage_id][["gauge_datetime", "gage_stage_m", "gage_id"]]
        gauge_df = gauge_df.rename(columns={"gage_stage_m": "gauge_wse"})
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.   
        time_tolerance = '24h'
    if gauge_source == 'China': # Reservoirs in China        
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["PLD_Lake_ID"] == feature_id] # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df = pd.DataFrame({
            "gauge_datetime": pd.to_datetime(gauge_df[["Year", "Month", "Day", "Hour", "Minute", "Second"]]),
            "gauge_wse": gauge_df["WSE/m"]
            }) # Convert to gauge_df with required column format
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
        time_tolerance = '24h'
    if gauge_source == 'West_Africa': # Reservoirs in west Africa           
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["PLD_Lake_ID"] == feature_id] # Filter for PLD_Lake_ID based on the current feature_id
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
        gauge_df = gauge_df[gauge_df["PLD_Lake_ID"] == feature_id] # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
        time_tolerance = '24h'
    if gauge_source == 'Ceara_Brazil': # Small reservoirs in Ceara State, Brazil
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["PLD_Lake_ID"] == feature_id] # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side. 
        time_tolerance = '24h'
    if gauge_source == 'India': # Reservoirs in India
        reservoir_name = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'reservoir_name'].values[0]
        reservoir_state = test_lakes.loc[test_lakes['lake_id'] == feature_id, 'reservoir_state'].values[0]
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[(gauge_df["Reservoir Name"] == reservoir_name) & (gauge_df["State"] == reservoir_state)]
        gauge_df["gauge_datetime"] = pd.to_datetime(gauge_df["Date"] + "-15 12:00:00", format='%Y-%m-%d %H:%M:%S') #Assuming 15th of each month for now. 
        gauge_df = gauge_df.rename(columns={"Level": "gauge_wse"})
        # Find the "District" with the longest time span for this lake (there could be multiple distrcits for the same lake)
        def time_range(gdf):
            s = gdf['gauge_datetime'].dropna()
            return (s.max() - s.min()).days if not s.empty else float('nan')
        longest_gage_id = (
            gauge_df.groupby('District', dropna=False)
            .apply(time_range, include_groups=False)  # silence the warning
            .idxmax()
            )
        # Filter for best gage_id and format output
        gauge_df = gauge_df[gauge_df["District"] == longest_gage_id]
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.   
        time_tolerance = '16d' 
    if gauge_source == 'Other_Brazil': # Other reservoirs in Brazil
        gauge_df = pd.read_csv(gauge_dir)
        gauge_df = gauge_df[gauge_df["PLD_Lake_ID"] == feature_id] # Filter for PLD_Lake_ID based on the current feature_id
        gauge_df['gauge_datetime'] = pd.to_datetime(gauge_df['gauge_datetime']) # Ensure datetime is in datetime64 format.
        gauge_df = gauge_df.sort_values('gauge_datetime') # Sort gauge_df by time, to be on the safe side.
        time_tolerance = '24h'
    
    # Some individual gague observations appear erroneous.
    # Remove these individual errors from the gauge record:
    if gauge_source is None: 
        gauge_df = None
    else:
        # Major cap (e.g., 3+ months) in freeze-up period: will be excluded
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
        if feature_id == 4530241183:
            gauge_df.loc[(gauge_df['gauge_wse'] < 515), 'gauge_wse'] = None    
        if feature_id == 7720027743:
            gauge_df.loc[(gauge_df['gauge_wse'] < 2400), 'gauge_wse'] = None 
        if feature_id == 7740035873:
            gauge_df.loc[(gauge_df['gauge_wse'] < 226), 'gauge_wse'] = None 
        if feature_id == 8320311912:
            gauge_df.loc[(gauge_df['gauge_wse'] < 300), 'gauge_wse'] = None         
        gauge_df = gauge_df.dropna(subset=['gauge_wse']) #Drop all None values     
    
    
    # Retrieve LakeSP time series based on the preferred method
    if SP_retrieval_method == 'Hydrocron': # from Hydrocron directly. 
        # Read LakeSP data from Hydrocron
        feature = "PriorLake"
        output =  "csv" #"geojson"
        fields = 'lake_id,reach_id,obs_id,overlap,n_overlap,time,time_tai,time_str,wse,wse_u,wse_r_u,wse_std,area_total,area_tot_u,area_detct,area_det_u,layovr_val,xtrk_dist,ds1_l,ds1_l_u,ds1_q,ds1_q_u,ds2_l,ds2_l_u,ds2_q,ds2_q_u,quality_f,dark_frac,ice_clim_f,ice_dyn_f,partial_f,xovr_cal_q,geoid_hght,solid_tide,load_tidef,load_tideg,pole_tide,dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,lake_name,p_res_id,p_lon,p_lat,p_ref_wse,p_ref_area,p_date_t0,p_ds_t0,p_storage,cycle_id,pass_id,continent_id,range_start_time,range_end_time,crid,geometry,PLD_version,collection_shortname'
        enquiry_input =  "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?feature=" + \
                        feature + "&feature_id=" + str(feature_id) + "&start_time=" + start_time + "&end_time=" + end_time + "&output=" + output + "&fields=" + fields
        hydrocron_response = requests.get(enquiry_input).json()
        
        try: #if this Hydrocron reading is not empty.
            extracted_data = hydrocron_response['results'][output]
            df = pd.read_csv(StringIO(extracted_data))     
        except: #make this df an empty dataframe, but with the LakeSP column header
            # Assume the first lake does not return an empty df so SP attributes already exists in df_Hydrocron. 
            df = df_Hydrocron[df_Hydrocron["lake_id"] == -999]
            
        # Add df to df_Hydrocron. Note nothing will be added if df is empty. 
        df_Hydrocron = pd.concat([df_Hydrocron, df], ignore_index=True) #ignore_index=True resets the index.    
        
    else: # on-premise: from the file previously saved in the local disk
        df = df_Hydrocron[df_Hydrocron["lake_id"] == feature_id] # Filter for lake_id based on the current feature_id
       
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
    Compute heuristic quality thresholds based on long-term lake statistics
    
    Logic: 
        - 1. Rather than directly using the LakeSP summary quality flags for filtering, we leverage these flags to identify high-quality observations. 
        - 2. From these selected observations, we calibrate heuristic maximum thresholds for the following key metrics (determined by random forest test):
            • wse_std_threshold: Represents the minimum acceptable surface water level consistency across (wse_std) the lake.
            • wse_u_threshold: Represents the maximum acceptable uncertainty from data processing (wse_u).
            • xtrk_dist_threshold: Represents the minimum acceptable absolute distance to the central track (|xtrk_dist|).  
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
    
    # Caution: Applying pass and version groupings for wse_std or wse_u may lead to avoid over-rejection.
    #          Pass groupiong is theoretically needed for xtrk_dist as lake position varies in different passes. 
    df_heuristic_thresholds = calibrate_heuristic_thresholds(df_eval, conservative_SQL,
                                       by_crid_scenario = [False, False, False], #boolean sequence for wse_std, wse_u, xtrk_dist
                                       by_pass_id = [False, False, True], #wse_std, wse_u, xtrk_dist
                                       by_ice = [True, True, True]) #fixed here; wse_std, wse_u, xtrk_dist
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
    # wse_u_ice_min = 0.1 # Elevate min wse_u threshold for ice/freeze-up conditions to allow for data. 0.1 m is also set to include valid extremes. 
    
    # Per-metric [wse_std, wse_u, xtrk_dist], with four unique rules 'ice-free','ice-covered','both, and 'not apply'
    rules_for_ice_free_data   = ['ice-free', 'ice-free', 'not apply'] # The use of xtrk_dist thresholds is deprecated. 
    rules_for_ice_covered_data= ['ice-free', 'ice-free', 'not apply']
     
    
    # Execute the filter function   
    # Attempt 1: strict
    filter_attempt = '1_strict' # the most strict filtering
    df_filtered, n_while_filtered, filter_status = apply_customized_filter(df_eval, df_heuristic_thresholds,
                                         # Bound overrides for applied df_heuristic_thresholds
                                         wse_std_threshold_bounds = [0, 3],
                                         wse_u_threshold_bounds = [0, 0.5],
                                         xtrk_dist_threshold_bounds = [0, 75000], #deprecated
                                       
                                         # Ice overrides for applied thresholds on ice-affected rows
                                         wse_std_ice_min = 3, #use 0 if not applying the override
                                         wse_u_ice_min = 0.1, #use 0 if not applying the override
                                         
                                         allow_major_gap = 'no', # 'yes/no', to indicate if gap in the filtered time series is allowed.
                                         max_temporal_gap = 90, #Maximum temporal gap (days) for filtering
                                         min_temporal_range = 365, # Minimum tmeporal range (days) for filtering                                         
                                       
                                         # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
                                         # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
                                         rules_for_ice_free_data = rules_for_ice_free_data,
                                         rules_for_ice_covered_data = rules_for_ice_covered_data,
                                       
                                         gauge_df = gauge_df, # enter gauge_df; None if no gauge data is available. 
                                         plot_period = [start_time, end_time], #Defining start and end time for plotting.
    
                                         apply_low_pass_filter = apply_low_pass_filter, 
                                         evaluating_at_full_data = evaluating_at_full_data,
                                         r2_filter = r2_filter,
                                         filter_type = filter_type, 
                                         z_score_thresholds = z_score_thresholds, 
                                         maximum_residual_spreads = maximum_residual_spreads,
                                         show_filtering_evolution = show_filtering_evolution)
    
    # In case of no valid SWOT data for this lake (df_eval/df is empty)
    if filter_status == 'no data': #n_while_r2 is always -9 when n_while is -9. 
        filter_attempt = 'no data' # Overwrite 'strict' by 'no data'
                                   
    # Attempt 2: lenient
    if filter_status == 'fail': 
        
        # Execute the customized filter with more lenient parameters
        filter_attempt = '2_lenient'  # More lenient filtering   
        df_filtered, n_while_filtered, filter_status = apply_customized_filter(df_eval, df_heuristic_thresholds,
                                             # Bound overrides for applied df_heuristic_thresholds
                                             wse_std_threshold_bounds = [0.5, 5],
                                             wse_u_threshold_bounds = [0.1, 0.5],
                                             xtrk_dist_threshold_bounds = [0, 75000], #deprecated
                                             
                                             # Ice overrides for applied thresholds on ice-affected rows
                                             wse_std_ice_min = 5, #use 0 if not applying the override
                                             wse_u_ice_min = 0.1, #use 0 if not applying the override
                                             
                                             allow_major_gap = 'yes', # 'yes/no', to indicate if gap in the filtered time series is allowed.
                                             max_temporal_gap = 90, #Maximum temporal gap (days) for filtering
                                             min_temporal_range = 365, # Minimum tmeporal range (days) for filtering                                             
                                           
                                             # Per-metric rules (length = 3 for [wse_std, wse_u, xtrk_dist])
                                             # Valid values per metric item: 'ice-free' | 'ice-covered' | 'both' | 'not apply'
                                             rules_for_ice_free_data = rules_for_ice_free_data,
                                             rules_for_ice_covered_data = rules_for_ice_covered_data,
                                             
                                             gauge_df = gauge_df, # enter gauge_df; None if no gauge data is available. 
                                             plot_period = [start_time, end_time], #Defining start and end time for plotting.
                                             
                                             apply_low_pass_filter = apply_low_pass_filter, 
                                             evaluating_at_full_data = evaluating_at_full_data,
                                             r2_filter = r2_filter,
                                             filter_type = filter_type, 
                                             z_score_thresholds = z_score_thresholds, 
                                             maximum_residual_spreads = maximum_residual_spreads,
                                             show_filtering_evolution = show_filtering_evolution)    
                                                                               
    # Attempt 3: baseline Tukey (IQR)
    if filter_status == 'fail': 
        
        filter_attempt = '3_tukey' # Simple baseline turkey filter
        # This attempt is only to infer WSE variability uncertainty, not for characterize phenology
        # So, the previous customized filter does not apply, and instead we use a simple baseline_tukey method
        
        # Define the baseline condition as a boolean mask
        baseline_SQL = '(quality_f == 0) & (xovr_cal_q == 0) & (ice_clim_f < 2)'
                  
        # Remove remaining isolated extreme outliers using Tukey method (IQR method) 
        df_filtered, n_while_filtered, filter_status = apply_baseline_tukey_filter(df_eval, baseline_SQL, 
                                                                       multiplier = 3, 
                                                                       lower_q = 0.25,
                                                                       upper_q = 0.75,
                                                                       iteration_n=5)    
        n_while_filtered = [n_while_filtered, -2] #Update n_while_filtered to be consistent with the size of other attempts
    
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
    # Only return Option 3:
    _, _, df_filtered = sp_cycle_adjustment(df_filtered)   
    
    
    """
    Label filtered result back to the original df 
    """   
    # Label survivals (non-outlier LakeSP observations) back to the original df through index_col 
    #     assign df.filter_flag to be 0 when df.index_col goes beyond df_filtered.index_col (i.e., outliers)
    #     filter_flag: 1 means good; 0 means outlier
    df.loc[~df['index_col'].isin(df_filtered['index_col']), 'filter_flag'] = 0
    # Note: df.query('filter_flag != 0') will be the final original LakeSP observations that survived the filtering!
    # It equals df_filtered in size, but df.query('filter_flag != 0') keeps the original attribute structure. 
    
    # Left-join the 'wse_adjusted' column from df_filtered into df, based on the unique key index_col.
    df = df.merge(
        df_filtered[['index_col', 'wse_adjusted']],  # only bring the column(s) that are shared (retained after filtering)
        on='index_col',
        how='left'  # keep all rows from df, fill unmatched ones with NaN
        ) # Now df will have a new column "wse_adjusted". This also works if df_filtered is empty. 
    # Note again:
    #    wse_adjusted is only valid for filtered results (good observations)
    #    If no wse_adjusted was assigned (i.e., outliers, not in df_filtered), the value of df.wse_adjusted will be left nan. 
    #    wse_adjusted will equal wse if no cycle adjustment is needed. 
    #    So it is safe to just use wse_adjusted for representing filtered results.   
    
    # Assign filter_attempt, n_while, and n_while_r2 values into new columns
    df[['n_while', 'n_while_r2', 'filter_attempt']] = [n_while, n_while_r2, filter_attempt]
    
    
    
               
    """
    Plot filtered time series and compute statistics for this lake
    """        
    # Define CNES baselines: baseline 1 (stringent) and baseline 2 (lenient)
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
    if gauge_source is not None: # If this lake has gauge data (this works if df_filtered is empty)       
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
            tolerance=pd.Timedelta(time_tolerance)
            ) # Note: this will generate two extra attributes (gauge_datatime and gauge_wse) in df, if the lake has gauge data.        
        #df.to_csv(r"D:\D\Research\Projects\SWOT\Initial_global_lakes\Codes\Updated_codes_for_processing_LakeSP\gauge_data\Lake-Amazon-from-Ayan\test.csv", index=False)
        
        # Computing the bias between gauge and SWOT
        # (07/11/2025) Adopt Melanie's suggestion: unbias using the median of the difference on ice-free observations
        # The bias can be cau by unknown levelling of the gauge data, or 
        # by the difference between the average geoid of the lake an the geoid at the gauge station                  
        # Use wse_adjusted as the preferred SWOT wse field if available; otherwise, use wse
        if df['wse_adjusted'].isna().all(): # If 'wse_adjusted' (filtered wse) is entirely NaN or unavailable
            bias_correction_field = 'wse'
        else:
            bias_correction_field = 'wse_adjusted'  
        #bias_correction_field = 'wse' #============================
        # Use ice-free observations if possible; otherwise, use all observations
        if (df['ice_clim_f'] < 1).any(): # if there are ice-free observations in df
            mask = df['ice_clim_f'] < 1
            bias_swot_gauge = np.nanmedian(df.loc[mask, 'gauge_wse'].values - df.loc[mask, bias_correction_field].values)
            # Not recommended to do the following as it might lead to index misalginment
            #bias_swot_gauge = np.nanmedian(df['gauge_wse'][df['ice_clim_f']<1] - df[bias_correction_field][df['ice_clim_f']<1])
            
            # In case gauge and valid SP wse values during ice-free period do not overlap: 
            if np.isnan(bias_swot_gauge):                 
                bias_swot_gauge = np.nanmedian(df['gauge_wse'] - df[bias_correction_field]) # Use full period
        else: # Use full period
            bias_swot_gauge = np.nanmedian(df['gauge_wse'] - df[bias_correction_field]) 
        # In case gauge values and valid SP wse values (e.g., if wse_adjusted is used) still do not overlap:
        if np.isnan(bias_swot_gauge):
            bias_swot_gauge = np.nanmedian(df['gauge_wse'] - df['wse'])  # Us original wse to increase the chance
        # Rare: in case no valid wse values during gauge available period
        if np.isnan(bias_swot_gauge):
            bias_swot_gauge = np.nanmedian(df['gauge_wse']) - np.nanmedian(df[bias_correction_field])
        # In case df is empty. 
        if np.isnan(bias_swot_gauge):
            bias_swot_gauge = 0
        # Assign the bias corrected gauge wse into a new field of df (this will also be used for the random forest analysis)
        df['gauge_wse_bias_corrected'] = df['gauge_wse']-bias_swot_gauge
        
        # Plot the gauge time series    
        ax.plot(gauge_df['gauge_datetime'], gauge_df['gauge_wse']-bias_swot_gauge, \
               label='gauge', color='green', marker = 'o', markersize=6, linestyle='--') # Shift gauge datum to SWOT
        #ax.plot(df['datetime'], df['gauge_wse_bias_corrected'], \
        #       label='gauge', color='green', marker = 'o', markersize=6, linestyle='--') # Shift gauge datum to SWOT
            
        # Compute RMSE        
        rmse_unfiltered = compute_rmse(df['wse'], df['gauge_wse_bias_corrected'])
        # Using only filtered (retained good observation) data with filter_flag == 1:
        rmse = compute_rmse(df['wse_adjusted'], df['gauge_wse_bias_corrected'])
        # Using CNES baseline 1:
        rmse_baseline1 = compute_rmse(df['wse_baseline1'], df['gauge_wse_bias_corrected'])
        # Using CNES baseline 2:
        rmse_baseline2 = compute_rmse(df['wse_baseline2'], df['gauge_wse_bias_corrected'])

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
        # To better assess how well the filter captures the "actual" intra-annual lake WSE variability,
        # including the impact of interpolation/extrapolation induced by noise removal, 
        # we first interpolate/extrapolate the filtered time series into consecutive daily timestamps 
        # during the full timespan of the original df (LakeSP time series). Then, we identify the 
        # overlapping time range between df and the gauge record, and if needed, interpolate the gauge data
        # into consecutive daily timestamps within this range. And based on the two daily time series, 
        # we compare their WSE variability over the shared time window.
        # Caution: some of the gauge time series exhibit major temporal gaps. For example, some US gauges are
        # not operational during the winter or frozen period (assuming minimal hydrological activities despite
        # possible drawdowns). Interpolating gauge time series into daily timestamps may introduce 
        # uncertainties, which are left unaddressed yet at this moment (any help is appreciated). 
        daily_series = convert_to_daily_series(df, gauge_df, 
                                            time_col='datetime', gauge_time_col='gauge_datetime',
                                            wse_col='wse', wse_filtered_col='wse_adjusted', gauge_wse_col='gauge_wse',
                                            interp_method='linear', major_gap_days=90)        
        #filtered daily variability during the full period
        val = daily_series.get('daily_wse_filtered', np.nan)
        var_swot_daily = val.std() if isinstance(val, pd.Series) and not val.isna().all() else \
            (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))
        #Series with data → compute variability (std).
        #Series all NaN → variability undefined, so np.nan.
        #Scalar NaN → undefined, so np.nan.
        #Scalar number → no variability (a single fixed value), so 0.
        #gauge daily variability during the full period
        val = daily_series.get('daily_gauge', np.nan)
        var_gauge_daily = val.std() if isinstance(val, pd.Series) and not val.isna().all() else \
            (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))
        #raw daily variability during the full period
        val = daily_series.get('daily_wse', np.nan)
        var_swot_daily_unfiltered = val.std() if isinstance(val, pd.Series) and not val.isna().all() else \
            (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))
        
        # Using CNES baseline 1
        daily_series_baseline1 = convert_to_daily_series(df, gauge_df, 
                                            time_col='datetime', gauge_time_col='gauge_datetime',
                                            wse_col='wse', wse_filtered_col='wse_baseline1', gauge_wse_col='gauge_wse',
                                            interp_method='linear', major_gap_days=90) 
        val = daily_series_baseline1.get('daily_wse_filtered', np.nan)
        var_swot_daily_baseline1 = val.std() if isinstance(val, pd.Series) and not val.isna().all() else \
            (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0)) 
        
        # Using CNES baseline 2
        daily_series_baseline2 = convert_to_daily_series(df, gauge_df, 
                                            time_col='datetime', gauge_time_col='gauge_datetime',
                                            wse_col='wse', wse_filtered_col='wse_baseline2', gauge_wse_col='gauge_wse',
                                            interp_method='linear', major_gap_days=90)  
        val = daily_series_baseline2.get('daily_wse_filtered', np.nan)
        var_swot_daily_baseline2 = val.std() if isinstance(val, pd.Series) and not val.isna().all() else \
            (np.nan if isinstance(val, pd.Series) and val.isna().all() else (np.nan if np.isnan(val) else 0))  
    
        # Compute correlation coefficients (based on daily time series)
        # Using all data points:
        correlation_unfiltered = compute_correlation(daily_series['daily_wse'], daily_series['daily_gauge'], method='pearson') # Nan will be ignored
        # Using only filtered data with filter_flag == 1:
        correlation = compute_correlation(daily_series['daily_wse_filtered'], daily_series['daily_gauge'], method='pearson')
        # Using CNES baseline 1: 
        correlation_baseline1 = compute_correlation(daily_series_baseline1['daily_wse_filtered'], daily_series_baseline1['daily_gauge'], method='pearson')
        # Using CNES baseline 2: 
        correlation_baseline2 = compute_correlation(daily_series_baseline2['daily_wse_filtered'], daily_series_baseline2['daily_gauge'], method='pearson')       
        
    else: # if this lake has no gauge data
        rmse = np.nan            
        correlation = np.nan            
        var_swot = np.nan  #invalid lacking gauge reference          
        var_gauge = np.nan              
        var_swot_daily = np.nan            
        var_gauge_daily = np.nan
            
        rmse_baseline1 = np.nan            
        correlation_baseline1 = np.nan            
        var_swot_baseline1 = np.nan            
        var_gauge_baseline1 = np.nan              
        var_swot_daily_baseline1 = np.nan            
            
        rmse_baseline2 = np.nan            
        correlation_baseline2 = np.nan            
        var_swot_baseline2 = np.nan          
        var_gauge_baseline2 = np.nan             
        var_swot_daily_baseline2 = np.nan         
                                  
        rmse_unfiltered = np.nan
        correlation_unfiltered = np.nan
        var_swot_unfiltered = np.nan
        var_gauge_unfiltered = np.nan           
        var_swot_daily_unfiltered = np.nan
    
    if len(df) > 0: # If this lake has valid SWOT observations
        retention_n = len(df_filtered)
        retention_rate = len(df_filtered)/len(df)
        retention_rate_baseline1 = len(df.query(CNES_baseline1))/len(df)
        retention_rate_baseline2 = len(df.query(CNES_baseline2))/len(df)  
        
        # Compute the proportion of fully ice-covered period in the original time series
        ice_duration = (df['ice_clim_f'] == 2).sum() / len(df) # Simple approach for now: just use the record number (not exact time)
        
    else: # This lake has no valid SWOT observations (e.g., lake_id 4330037643)
        retention_n = np.nan #nan meaning not applicable to this lake as it has no SWOT observations. 
        retention_rate = np.nan
        retention_rate_baseline1 = np.nan
        retention_rate_baseline2 = np.nan
        ice_duration = np.nan
        # This was already returned
        #n_while = -9 #np.nan (use -9 to keep this variable as integer)
        #n_while_r2 = -9 #np.nan                 
    
    # Construct a lake stats dataframe for this lake
    df_this_lake_stats = pd.DataFrame([{
        'lake_id': feature_id,
        'rmse': rmse,            
        'correlation': correlation,            
        'var_swot': var_swot,            
        'var_gauge': var_gauge,              
        'var_swot_daily': var_swot_daily,            
        'var_gauge_daily': var_gauge_daily,
        
        'rmse_baseline1': rmse_baseline1,            
        'correlation_baseline1': correlation_baseline1,            
        'var_swot_baseline1': var_swot_baseline1,            
        'var_gauge_baseline1': var_gauge_baseline1,              
        'var_swot_daily_baseline1': var_swot_daily_baseline1,            
        
        'rmse_baseline2': rmse_baseline2,            
        'correlation_baseline2': correlation_baseline2,            
        'var_swot_baseline2': var_swot_baseline2,            
        'var_gauge_baseline2': var_gauge_baseline2,              
        'var_swot_daily_baseline2': var_swot_daily_baseline2,
        
        'rmse_unfiltered': rmse_unfiltered,
        'correlation_unfiltered': correlation_unfiltered,
        'var_swot_unfiltered': var_swot_unfiltered,
        'var_gauge_unfiltered': var_gauge_unfiltered,            
        'var_swot_daily_unfiltered': var_swot_daily_unfiltered,
        
        'retention_n': retention_n,
        'retention_rate': retention_rate,
        'retention_rate_baseline1': retention_rate_baseline1,
        'retention_rate_baseline2': retention_rate_baseline2,  
        
        'n_while': n_while, 
        'n_while_r2': n_while_r2,
        'filter_attempt': filter_attempt,
        
        'ice_duration': ice_duration
        }])
    
    # To sum up for dataframe df_lake_stats:
    #       For lakes that have no SWOT data (df is empty): n_while, n_while_r2 are -9, and retention metrics are np.nan, 
    #                 and validation metrics are also np.nan.
    #       For lakes that are abandoned: n_while and n_while_r2 are -1, and retention metrics are 0,
    #                 and validation metrics involving filtered SWOT data (e.g., rmse) are all np.nan
    #       For lakes that do not have gauge data: validation metrics such as rmse are np.nan, 
    #                 but n_while, n_while_r2, and retention metrics may have values. 
    
    # Concatenate df_lake_time_series by this df, df_lake_stats by df_this_lake_stats, 
    # and df_lake_heuristic_thresholds by df_heuristic_thresholds
    df_lake_stats = pd.concat([df_lake_stats, df_this_lake_stats], ignore_index=True) #ignore_index=True resets the index.
    df_lake_heuristic_thresholds = pd.concat([df_lake_heuristic_thresholds, df_heuristic_thresholds], ignore_index=True) 
    df_lake_time_series = pd.concat([df_lake_time_series, df], ignore_index=True) #Nothing is added if df is empty.    
    # Note: for df_lake_time_series, if the lake has no valid SWOT data (i.e., df is empty), no record is added. 
    #       for df_lake_stats, if the lake has no SWOT data, lake_id will be kept, but other attributes are nan.  
    #       for df_lake_heuristic_thresholds, if the lake has no SWOT data, lake_id will be kept as well. 
    
    
    # Continue to finish the plot.  
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
    
    # Only for visual reference. 
    mask = (df["xtrk_dist"].abs() < 10000) | (df["xtrk_dist"].abs() > 60000)
    ax.plot(df[mask].datetime, df[mask].wse,
            label='xtrk_dist out [10, 60]km', marker='s', linestyle='', markersize=15,
            markerfacecolor='none', markeredgecolor='black')
    
    # Plot filtered result (use wse_ajusted to account for possible cycle adjustment)
    ax.errorbar(df_filtered.datetime, df_filtered.wse_adjusted, df_filtered.wse_u,
           label='heuristic filter', color='black', marker='o',
           markersize=4, capsize=3, linestyle='--')    

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    fig.autofmt_xdate()
    ax.set_xlim(pd.to_datetime(start_time), pd.to_datetime(end_time))  
    if len(df_filtered) >= 1 and df_filtered['wse'].notna().any(): #at least one non-Nan value
        range_wse = np.nanmax(df_filtered.wse)-np.nanmin(df_filtered.wse)
        plt.ylim(np.nanmin(df_filtered.wse)-range_wse*2, np.nanmax(df_filtered.wse)+range_wse*2)
    elif len(df) >= 1 and df['wse'].notna().any():
        plt.ylim(np.nanmin(df.wse), np.nanmax(df.wse)) 
    #plt.ylim(20, 40)
    
    # Axis labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('WSE (m)', fontsize=12)
    ax.set_title('Lake ID ' + str(feature_id) + ' Time Series Plot. Filter: ' + filter_type + '. Gauge: ' + str(gauge_source))
          
    # Add statistics as a text box
    if gauge_source is not None: # With gauge data    
        textstr = f'nrmse_unfiltered = {rmse_unfiltered:.3f}\nrmse = {rmse:.3f}'
        props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    #ax.legend()  
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.) #legend outside of the plot
    
    #Save the plot
    plt.savefig(work_dir+r'\Plots\Attmpt_'+str(filter_attempt)+'_lakeID_'+str(feature_id)+'_'+filter_type+'_'+script_version+'.png', bbox_inches='tight') 
    # Do NOT call plt.show(), so nothing opens in Spyder
    if show_filtering_evolution == 'no':
        plt.close()  # Optional: frees up memory if many plots are generated
    
## Optional: if we want to save Hydrocron time series into local disk, check the line below. 
#if SP_retrieval_method == 'Hydrocron':
#    df_Hydrocron.to_csv(work_dir+'/df_Hydrocron.csv', index=False) 



"""
Present summative statistics for all validated lakes

Recall: major outputs from sections above include:
    df_lake_time_series
    df_lake_stats   
"""
# Compute the proportion of lakes for each filtering scenario
proportion_filter_attempt_1 = (len(df_lake_stats[(df_lake_stats['filter_attempt']=='1_strict')]) / len(df_lake_stats))*100
print('Proportion (%) of lakes with strict filter (attempt 1): ' + str(proportion_filter_attempt_1))

proportion_filter_attempt_2 = (len(df_lake_stats[(df_lake_stats['filter_attempt']=='2_lenient')]) / len(df_lake_stats))*100
print('Proportion (%) of lakes with lenient filter (attempt 2): ' + str(proportion_filter_attempt_2))

proportion_filter_attempt_3 = (len(df_lake_stats[(df_lake_stats['filter_attempt']=='3_tukey')]) / len(df_lake_stats))*100
print('Proportion (%) of lakes with baseline tukey IQR filter (attempt 3): ' + str(proportion_filter_attempt_3))

proportion_filter_attempt_4 = (len(df_lake_stats[(df_lake_stats['filter_attempt']=='4_none')]) / len(df_lake_stats))*100
print('Proportion (%) of lakes with no filter (attempt 4): ' + str(proportion_filter_attempt_4))

proportion_filter_no_data = (len(df_lake_stats[(df_lake_stats['filter_attempt']=='no data')]) / len(df_lake_stats))*100
print('Proportion (%) of lakes with no data (attempt 0): ' + str(proportion_filter_no_data))

print('Total (100%): ' + str(proportion_filter_attempt_1+proportion_filter_attempt_2+proportion_filter_attempt_3+proportion_filter_attempt_4+proportion_filter_no_data))

# Compute summary validation statistics for the reference lakes, defined as those that have:
# (1) valid data in the time series (df)
# (2) gauge data, 
# (3) successfully filtered by '1_strict' or '2_lenient' attempt. 
# df_lake_stats = df_lake_stats[df_lake_stats['filter_attempt'].isin(['1_strict', '2_lenient'])]
df_lake_stats = df_lake_stats[df_lake_stats['filter_attempt'].isin(['1_strict'])] # Just include 1_strict only for now. 
unique_lakes = df_lake_stats.loc[df_lake_stats['rmse'].notna(), 'lake_id'].tolist() 
if unique_lakes: # If there is at least one lake with gauge data (i.e., rmse is not NaN)
    # Create the figure with 3 subplots
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()  # This will make axes[0], axes[1], axes[2], axes[3] each refer to one subplot.
    
    # Scatter plot for gauge WSEs vs SWOT WSEs for each of the lakes
    # Initialize the axis range for the lake WSE plot. 
    min_val = []
    max_val = []
    # Initialize WSE anomalies (w.r.t. each lake WSE mean to condense axis range for visualization)
    SWOT_wse_anomalies = []
    SWOT_wse_anomalies_unfiltered = []
    SWOT_wse_anomalies_baseline1 = []
    SWOT_wse_anomalies_baseline2 = []
    gauge_wse_anomalies = []        
    for lake_id in unique_lakes: #Loop through each lake
        subset = df_lake_time_series[df_lake_time_series['lake_id'] == lake_id]
        wse_mean = np.nanmean(subset['wse_adjusted'])
        swot_anom = subset['wse_adjusted'] - wse_mean
        swot_anom_unfiltered = subset['wse'] - wse_mean
        swot_anom_baseline1 = subset['wse_baseline1'] - wse_mean
        swot_anom_baseline2 = subset['wse_baseline2'] - wse_mean
        gauge_anom = subset['gauge_wse_bias_corrected'] - wse_mean
        # Extend flat sequences
        SWOT_wse_anomalies.extend(swot_anom.tolist())
        SWOT_wse_anomalies_unfiltered.extend(swot_anom_unfiltered.tolist())
        SWOT_wse_anomalies_baseline1.extend(swot_anom_baseline1.tolist())
        SWOT_wse_anomalies_baseline2.extend(swot_anom_baseline2.tolist())
        gauge_wse_anomalies.extend(gauge_anom.tolist())
        # Append min/max from current anomalies only (not the whole list)
        min_val.append(min(np.nanmin(gauge_anom), np.nanmin(swot_anom)))
        max_val.append(max(np.nanmax(gauge_anom), np.nanmax(swot_anom)))
        #if np.nanmax(gauge_anom.tolist()) > 40:
        #    print(lake_id) #4520076683, 4530267193, 7421035423, 4540003332
    
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
    axes[0].set_title('Lake WSE (wrt each lake median level)')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].legend() 
        
    # Scatter plot for gauge daily WSE variability vs SWOT daily WSE variability
    # Retrieved only reference lakes defined above    
    subset = df_lake_stats[df_lake_stats['rmse'].notna()]
    axes[1].scatter(subset.var_gauge_daily, subset.var_swot_daily_unfiltered, color='gray', \
                    label='raw SP', s=50, linewidth=0, alpha=0.4) # unfiltered result
    axes[1].scatter(subset.var_gauge_daily, subset.var_swot_daily_baseline1, color='blue', \
                    label='stringent CNES baseline', s=50, linewidth=0, alpha=0.4) # baseline 1 result
    axes[1].scatter(subset.var_gauge_daily, subset.var_swot_daily_baseline2, color='orange', \
                    label='lenient CNES baseline', s=50, linewidth=0, alpha=0.4) # baseline 2 result
    #axes[1].scatter(subset.var_gauge_daily, subset.var_swot_daily, color='r', \
    #                label='customized filter', s=50, linewidth=0, alpha=0.4) # filtered result
    axes[1].scatter(subset.var_gauge_daily, subset.var_swot_daily, color='r', \
                    label='customized filter', s=100, linewidth=3, marker='+', alpha=0.4) # filtered result
    # Add 1:1 diagonal line
    #min_val = min(np.nanmin(subset.var_gauge_daily), np.nanmin(subset.var_swot_daily_unfiltered), \
    #              np.nanmin(subset.var_swot_daily))  
    # Filter out NaNs and zeros
    vals_min = [np.nanmin(subset.var_gauge_daily), np.nanmin(subset.var_swot_daily_unfiltered), np.nanmin(subset.var_swot_daily)]
    valid_vals = [v for v in vals_min if not np.isnan(v) and v > 0]
    min_val = min(valid_vals) if valid_vals else np.nan   
    max_val = max(np.nanmax(subset.var_gauge_daily), np.nanmax(subset.var_swot_daily_unfiltered), \
                  np.nanmax(subset.var_swot_daily))
    #test
    #lake_ids_var = subset.loc[subset["var_gauge_daily"] == 0, "lake_id"] #4530746652
    
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
    print(np.nanpercentile(abs(subset.var_swot_daily_unfiltered - subset.var_gauge_daily)/subset.var_gauge_daily, 68))
    print(np.nanpercentile(abs(subset.var_swot_daily_baseline1 - subset.var_gauge_daily)/subset.var_gauge_daily, 68))
    print(np.nanpercentile(abs(subset.var_swot_daily_baseline2 - subset.var_gauge_daily)/subset.var_gauge_daily, 68))
    print(np.nanpercentile(abs(subset.var_swot_daily - subset.var_gauge_daily)/subset.var_gauge_daily, 68))
    
    print(np.nanpercentile(abs(subset.var_swot_daily_unfiltered - subset.var_gauge_daily)/subset.var_gauge_daily, 50))
    print(np.nanpercentile(abs(subset.var_swot_daily_baseline1 - subset.var_gauge_daily)/subset.var_gauge_daily, 50))
    print(np.nanpercentile(abs(subset.var_swot_daily_baseline2 - subset.var_gauge_daily)/subset.var_gauge_daily, 50))
    print(np.nanpercentile(abs(subset.var_swot_daily - subset.var_gauge_daily)/subset.var_gauge_daily, 50))
    
    ## Another option is to compare variability based on paired swot and gauge duplets (without interpolation), 
    ## but this may not represent the full scale of lake WSE variability. 
    # axes[1].scatter(subset.var_gauge_unfiltered, subset.var_swot_unfiltered, color='gray', \
    #                 label='raw SP', s=50, linewidth=0, alpha=0.6) # unfiltered result
    # axes[1].scatter(subset.var_gauge_baseline1, subset.var_swot_baseline1, color='blue', \
    #                 label='stringent CNES baseline', s=50, linewidth=0, alpha=0.6) # baseline 1 result
    # axes[1].scatter(subset.var_gauge_baseline2, subset.var_swot_baseline2, color='orange', \
    #                 label='lenient CNES baseline', s=50, linewidth=0, alpha=0.6) # baseline 2 result
    # axes[1].scatter(subset.var_gauge, subset.var_swot, color='r', \
    #                 label='customized filter', s=50, linewidth=0, alpha=0.8) # filtered result
    # # Add 1:1 diagonal line
    # min_val = min(np.nanmin(subset.var_gauge_unfiltered), np.nanmin(subset.var_swot_unfiltered), \
    #               np.nanmin(subset.var_swot))
    # max_val = max(np.nanmax(subset.var_gauge_unfiltered), np.nanmax(subset.var_swot_unfiltered), \
    #               np.nanmax(subset.var_swot))
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
    # print(np.nanpercentile(abs(subset.var_swot_unfiltered - subset.var_gauge_unfiltered)/subset.var_gauge_unfiltered, 68))
    # print(np.nanpercentile(abs(subset.var_swot_baseline1 - subset.var_gauge_baseline1)/subset.var_gauge_baseline1, 68))
    # print(np.nanpercentile(abs(subset.var_swot_baseline2 - subset.var_gauge_baseline2)/subset.var_gauge_baseline2, 68))
    # print(np.nanpercentile(abs(subset.var_swot - subset.var_gauge)/subset.var_gauge, 68))
    
    # Plot CDF of correlation
    # Sort values for CDF
    correlation_unfiltered_sorted = np.sort(subset['correlation_unfiltered'].dropna().values)
    correlation_baseline1_sorted = np.sort(subset['correlation_baseline1'].dropna().values)
    correlation_baseline2_sorted = np.sort(subset['correlation_baseline2'].dropna().values)
    correlation_filtered_sorted = np.sort(subset['correlation'].dropna().values)
    # Compute empirical CDF values
    cdf_correlation_unfiltered = np.linspace(0, 1, len(correlation_unfiltered_sorted), endpoint=False)
    cdf_correlation_baseline1 = np.linspace(0, 1, len(correlation_baseline1_sorted), endpoint=False)
    cdf_correlation_baseline2 = np.linspace(0, 1, len(correlation_baseline2_sorted), endpoint=False)
    cdf_correlation_filtered = np.linspace(0, 1, len(correlation_filtered_sorted), endpoint=False)
    # Plot
    axes[2].plot(correlation_unfiltered_sorted, cdf_correlation_unfiltered, label='raw SP', color='gray', alpha=0.6)
    axes[2].plot(correlation_baseline1_sorted, cdf_correlation_baseline1, label='stringent CNES baseline', color='blue', alpha=0.6)
    axes[2].plot(correlation_baseline2_sorted, cdf_correlation_baseline2, label='lenient CNES baseline', color='orange', alpha=0.6)
    axes[2].plot(correlation_filtered_sorted, cdf_correlation_filtered, label='customized filter', color='red', alpha=0.8)
    # Axis labels and scaling
    axes[2].set_xlim(-1, 1) 
    axes[2].set_ylim(0, 1) 
    axes[2].set_xscale('linear')
    axes[2].set_yscale('linear')
    axes[2].set_xlabel('Correlation (Pearson)')
    axes[2].set_ylabel('CDF')
    axes[2].set_title('CDF curves of correlation')
    axes[2].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[2].legend() 
    
    # Plot CDF of retention rates
    # Sort values for CDF    
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
    plt.savefig(work_dir+'\Plots\Lake_validation_stats'+'_'+script_version+'.png')    
       
    
    # Explore the relationship between ice duration and seasonal variability error. 
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
    plt.savefig(work_dir+'\Plots\Ice_error_scatter'+'_'+script_version+'.png') 
    
    # Box plot
    bins = np.arange(0, 1.01, 0.1)
    labels = [f'{round(bins[i], 1)}–{round(bins[i+1], 1)}' for i in range(len(bins)-1)]
    # Add binned ice_duration column
    df_lake_stats['ice_duration_bin'] = pd.cut(
        df_lake_stats['ice_duration'],
        bins=bins,
        labels=labels,
        include_lowest=True
        ) #NaN will be ignored.     
    try: # Error may arise if rel_err_var has inf value due to 0 var_gauge_daily. 
        sns.boxplot(x='ice_duration_bin', y='rel_err_var', data=df_lake_stats) #Nan will be ignored. 
        plt.xlabel('Binned Ice Duration')
        plt.ylabel('Relative Error in Variability')
        plt.title('Boxplot of Relative Error vs Ice Duration')
        plt.xticks(rotation=45)
        plt.tight_layout()
        #plt.show()
        plt.savefig(work_dir+'\Plots\Ice_error_box_'+script_version+'.png') 
    except Exception as e:
        # skip plotting if any error occurs
        print(f"Skipping boxplot due to error: {e}")
    
    




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

plt.savefig(work_dir+'\Plots\Threshold_sensitivity_test'+'_'+script_version+'.png') 





'''
Random Forest test of feature importance for |swot_error|. 

Tested features include:
    xovr_cal_q
    xtrk_dist
    quality_f
    wse_std
    wse_u
    ice_clim_f

# Based on all data, xtrk_dist turned out to be another important feature,
# probably related to specular ringing or other errors near the nadir. 
'''
if 'gauge_wse_bias_corrected' in df_lake_time_series.columns: #if there are at least lake with gauge data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import PartialDependenceDisplay
    from sklearn.model_selection import train_test_split
    
    # Compute |wse error|
    # Initialize containers
    SWOT_error = pd.Series() # store absolute WSE error (float)
    X_RF = pd.DataFrame() # store feature matrix
    
    # Extract unique lake_ids from df_lake_time_series where both gauge_wse_bias_corrected and wse 
    # have at least one non-NaN value:
    valid_lakes = df_lake_time_series.groupby('lake_id').filter(
        lambda g: g['gauge_wse_bias_corrected'].notna().any() and g['wse'].notna().any()
    )['lake_id'].unique()
    
    # Loop through valid lakes and accumulate SWOT error and features
    for lake_id in valid_lakes:
        subset = df_lake_time_series[df_lake_time_series['lake_id'] == lake_id]        
        
        swot_error = np.abs(subset['wse'] - subset['gauge_wse_bias_corrected'])        
        SWOT_error = pd.concat([SWOT_error,  swot_error], ignore_index=True)
        
        x_rf = subset[['xovr_cal_q','xtrk_dist','quality_f','wse_std','wse_u','ice_clim_f']]
        X_RF = pd.concat([X_RF,  x_rf], ignore_index=True) 
        
    ## Select features to predict SWOT WSE error (all selected here)
    #X_RF = X_RF[['xovr_cal_q','xtrk_dist','quality_f','wse_std','wse_u','ice_clim_f']]
    
    # Take the absolute value of xtrk_dist
    X_RF.xtrk_dist=np.abs(X_RF.xtrk_dist)  
    
    # Removes observations with extreme errors
    #mask = SWOT_error < 30 
    #mask = (SWOT_error < 30) & (X_RF['ice_clim_f'] >= 0)
    mask = (SWOT_error >= 0) & (X_RF['ice_clim_f'] >= 0)
    X_RF = X_RF[mask]
    SWOT_error = SWOT_error[mask]
    #plt.plot(X_RF.xtrk_dist, SWOT_error,'x')
    #plt.plot(X_RF.wse_std, SWOT_error,'x')
    
    # Split the data into training and testing sets, e.g., two-thirds training and one-third testing
    X_train, X_test, y_train, y_test = train_test_split(X_RF, SWOT_error, test_size = 0.33)
    
    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Compute feature importances
    importances = model.feature_importances_
    # Generate a dataframe ranking features by their contribution
    feature_importance_df = pd.DataFrame({
         'Feature': X_RF.columns,
         'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Display the feature importance in descending order.
    print(feature_importance_df)
    
    # Plot and save the feature importance result
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.savefig(work_dir+'\Plots\RF_feature_importance'+'_'+script_version+'.png') 
    
    # Show dependence for all features
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X_RF, X_RF.columns, ax=ax)
    plt.show()
    #If the PDP line slopes upward for a feature, higher values of that feature tend to increase the predicted error.
    #If it slopes downward, higher values tend to decrease the predicted error.
    plt.savefig(work_dir+'\Plots\RF_feature_dependence'+'_'+script_version+'.png') 
