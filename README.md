This folder serves as a repository for tracking progress on improving the LakeSP filtering.
Latest updated: 11/21/2025

Files include:
> LakeSP_heuristic_filter_basinArray_joblib_08212025.py
THIS IS THE MAIN AND LATEST executing script to apply the LakeSP heuristic filter on individual lake time seriese csv files downloaded from Hydrocron.

This script was modified from LakeSP-heuristic-filter-with-validation-v10-10022025.py to allow joblib parallel per-lake: 
    1. Parallelize *within each basin* over lakes using joblib. 
    2. Return one output CSV per basin.
    3. No need to perform gauge validation as in LakeSP-heuristic-filter-with-validation-vxx.py

> LakeSP-heuristic-filter-with-validation-v10-10022025.py
Latest script to implement the heuristic filter (version 10, updated October 2, 2025) on the validation data.
This script is not for global batch processing, but was instead used to develop the filter based on validation data. 

> customized_functions.py
All functions needed for executing the heuristic filter.

> df_Hydrocron.zip
Latest zipped CSV file containing LakeSP time series for about 1070 tested lakes, retrieved from the Hydrocron API

> gauge_data: Latest gauge data for validating the LakeSP filter.
Due to size, the gauge data is shared through this Google Drive:
https://drive.google.com/drive/folders/1etGNuR0eQdQ-TFFH7L2a6saO0IfRUu8s?usp=sharing

> LakeSP-filter-results.pptx
Latest results

> Other files and folders
Old versions for the record
