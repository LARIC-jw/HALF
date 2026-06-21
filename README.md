## Heuristic Adaptive Lake Filter (HALF) scripts

The `scripts/` folder contains the Python implementation of the Heuristic Adaptive Lake Filter (HALF) for SWOT LakeSP time series.

* `half_v1_functions.py` contains the core reusable functions, including lake-specific threshold calibration, threshold application, iterative low-pass filtering, intra-cycle cross-pass bias correction, validation metrics, and daily interpolation utilities.

* `half_v1_run.py` provides a user-facing workflow for applying HALF to one or more PLD lake IDs queried directly through the Hydrocron API. This script does not require local LakeSP input files. Users only need to specify the target lake IDs, LakeSP collection, time window, and output directory. The script writes the filtered LakeSP observation table, threshold-summary table, and per-lake diagnostic plots.

* `half_v1_validation.py` provides the validation workflow used to compare raw and filtered LakeSP WSE against paired gauge records and to reproduce the validation statistics and figures reported in Trudel et al. (2026, in review). This script requires the validation dataset described below.

Users should review and edit the configuration dictionaries near the beginning of each script before running, including the LakeSP collection, lake IDs, time window, input/output directories, and filtering parameters/options. The scripts are configured to run with repository-relative paths by default, but users may reorganize the folders or use absolute local paths by updating the relevant settings in `RUN_CONFIG`. The individual script files provide detailed definitions, methodological notes, parameter descriptions, and implementation details.

## Data availability

Due to GitHub file-size limits, the compiled gauge dataset used as validation input for `half_v1_validation.py` in Trudel et al. (2026, in review) is not stored directly in this GitHub repository. It is available through Zenodo:

**Zenodo dataset:** https://doi.org/10.5281/zenodo.20781336

After downloading the Zenodo archive, users may place or symlink the data folder into the repository root using the expected folder name:

```text
gauge_data/
```

The validation script expects this folder by default when `work_dir = "."` and the script is run from the repository root. Alternatively, users may store the downloaded data elsewhere and edit the path settings in `RUN_CONFIG` to point to the corresponding local directory.

This GitHub repository provides the code and documentation used to process, filter, validate, and analyze the associated data.

## Version and authors

HALF version 1. Last updated: June 20, 2026.

We welcome feedback, questions, and suggestions. Please contact Jida Wang ([jidaw@illinois.edu](mailto:jidaw@illinois.edu)) and Mélanie Trudel ([melanie.trudel@usherbrooke.ca](mailto:melanie.trudel@usherbrooke.ca)).

Citation: Trudel, M., Wang, J., Biancamaria, S., Harlan, M. E., Shah, D., Gao, H., Collins, E., Getirana, A., Song, C., Reis Alencar Oliveira, R., Gosset, M., Rodrigues Martins, E. S., Fleischmann, A., Hymans, D., Grippa, M., Girard, F., Kergoat, L., Pottier, C., Fjørtoft, R., Oubanas, H., & Pavelsky, T. M. (2026). A Heuristic Adaptive Filter for SWOT Lake Vector Data Products. *Geophysical Research Letters*, in review.
