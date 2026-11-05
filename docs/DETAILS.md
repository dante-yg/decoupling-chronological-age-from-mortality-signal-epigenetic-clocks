# Pipeline details

This page gives the file paths, the run order with each step's inputs and outputs, and the main settings. For scientific background, see [ABOUT.md](ABOUT.md); for an overview, see the [main README](../README.md).

## File paths

All project paths are centralized in [`config.py`](../config.py) at the repository root and resolved dynamically. All output directories (`predictions/`, `saved_models/`, `scalers/`, `figures/`, `logs/`, `reviews/`) are created automatically on import if they do not exist.

Paths can be customized by setting environment variables (both Python scripts and the R deconvolution script read these):

| Environment Variable | Default Location | Description |
|---|---|---|
| `EPICLOCK_ROOT` | Repository root | Base project root directory |
| `EPICLOCK_DATA_DIR` | `$EPICLOCK_ROOT/data` | Processed input files (`FHS_non_normalized.csv`, etc.) |
| `EPICLOCK_PREDICTIONS_DIR` | `$EPICLOCK_ROOT/predictions` | Saved predictions and intermediate results |
| `EPICLOCK_SAVED_MODELS_DIR` | `$EPICLOCK_ROOT/saved_models` | Trained models (`.joblib`, PyTorch `.pt`) |
| `EPICLOCK_SCALERS_DIR` | `$EPICLOCK_ROOT/scalers` | Fitted `StandardScaler` objects |
| `EPICLOCK_FIGURES_DIR` | `$EPICLOCK_ROOT/figures` | Output figures (PDF, PNG, TIF) and summary tables |
| `EPICLOCK_LOGS_DIR` | `$EPICLOCK_ROOT/logs` | Execution log files |
| `EPICLOCK_REVIEWS_DIR` | `$EPICLOCK_ROOT/reviews` | Reviewer sensitivity analysis outputs |
| `EPICLOCK_RAW_DIR` | `$EPICLOCK_DATA_DIR/raw` | Raw dbGaP controlled-access files (for `01_preprocessing/`) |

### Historical path mappings
For provenance and reference, previously hard-coded analysis paths from the author's machines have been mapped to config variables as follows:
- `C:\F\base_3` and `C:\sharing\base_3` → `PROJECT_ROOT`
- `C:\F\base_2\data` → `DATA_DIR` (duplicate copy mapped to standard data folder)
- `C:\Yugan_Gogul\...` and `F:\dbgap\...` → `DATA_DIR` / `RAW_DIR`
- `D:\...` and `C:\Neural_network_survival_WHI\...` → `RAW_DIR` / `DATA_DIR`
- R scripts (`06_revision_analyses/.../deconvolution.R`) read the same environment variables via `Sys.getenv()`.

Start Jupyter from the project root so notebooks automatically find `config.py`.

---

## Run order

Each step reads the outputs of the step before it. `{RUN_ID}` is `V3_OPTUNA_SS` for the CpG + clinical model and `V3_OPTUNA_SS_CPG` for the CpG-only model. `{split}` is `train`, `val`, `test` or `whi`.

| # | File | Reads | Writes |
|---|---|---|---|
| 1 | `01_preprocessing/01_fhs_raw_processing.py` | FHS raw 450K files (Gen 3, JHU, UMN batches) | Cleaned FHS methylation files |
| 2 | `01_preprocessing/02_whi_raw_processing_knn_imputation.py` | WHI raw 450K file + WHI metadata | `WHI_imputed_knn5` (probes > 10 % missing dropped, KNN k = 5) |
| 3 | `01_preprocessing/03_combat_batch_correction.py` | Full FHS + WHI matrices | ComBat-corrected full matrix (neuroCombat on M-values, age and sex protected) |
| 4 | `01_preprocessing/04_build_model_inputs.py` | Outputs of steps 1–3 + phenotype files (skeleton; see file) | `data/FHS_non_normalized.csv`, `data/WHI_non_normalized.csv`, `data/Disease_data_with_censoring.csv` |
| 5 | `02_model_training/integrated_pipeline.ipynb` | `data/FHS_non_normalized.csv`, `data/WHI_non_normalized.csv` | 60/20/20 split (seed 42); per-model predictions `predictions/{model}_{split}_{RUN_ID}.csv`; out-of-fold stacking inputs `predictions/stacking_{split}_{RUN_ID}.csv`; `base_model_cindex_summary_{RUN_ID}.csv`; `base_model_feature_importance_{RUN_ID}.csv`; models in `saved_models/`, scalers in `scalers/` |
| 6 | `02_model_training/integrated_pipeline_cpg.ipynb` | Same inputs, CpGs only | Same outputs with the `_CPG` run ID |
| 7 | `02_model_training/stacking_coxmlp.ipynb` and `stacking_coxmlp_cpg.ipynb` | `predictions/stacking_{split}_{RUN_ID}.csv` | `predictions/stacking_predictions_{split}_{RUN_ID}.csv` |
| 8 | `03_epigenetic_clocks/estimate_biological_clocks.py` | Full methylation matrices | `predictions/fhs_biological_clocks.csv`, `predictions/whi_biological_clocks.csv` |
| 9 | `04_main_analyses/coxnet_age_predictor.ipynb` | CoxNet predictions | `predictions/coxnet_age_predictions_{split}_{RUN_ID}.csv` (risk score rescaled to years) |
| 10 | `04_main_analyses/all_models_risk_score_survival.ipynb` | All model predictions | `predictions/all_models_risk_score_hazard_ratios_{adjusted,unadjusted}.csv` |
| 11 | `04_main_analyses/clocks_comparison.ipynb` | Steps 8–9 + SVM predictions | `predictions/clocks_comparison_hazard_ratios_{adjusted,unadjusted}.csv`, `clocks_comparison_cindex.csv`, `clocks_comparison_age_correlations.csv` |
| 12 | `05_figures_and_tables/figures_and_tables.ipynb` | Everything in `predictions/` + `data/` | All manuscript figures and Tables 1–3 in `figures/` |

Steps 5–6 run Optuna hyperparameter tuning and take the longest. Steps 9–12 only read saved predictions and run in minutes.

The revision scripts in `06_revision_analyses/` all start from the saved predictions, models and scalers of steps 4–10. Each script's header lists what it reads and writes, and none of them retrain the published models.

---

## Main settings

| Setting | Value | Where |
|---|---|---|
| Random seed | 42 (NumPy, PyTorch, splits, CV, models) | Training notebooks |
| FHS split | 60 / 20 / 20, stratified on event; train 2,095, val 699, test 699 | `integrated_pipeline*.ipynb` |
| Scaling | `StandardScaler` fit on the training split only, then applied to val, test and WHI | Same |
| Tuning | Optuna, 10–15 trials per learner, 3-fold stratified CV on the training split | Same |
| Stacking | 5-fold out-of-fold predictions → CoxMLP meta-learner | `stacking_coxmlp*.ipynb` |
| Final model | CoxNet, CpG + clinical (`V3_OPTUNA_SS`); "risk score" = its linear predictor | — |
| Clock comparison | Cox models unadjusted and adjusted for age and sex (sex dropped in WHI, which is all female). HRs per 1 year of predicted age (original Table 3); per-SD HRs in Figure 2 and the rebuilt Table 3 | `clocks_comparison.ipynb`, `figures_and_tables.ipynb`, `reviewer_table3_rebuild.py` |

---
