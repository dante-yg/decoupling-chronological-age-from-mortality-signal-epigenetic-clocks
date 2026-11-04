# Pipeline details

This page gives the file paths, the run order with each step's inputs and outputs, and the main settings. For an overview, see the [main README](../README.md).

## File paths

The code uses absolute Windows paths. Replace these roots with your own locations:

| Path in the code | Meaning | Used by |
|---|---|---|
| `C:\F\base_3` | **Project root.** Holds `data/`, `predictions/`, `saved_models/`, `scalers/`, `figures/`, `logs/` | Almost every step |
| `C:\F\base_2\data` | Second copy of the same input files | `integrated_pipeline_cpg`, both stacking notebooks |
| `C:\Yugan_Gogul\...` | Raw and full-matrix methylation files | `01_preprocessing/`, GrimAge probe audit |
| `F:\dbgap\` | Full methylation matrices (`FHS_methylation_merged.parquet`, `WHI_full_imputed.parquet`) | `03_epigenetic_clocks/`, cell-composition analysis |
| `D:\...`, `C:\Neural_network_survival_WHI\` | Extra dbGaP files used only in the revision (blood counts, Gen3 exam 2 clinical, WHI BA23 N = 1,818) | `06_revision_analyses/` |
| `C:\sharing\base_3` | Same as the project root, on a second machine | GrimAge probe audit |

Notebooks without an absolute path (e.g. `clocks_comparison`, `figures_and_tables`) read `predictions/...` relative to where they are started, so **start Jupyter from the project root.**

---

## Run order

Each step reads the outputs of the step before it. `{RUN_ID}` is `V3_OPTUNA_SS` for the CpG + clinical model and `V3_OPTUNA_SS_CPG` for the CpG-only model. `{split}` is `train`, `val`, `test` or `whi`.

| # | File | Reads | Writes |
|---|---|---|---|
| 1 | `01_preprocessing/01_fhs_raw_processing.py` | FHS raw 450K files (Gen 3, JHU, UMN batches) | Cleaned FHS methylation files |
| 2 | `01_preprocessing/02_whi_raw_processing_knn_imputation.py` | WHI raw 450K file + WHI metadata | `WHI_imputed_knn5` (probes > 10 % missing dropped, KNN k = 5) |
| 3 | `01_preprocessing/03_combat_batch_correction.py` | Full FHS + WHI matrices | ComBat-corrected full matrix (neuroCombat on M-values, age and sex protected) |
| — | *(not in this folder; see `data/README.md`)* | Outputs of steps 1–3 + phenotype files | `data/FHS_non_normalized.csv`, `data/WHI_non_normalized.csv`, `data/Disease_data_with_censoring.csv` |
| 4 | `02_model_training/integrated_pipeline.ipynb` | `data/FHS_non_normalized.csv`, `data/WHI_non_normalized.csv` | 60/20/20 split (seed 42); per-model predictions `predictions/{model}_{split}_{RUN_ID}.csv`; out-of-fold stacking inputs `predictions/stacking_{split}_{RUN_ID}.csv`; `base_model_cindex_summary_{RUN_ID}.csv`; `base_model_feature_importance_{RUN_ID}.csv`; models in `saved_models/`, scalers in `scalers/` |
| 5 | `02_model_training/integrated_pipeline_cpg.ipynb` | Same inputs, CpGs only | Same outputs with the `_CPG` run ID |
| 6 | `02_model_training/stacking_coxmlp.ipynb` and `stacking_coxmlp_cpg.ipynb` | `predictions/stacking_{split}_{RUN_ID}.csv` | `predictions/stacking_predictions_{split}_{RUN_ID}.csv` |
| 7 | `03_epigenetic_clocks/estimate_biological_clocks.py` | Full methylation matrices | `predictions/fhs_biological_clocks.csv`, `predictions/whi_biological_clocks.csv` |
| 8 | `04_main_analyses/coxnet_age_predictor.ipynb` | CoxNet predictions | `predictions/coxnet_age_predictions_{split}_{RUN_ID}.csv` (risk score rescaled to years) |
| 9 | `04_main_analyses/all_models_risk_score_survival.ipynb` | All model predictions | `predictions/all_models_risk_score_hazard_ratios_{adjusted,unadjusted}.csv` |
| 10 | `04_main_analyses/clocks_comparison.ipynb` | Steps 7–8 + SVM predictions | `predictions/clocks_comparison_hazard_ratios_{adjusted,unadjusted}.csv`, `clocks_comparison_cindex.csv`, `clocks_comparison_age_correlations.csv` |
| 11 | `05_figures_and_tables/figures_and_tables.ipynb` | Everything in `predictions/` + `data/` | All manuscript figures and Tables 1–3 in `figures/` |

Steps 4–5 run Optuna hyperparameter tuning and take the longest. Steps 8–11 only read saved predictions and run in minutes.

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
