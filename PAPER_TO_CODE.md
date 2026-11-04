# Where each result comes from

`F&T` = `05_figures_and_tables/figures_and_tables.ipynb`. That notebook saves files under older working names, given in the last column so you can find them in `figures/`.

## Original submission

| Manuscript item | Computed in | Drawn / formatted in | Saved as |
|---|---|---|---|
| **Table 1** Cohort characteristics | — | F&T, cell 9 | `Table_1_Cohort_Demographics.csv` |
| **Table 2** C-index, all learners × feature sets × cohorts | `02_model_training/` (all four notebooks) | F&T, cell 10 | `Table_2_CIndex_Summary.csv` |
| **Table 3** HRs for EpiClock and four clocks, unadjusted and age/sex-adjusted | `04_main_analyses/clocks_comparison.ipynb` | F&T, cell 11 | `Table_3_Hazard_Ratios_Summary.csv` |
| **Figure 1** CoxNet-derived age vs chronological age | `04_main_analyses/coxnet_age_predictor.ipynb` | F&T, cell 7 | `Figure_4_Age_vs_EpiAge_Scatter` |
| **Figure 2** Forest plot, adjusted HRs per 1 SD | `04_main_analyses/clocks_comparison.ipynb` | F&T, cell 8 | `Figure_5_Mortality_HR_Forest` |
| **Fig. S1** ComBat PCA | `01_preprocessing/03_combat_batch_correction.py` | F&T, cell 12 | `Figure_S1_ComBat_PCA` |
| **Fig. S2** Risk score vs age, all learners | `02_model_training/` | F&T, cell 13 | `Figure_S2_PerModel_Age_Scatters` |
| **Fig. S3** Permutation feature importance | `02_model_training/integrated_pipeline*.ipynb` | F&T, cell 14 | `Figure_S3_Feature_Importance` |
| **Fig. S4** TD-AUROC and KM by cohort | `02_model_training/` | F&T, cell 15 | `Figure_S4_Supp_AUROC_and_KM` |
| **Fig. S5** TD-AUROC, FHS test | `02_model_training/` | F&T, cell 5 | `Figure_2_TD_AUROC_Test` |
| **Fig. S6** KM and CVD incidence by risk tertile | `02_model_training/` + `data/Disease_data_with_censoring.csv` | F&T, cell 6 | `Figure_3_KM_Disease_Incidence` |
| Section 3.4 HRs for CoxNet and SVM, all learners | `04_main_analyses/all_models_risk_score_survival.ipynb` | — | `predictions/all_models_risk_score_hazard_ratios_*.csv` |

## Revision analyses (`06_revision_analyses/`)

| Reviewer comment | Folder | Main script(s) |
|---|---|---|
| R1 C1 — benchmark against age alone (ΔC, LRT, PH checks, TD-AUROC) | `r1_c1_age_benchmark/` | `reviewer1_age_benchmark.py` |
| R1 C3 — preprocessing documentation, probe annotation, blood-cell composition | `r1_c3_preprocessing_and_cell_composition/` | `reviewer1_preprocessing_cells.py` → `deconvolution.R` → `reviewer1_cells_sensitivity.py` |
| R1 C4 — ComBat sensitivity (none / before split / train-only) | `r1_c4_combat_sensitivity/` | `reviewer1_combat_sensitivity.py` |
| R1 C5 + R3 — rebuilt Table 3 (per 1 SD), revised Figure 2, per-learner TD-AUROC | `r1_c5_r3_table3_rebuild/` | `reviewer_table3_rebuild.py` |
| R1 C10 — calibration (O/E, slope, ICI, Brier, WHI recalibration) | `r1_c10_calibration/` | `reviewer1_calibration_final_model.py` (final model), `reviewer1_calibration.py` (all models); plots: `reviewer1_calibration_plot_deciles.py`, `reviewer1_calibration_enlarged_timeline.py`, `reviewer1_calibration_1818_14y.py`, `generate_side_by_side_calibration.py`, `generate_base_models_calibration_v2.py`, `generate_sklearn_style_calibration.py` |
| R1 C11–C13 — clinical-marker ΔC, age-residualized KM, age acceleration | `r1_c11_c12_c13/` | `reviewer1_c11_c12_c13.py`; age-stratified KM: `generate_comment12_figures.py` |
| Family relatedness / leakage in FHS | `r1_family_leakage/` | `reviewer1_family_leakage.py` |
| GrimAge probe availability and restored-probe sensitivity (Supp. Table S2) | `grimage_probe_audit/` | `classify_missing_probes.py` → `extract_and_restore_109.py`, `run_whi_sensitivity.py` → `generate_supp_table_s2.py` |
| WHI follow-up audit | `whi_followup_audit/` | `audit_whi_survival.py` |

## Supporting analyses (`07_supporting_analyses/`, not in the manuscript)

| Notebook | Purpose |
|---|---|
| `nested_analysis.ipynb` | Nested Cox models: does the risk score add to age + sex (ΔC by bootstrap) |
| `baseline_comparison.ipynb` | Age/sex baselines vs epigenetic models |
| `cohort_splits_comparison.ipynb` | Split balance and multicollinearity |
| `coxnet_cpg_probes_survival.ipynb`, `coxnet_cpg_probes_residual_survival.ipynb` | Each CpG on its own, raw and age/sex-residualized |
| `coxnet_age_acceleration_survival.ipynb` | CoxNet age acceleration vs mortality |
| `test_bootstrap.ipynb`, `test_hypothesis.ipynb`, `test_split_swap.ipynb` | Stability across bootstrap samples, seeds and swapped splits |
