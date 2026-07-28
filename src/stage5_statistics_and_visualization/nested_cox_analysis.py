#!/usr/bin/env python
"""
Stage 5 Statistics: Nested Cox Regression Model Comparison (M1 vs M0)

This script performs a nested Cox Proportional Hazards regression comparison
between a reduced model (M0: age, sex, and clinical markers) and a full model
(M1: M0 plus the standardized EpiClock-based risk score). It evaluates the significance
of adding the EpiClock-based risk score using a Likelihood-Ratio Test (LRT) and
estimates the delta C-index along with 95 percent bootstrap confidence intervals.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from scipy.stats import chi2
from sksurv.metrics import concordance_index_censored

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

SEED = 42
np.random.seed(SEED)

def main():
    logger.info("Starting nested Cox model comparison...")
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = REPO_ROOT / "data"
    PRED_DIR = REPO_ROOT / "predictions"
    OUTPUT_DIR = REPO_ROOT / "outputs"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load FHS and WHI cohorts
    logger.info("Loading cohorts...")
    try:
        fhs_data = pd.read_csv(DATA_DIR / "FHS_non_normalized.csv", low_memory=False)
        fhs_data = fhs_data.rename(columns={"Share_ID": "patient_id"})
        whi_data = pd.read_csv(DATA_DIR / "WHI_non_normalized.csv", low_memory=False)
        whi_data = whi_data.rename(columns={"SUBJECT_ID": "patient_id"})
    except Exception as e:
        logger.error(f"Failed to load dataset files: {e}. Please ensure data files are in data/.")
        return

    # 2. Re-create splitting to align index mapping
    DROP_COLS = ["patient_id", "SUBJECT_ID", "Share_ID", "dbGaP_Subject_ID", "time", "event", "cohort", "batch", "age", "sex"]
    fhs_features = [c for c in fhs_data.columns if c not in DROP_COLS]
    whi_features = [c for c in whi_data.columns if c not in DROP_COLS]
    common_features = sorted(list(set(fhs_features).intersection(set(whi_features))))
    
    FHS = fhs_data[["patient_id", "time", "event", "age", "sex"] + common_features].copy()
    WHI = whi_data[["patient_id", "time", "event", "age", "sex"] + common_features].copy()

    FHS["event_stratify"] = FHS["event"].astype(int)
    train_val_df, test_df = train_test_split(FHS, test_size=0.20, random_state=SEED, stratify=FHS["event_stratify"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=SEED, stratify=train_val_df["event_stratify"])

    train_df = train_df.drop(columns=["event_stratify"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["event_stratify"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["event_stratify"]).reset_index(drop=True)

    # 3. Load predictions to extract risk scores
    logger.info("Loading prediction files...")
    train_preds_path = PRED_DIR / "coxnet_train_V3_OPTUNA_SS.csv"
    val_preds_path = PRED_DIR / "coxnet_val_V3_OPTUNA_SS.csv"
    test_preds_path = PRED_DIR / "coxnet_test_V3_OPTUNA_SS.csv"
    whi_preds_path = PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv"

    if not all([f.exists() for f in [train_preds_path, val_preds_path, test_preds_path, whi_preds_path]]):
        logger.error("Required modeling prediction files not found. Please run Stage 3 modeling first.")
        return

    train_preds = pd.read_csv(train_preds_path)
    val_preds = pd.read_csv(val_preds_path)
    test_preds = pd.read_csv(test_preds_path)
    whi_preds = pd.read_csv(whi_preds_path)

    def merge_predictions(df, preds):
        df = df.copy()
        df["patient_id"] = df["patient_id"].astype(int)
        preds = preds.copy()
        preds["patient_id"] = preds["patient_id"].astype(int)
        merged = pd.merge(df, preds[["patient_id", "risk_score"]], on="patient_id")
        assert len(merged) == len(df), f"Merge length mismatch: {len(df)} vs {len(merged)}"
        return merged

    train_df = merge_predictions(train_df, train_preds)
    val_df = merge_predictions(val_df, val_preds)
    test_df = merge_predictions(test_df, test_preds)
    whi_df = merge_predictions(WHI, whi_preds)

    cohorts = {
        "Test": test_df,
        "WHI": whi_df
    }

    clinical_markers = ["Total_Cholesterol", "HDL", "Glucose", "Triglycerides"]
    results_summary = {}

    for cohort_name, df_cohort in cohorts.items():
        logger.info(f"Analyzing cohort: {cohort_name.upper()}...")
        
        # Complete-case filter
        analysis_df = df_cohort[["time", "event", "age", "sex", "risk_score"] + clinical_markers].dropna().copy()
        analysis_df["event"] = analysis_df["event"].astype(int)

        # Standardize risk scores to unit variance within the cohort
        risk_sd = analysis_df["risk_score"].std()
        risk_mean = analysis_df["risk_score"].mean()
        analysis_df["risk_score_std"] = (analysis_df["risk_score"] - risk_mean) / risk_sd

        # Set covariates: if sex is constant (e.g. WHI is all-female), drop sex
        covs_M0 = ["age"] + clinical_markers
        if analysis_df["sex"].nunique() > 1:
            covs_M0 = ["age", "sex"] + clinical_markers

        covs_M1 = covs_M0 + ["risk_score_std"]

        # Fit Reduced Model (M0)
        cph0 = CoxPHFitter()
        cph0.fit(analysis_df[covs_M0 + ["time", "event"]], duration_col="time", event_col="event")

        # Fit Full Model (M1)
        cph1 = CoxPHFitter()
        cph1.fit(analysis_df[covs_M1 + ["time", "event"]], duration_col="time", event_col="event")

        # Likelihood-Ratio Test calculation
        loglik_M0 = cph0.log_likelihood_
        loglik_M1 = cph1.log_likelihood_
        LR_stat = 2.0 * (loglik_M1 - loglik_M0)
        p_lrt = 1.0 - chi2.cdf(LR_stat, df=1)

        # Extract standardized HR from M1
        summary = cph1.summary
        hr = summary.loc["risk_score_std", "exp(coef)"]
        hr_lower = summary.loc["risk_score_std", "exp(coef) lower 95%"]
        hr_upper = summary.loc["risk_score_std", "exp(coef) upper 95%"]
        p_hr = summary.loc["risk_score_std", "p"]

        # Concordance index calculations
        pred_M0 = cph0.predict_partial_hazard(analysis_df[covs_M0])
        pred_M1 = cph1.predict_partial_hazard(analysis_df[covs_M1])

        event_arr = analysis_df["event"].values.astype(bool)
        time_arr = analysis_df["time"].values

        c0 = concordance_index_censored(event_arr, time_arr, pred_M0)[0]
        c1 = concordance_index_censored(event_arr, time_arr, pred_M1)[0]
        delta_c = c1 - c0

        # Paired bootstrap for Delta C-index (1000 iterations)
        logger.info("Computing bootstrap confidence intervals for Delta C-index...")
        n_boot = 1000
        rng = np.random.RandomState(SEED)
        boot_indices = rng.choice(len(analysis_df), size=(n_boot, len(analysis_df)), replace=True)

        delta_boots = []
        for b in range(n_boot):
            idx = boot_indices[b]
            boot_df = analysis_df.iloc[idx].reset_index(drop=True)
            try:
                cph0_b = CoxPHFitter()
                cph0_b.fit(boot_df[covs_M0 + ["time", "event"]], duration_col="time", event_col="event")
                cph1_b = CoxPHFitter()
                cph1_b.fit(boot_df[covs_M1 + ["time", "event"]], duration_col="time", event_col="event")

                pred_M0_b = cph0_b.predict_partial_hazard(boot_df[covs_M0])
                pred_M1_b = cph1_b.predict_partial_hazard(boot_df[covs_M1])

                ev_b = boot_df["event"].values.astype(bool)
                ti_b = boot_df["time"].values

                ci0 = concordance_index_censored(ev_b, ti_b, pred_M0_b)[0]
                ci1 = concordance_index_censored(ev_b, ti_b, pred_M1_b)[0]
                delta_boots.append(ci1 - ci0)
            except Exception:
                pass

        delta_boots = np.array(delta_boots)
        ci_lower = np.percentile(delta_boots, 2.5)
        ci_upper = np.percentile(delta_boots, 97.5)

        results_summary[cohort_name] = {
            "p_lrt": p_lrt,
            "delta_c": delta_c,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "hr": hr,
            "hr_lower": hr_lower,
            "hr_upper": hr_upper,
            "p_hr": p_hr
        }

    # Save nested Cox results to outputs folder
    summary_data = []
    for c_name in ["Test", "WHI"]:
        res = results_summary[c_name]
        summary_data.append({
            "Cohort": c_name,
            "LRT p-value": f"{res['p_lrt']:.4e}" if res["p_lrt"] < 0.0001 else f"{res['p_lrt']:.4f}",
            "Delta C-index": f"{res['delta_c']:.4f}",
            "Delta C-index 95% CI Lower": f"{res['ci_lower']:.4f}",
            "Delta C-index 95% CI Upper": f"{res['ci_upper']:.4f}",
            "Standardized HR": f"{res['hr']:.4f}",
            "HR 95% CI Lower": f"{res['hr_lower']:.4f}",
            "HR 95% CI Upper": f"{res['hr_upper']:.4f}",
            "HR p-value": f"{res['p_hr']:.4e}" if res["p_hr"] < 0.0001 else f"{res['p_hr']:.4f}"
        })
    df_output = pd.DataFrame(summary_data)
    df_output.to_csv(OUTPUT_DIR / "nested_cox_comparison_results.csv", index=False)
    
    logger.info("Nested Cox regression comparison completed. Output written to outputs/nested_cox_comparison_results.csv")

if __name__ == "__main__":
    main()
