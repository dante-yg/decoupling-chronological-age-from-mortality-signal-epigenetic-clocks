#!/usr/bin/env python
"""
Stage 5 Statistics: Cohort Demographics and Baseline Characteristics

This script aggregates predictions and cohort data to construct Table 1, summarizing
sample sizes, sex distributions, event frequencies, chronological age metrics, and 
follow-up times for FHS splits and the WHI cohort.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

def main():
    logger.info("Generating cohort summary table demographics...")
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    PRED_DIR = REPO_ROOT / "predictions"
    OUTPUT_DIR = REPO_ROOT / "outputs"
    FIGURE_DIR = REPO_ROOT / "figures"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    train_file = PRED_DIR / "coxnet_age_predictions_train_V3_OPTUNA_SS.csv"
    val_file = PRED_DIR / "coxnet_age_predictions_val_V3_OPTUNA_SS.csv"
    test_file = PRED_DIR / "coxnet_age_predictions_test_V3_OPTUNA_SS.csv"
    whi_file = PRED_DIR / "coxnet_age_predictions_whi_V3_OPTUNA_SS.csv"

    if not all([f.exists() for f in [train_file, val_file, test_file, whi_file]]):
        logger.error("Required prediction files not found. Please run Stage 3 modeling first.")
        return

    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    whi = pd.read_csv(whi_file)

    fhs_total = pd.concat([train, val, test], ignore_index=True)
    all_combined = pd.concat([fhs_total, whi], ignore_index=True)

    cohorts = [
        ("FHS (Train Split)", train),
        ("FHS (Validation Split)", val),
        ("FHS (Test Split)", test),
        ("FHS Cohort (Total)", fhs_total),
        ("WHI Cohort (External)", whi),
        ("Combined Overall (FHS + WHI)", all_combined)
    ]

    data = []
    for name, df in cohorts:
        n = len(df)
        n_male = int((df["sex"] == 1).sum())
        pct_male = (n_male / n) * 100
        n_female = int((df["sex"] == 2).sum())
        pct_female = (n_female / n) * 100
        
        n_event = int((df["event"] == True).sum())
        pct_event = (n_event / n) * 100
        n_censored = int((df["event"] == False).sum())
        pct_censored = (n_censored / n) * 100
        
        age_mean, age_sd, age_med = df["age"].mean(), df["age"].std(), df["age"].median()
        fu_mean, fu_sd, fu_med = df["time"].mean(), df["time"].std(), df["time"].median()
        
        data.append({
            "Cohort": name,
            "n": n,
            "Male(n) (%)": f"{n_male} ({pct_male:.1f}%)",
            "Female (n) (%)": f"{n_female} ({pct_female:.1f}%)",
            "Event (n) (%)": f"{n_event} ({pct_event:.1f}%)",
            "Censored (n) (%)": f"{n_censored} ({pct_censored:.1f}%)",
            "Age (mean +- SD)": f"{age_mean:.2f} \u00b1 {age_sd:.2f}",
            "Age (median)": f"{age_med:.1f}",
            "Follow-up (mean +- SD)": f"{fu_mean:.2f} \u00b1 {fu_sd:.2f}",
            "Follow-up Median": f"{fu_med:.2f}"
        })

    res_df = pd.DataFrame(data)

    res_df.to_csv(OUTPUT_DIR / "cohort_summary_table.csv", index=False)
    res_df.to_csv(FIGURE_DIR / "Cohort_Summary_Table.csv", index=False)

    logger.info(f"Saved successfully to outputs/cohort_summary_table.csv and figures/Cohort_Summary_Table.csv")

if __name__ == "__main__":
    main()
