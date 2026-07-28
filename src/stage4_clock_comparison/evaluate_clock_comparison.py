#!/usr/bin/env python
"""
Stage 4 Clock Comparison: Epigenetic Clocks and ML Models Comparative Analysis

This script compares the chronological age prediction accuracy and mortality risk
prediction power of our trained Coxnet and SVM models against standard epigenetic clocks
(Horvath, Hannum, PhenoAge, and GrimAge). It computes Pearson correlation coefficients with
chronological age, Harrell's C-indices, and hazard ratios from unadjusted and adjusted Cox
proportional hazards regression.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored
from sklearn.linear_model import LinearRegression

# Configure Plot Styling (Publication Quality)
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 18
})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

PREDICTORS = ["coxnet_age", "svm_age", "horvath2013", "hannum", "dnamphenoage", "grimage"]
PREDICTOR_LABELS = {
    "coxnet_age": "Coxnet (Our Model)",
    "svm_age": "SVM (Our Model)",
    "horvath2013": "Horvath 2013 Clock",
    "hannum": "Hannum Clock",
    "dnamphenoage": "PhenoAge Clock",
    "grimage": "GrimAge Clock"
}
PREDICTOR_COLORS = {
    "coxnet_age": "#9C27B0",
    "svm_age": "#795548",
    "horvath2013": "#2196F3",
    "hannum": "#4CAF50",
    "dnamphenoage": "#FF9800",
    "grimage": "#E91E63"
}

def fit_cox_model(df, covariates, predictor_col):
    """
    Fits a Cox PH model and extracts the exponentiated coefficient (HR),
    95 percent confidence intervals, and the corresponding p-value.
    """
    cols = covariates + ["time", "event"]
    model_df = df[cols].dropna()
    
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="time", event_col="event")
    
    summary = cph.summary
    hr = summary.loc[predictor_col, "exp(coef)"]
    hr_lower = summary.loc[predictor_col, "exp(coef) lower 95%"]
    hr_upper = summary.loc[predictor_col, "exp(coef) upper 95%"]
    p_val = summary.loc[predictor_col, "p"]
    
    return hr, hr_lower, hr_upper, p_val

def plot_combined_forest(results_df, title, save_path):
    """
    Generates a grouped forest plot comparing all predictors across splits.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid(False)
    
    splits = results_df["Split"].unique()
    n_predictors = len(PREDICTORS)
    
    y_positions = []
    y_labels = []
    
    current_y = 0.0
    for split in splits:
        split_data = results_df[results_df["Split"] == split]
        y_labels.append((current_y + (n_predictors - 1) * 0.15 / 2.0, split))
        
        for i, p in enumerate(PREDICTORS):
            row = split_data[split_data["Predictor"] == p]
            if row.empty:
                continue
            hr = row["HR"].values[0]
            lower = row["HR_lower"].values[0]
            upper = row["HR_upper"].values[0]
            p_val = row["p_value"].values[0]
            
            y_pos = current_y + i * 0.15
            y_positions.append(y_pos)
            
            xerr = np.vstack([hr - lower, upper - hr])
            
            ax.errorbar(
                x=hr,
                y=y_pos,
                xerr=xerr,
                fmt="s",
                color=PREDICTOR_COLORS[p],
                ecolor="#37474F",
                elinewidth=1.8,
                capsize=5,
                markersize=8,
                label=PREDICTOR_LABELS[p] if split == splits[0] else ""
            )
            
            p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
            text_label = f"HR = {hr:.3f} (95% CI: {lower:.3f} to {upper:.3f}, {p_str})"
            
            ax.text(
                ax.get_xlim()[1] * 1.02 if ax.get_xlim()[1] > 0 else 1.3,
                y_pos,
                text_label,
                va="center",
                ha="left",
                fontsize=10,
                fontweight="semibold",
                color="#37474F"
            )
            
        current_y += 1.2
        if split != splits[-1]:
            ax.axhline(y=current_y - 0.4, color="#CFD8DC", linestyle="--", linewidth=0.8)
            
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.5)
    
    tick_positions, tick_labels = zip(*y_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    
    ax.set_xlabel("Hazard Ratio (per 1-Year Increase in Biological Age)", fontsize=13, fontweight="semibold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=25)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(0.0, 1.05),
              ncol=3, fontsize=10, frameon=True, facecolor="white", edgecolor="none")
    
    sns.despine(left=True, bottom=True)
    xlim = ax.get_xlim()
    ax.set_xlim(min(xlim[0], 0.9), xlim[1] * 1.15)
    
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Grouped forest plot saved -> {save_path}")
    except Exception as e:
        logger.error(f"Failed to save forest plot: {e}")
    plt.close()

def main():
    logger.info("Initializing clock comparison evaluation...")
    
    REPO_ROOT = Path(__file__).resolve().parents[2]
    PRED_DIR = REPO_ROOT / "predictions"
    PLOT_DIR = REPO_ROOT / "plots" / "combined"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load biological clock age estimates
    fhs_clocks_path = PRED_DIR / "fhs_biological_clocks.csv"
    whi_clocks_path = PRED_DIR / "whi_biological_clocks.csv"
    
    if not fhs_clocks_path.exists() or not whi_clocks_path.exists():
        logger.error("Biological clocks files not found. Please run estimate_biological_clocks.py first.")
        return
        
    fhs_clocks = pd.read_csv(fhs_clocks_path)
    whi_clocks = pd.read_csv(whi_clocks_path)
    
    fhs_clocks["patient_id"] = fhs_clocks["patient_id"].astype(int)
    whi_clocks["patient_id"] = whi_clocks["patient_id"].astype(int)
    
    splits = ["train", "val", "test", "whi"]
    split_dfs = {}
    
    for s in splits:
        coxnet_file = PRED_DIR / f"coxnet_age_predictions_{s}_V3_OPTUNA_SS.csv"
        svm_file = PRED_DIR / f"svm_{s}_V3_OPTUNA_SS.csv"
        
        if not coxnet_file.exists() or not svm_file.exists():
            logger.error(f"Required ML prediction files for {s} split not found. Exiting.")
            return
            
        coxnet_df = pd.read_csv(coxnet_file)
        svm_df = pd.read_csv(svm_file)
        
        coxnet_df["patient_id"] = coxnet_df["patient_id"].astype(int)
        svm_df["patient_id"] = svm_df["patient_id"].astype(int)
        
        # Merge Coxnet risk score and SVM risk score
        merged = pd.merge(
            coxnet_df[["patient_id", "age", "sex", "event", "time", "survival_age"]],
            svm_df[["patient_id", "risk_score"]],
            on="patient_id"
        )
        merged = merged.rename(columns={"survival_age": "coxnet_age", "risk_score": "svm_risk"})
        
        # Merge with biological clocks
        clocks_df = whi_clocks if s == "whi" else fhs_clocks
        merged = pd.merge(
            merged,
            clocks_df[["patient_id", "horvath2013", "hannum", "dnamphenoage", "grimage"]],
            on="patient_id"
        )
        split_dfs[s] = merged
        logger.info(f"Split {s.upper()} merged size: {merged.shape}")
        
    # Calibrate SVM risk score on FHS Train set to units of years
    logger.info("Calibrating SVM risk scores to chronological age in years on FHS Train split...")
    train_df = split_dfs["train"]
    X_train = train_df[["svm_risk"]].values
    y_train = train_df["age"].values
    
    svm_calibrator = LinearRegression()
    svm_calibrator.fit(X_train, y_train)
    
    beta = svm_calibrator.coef_[0]
    alpha = svm_calibrator.intercept_
    logger.info(f"SVM Calibration Formula: survival_age = {alpha:.4f} + ({beta:.4f}) * svm_risk")
    
    for s in splits:
        X_split = split_dfs[s][["svm_risk"]].values
        split_dfs[s]["svm_age"] = svm_calibrator.predict(X_split)
        
    # 1. Evaluate Chronological Age Correlation
    logger.info("Calculating Pearson correlation with chronological age...")
    correlations = []
    clock_names = ["horvath2013", "hannum", "dnamphenoage", "grimage"]
    for s in splits:
        df = split_dfs[s]
        row = {"Split": s.upper()}
        for c in clock_names:
            r = df[c].corr(df["age"])
            row[c] = r
        correlations.append(row)
    df_correlations = pd.DataFrame(correlations)
    df_correlations.to_csv(PRED_DIR / "clocks_comparison_age_correlations.csv", index=False)
    
    # 2. Evaluate Concordance Index (C-Index)
    logger.info("Calculating Concordance Index (C-Index) against mortality...")
    cindices = []
    for s in splits:
        df = split_dfs[s]
        row = {"Split": s.upper()}
        y_event = df["event"].values.astype(bool)
        y_time = df["time"].values
        for p in PREDICTORS:
            ci = concordance_index_censored(y_event, y_time, df[p])[0]
            row[p] = ci
        cindices.append(row)
    df_cindex = pd.DataFrame(cindices)
    df_cindex_out = df_cindex.rename(columns={"coxnet_age": "coxnet", "svm_age": "svm"})
    df_cindex_out.to_csv(PRED_DIR / "clocks_comparison_cindex.csv", index=False)
    
    # 3. Fit Cox Proportional Hazards Models
    logger.info("Running Cox regression analysis...")
    results_unadjusted = []
    results_adjusted = []
    
    for s in splits:
        df = split_dfs[s]
        
        for p in PREDICTORS:
            # Unadjusted Cox PH
            covariates_unadj = [p]
            hr, hr_lower, hr_upper, p_val = fit_cox_model(df, covariates_unadj, p)
            results_unadjusted.append({
                "Split": s.upper(),
                "Predictor": p,
                "HR": hr,
                "HR_lower": hr_lower,
                "HR_upper": hr_upper,
                "p_value": p_val
            })
            
            # Adjusted Cox PH (for age, sex where applicable)
            covariates_adj = [p, "age"]
            if df["sex"].nunique() > 1:
                covariates_adj.append("sex")
            
            hr, hr_lower, hr_upper, p_val = fit_cox_model(df, covariates_adj, p)
            results_adjusted.append({
                "Split": s.upper(),
                "Predictor": p,
                "HR": hr,
                "HR_lower": hr_lower,
                "HR_upper": hr_upper,
                "p_value": p_val
            })
            
    df_unadjusted = pd.DataFrame(results_unadjusted)
    df_adjusted = pd.DataFrame(results_adjusted)
    
    df_unadjusted.to_csv(PRED_DIR / "clocks_comparison_unadjusted_cox.csv", index=False)
    df_adjusted.to_csv(PRED_DIR / "clocks_comparison_adjusted_cox.csv", index=False)
    
    # Generate Forest Plots
    plot_combined_forest(
        df_unadjusted, 
        "Mortality Hazard Ratios per 1-Year Increase in Biological Age (Unadjusted)",
        PLOT_DIR / "Figure_S2_Unadjusted_Forest.png"
    )
    plot_combined_forest(
        df_adjusted, 
        "Mortality Hazard Ratios per 1-Year Increase in Biological Age (Age-Adjusted)",
        PLOT_DIR / "Figure_3A_Adjusted_Forest.png"
    )
    
    logger.info("Stage 4 evaluation completed successfully.")

if __name__ == "__main__":
    main()
