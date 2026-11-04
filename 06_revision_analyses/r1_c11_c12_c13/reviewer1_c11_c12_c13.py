#!/usr/bin/env python
# coding: utf-8
"""
Reviewer 1, Comments 11, 12, 13:
Clinical-marker ΔC, Age-residualized KM curves, and Epigenetic Age Acceleration
Manuscript Revision: AGG-D-26-04256

Deliverables produced in C:\\F\\base_3\\reviews\\c11_c12_c13\\:
1. c11_biomarkers_delta_cindex.csv
2. c11_time_dependent_auroc.csv
3. c11_benchmark_reference_m5_m6_m7.csv
4. c12_km_tertile_cox_hr.csv
5. c12_cardiovascular_residual_tertiles.csv
6. c13_epigenetic_age_models.csv
7. c13_age_acceleration_metrics.csv
8. Figures (600 dpi PDF & TIFF, plus PNG):
   - reviewer1_km_age_residualized_test.{pdf,tif,png}
   - reviewer1_km_age_residualized_whi.{pdf,tif,png}
   - reviewer1_cardiovascular_residual_km.{pdf,tif,png}
9. reviewer1_c11_c12_c13_summary.md

Reporting policy: Plain numbers only.
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import logging
import warnings
import shutil
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import pearsonr, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION & DIRECTORIES
# ==============================================================================
BASE_DIR = Path(r"C:\F\base_3")
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "predictions"
OUT_DIR  = BASE_DIR / "reviews" / "c11_c12_c13"
FIG_DIR  = BASE_DIR / "figures"
LOG_DIR  = BASE_DIR / "logs"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "reviewer1_c11_c12_c13.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Reviewer1_C11_C12_C13")

SEED = 42
N_BOOT = 1000
MIN_P_BOUND = 2.0 / N_BOOT  # 0.002 floor

# Styling for plots
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})

# Okabe-Ito colour-blind-friendly palette and line styles
RISK_PALETTE = {
    "Low Risk": "#0072B2",     # Okabe-Ito Blue
    "Medium Risk": "#E69F00",  # Okabe-Ito Orange
    "High Risk": "#D55E00"     # Okabe-Ito Vermillion
}
RISK_STYLES = {
    "Low Risk": "-",
    "Medium Risk": "--",
    "High Risk": ":"
}

# ==============================================================================
# DATA LOADING & SPLIT SETUP
# ==============================================================================
def load_and_prepare_data():
    logger.info("Loading cohort data and existing predictions...")
    
    # 1. Raw cohort datasets
    fhs_raw = pd.read_csv(DATA_DIR / "FHS_non_normalized.csv", low_memory=False).rename(columns={"Share_ID": "patient_id"})
    whi_raw = pd.read_csv(DATA_DIR / "WHI_non_normalized.csv", low_memory=False).rename(columns={"SUBJECT_ID": "patient_id"})
    
    # 2. Stratified splits matching reviewer1_age_benchmark.py
    fhs_raw["event_stratify"] = fhs_raw["event"].astype(int)
    tr_va_df, te_df = train_test_split(fhs_raw, test_size=0.20, random_state=SEED, stratify=fhs_raw["event_stratify"])
    tr_df, va_df = train_test_split(tr_va_df, test_size=0.25, random_state=SEED, stratify=tr_va_df["event_stratify"])
    
    # Exact row ordering matching reviewer1_age_benchmark.py
    tr_idx = fhs_raw[fhs_raw["patient_id"].isin(set(tr_df["patient_id"]))].index
    va_idx = fhs_raw[fhs_raw["patient_id"].isin(set(va_df["patient_id"]))].index
    te_idx = fhs_raw[fhs_raw["patient_id"].isin(set(te_df["patient_id"]))].index
    
    splits_raw = {
        "Train (OOF)": fhs_raw.loc[tr_idx].drop(columns=["event_stratify"]).copy(),
        "Val": fhs_raw.loc[va_idx].drop(columns=["event_stratify"]).copy(),
        "Test": fhs_raw.loc[te_idx].drop(columns=["event_stratify"]).copy(),
        "WHI": whi_raw.copy()
    }
    
    # Also define Train (apparent) mapping for model fitting
    # 3. Load Biological Clocks
    fhs_clocks = pd.read_csv(PRED_DIR / "fhs_biological_clocks.csv")
    whi_clocks = pd.read_csv(PRED_DIR / "whi_biological_clocks.csv")
    
    clock_cols = ["horvath2013", "hannum", "dnamphenoage", "grimage"]
    fhs_clocks_map = {c: dict(zip(fhs_clocks["patient_id"], fhs_clocks[c])) for c in clock_cols}
    whi_clocks_map = {c: dict(zip(whi_clocks["patient_id"], whi_clocks[c])) for c in clock_cols}
    
    for s_name, s_df in splits_raw.items():
        is_whi = (s_name == "WHI")
        cmap = whi_clocks_map if is_whi else fhs_clocks_map
        for c in clock_cols:
            s_df[c] = s_df["patient_id"].map(cmap[c])
            
    # 4. Disease data
    disease_df = pd.read_csv(DATA_DIR / "Disease_data_with_censoring.csv")
    share_to_dbgap = dict(zip(fhs_raw["patient_id"], fhs_raw["dbGaP_Subject_ID"]))
    
    cpg_cols = [c for c in fhs_raw.columns if c.startswith("cg")]
    clin_cols = ["Glucose", "HDL", "Total_Cholesterol", "Triglycerides"]
    
    logger.info(f"Loaded splits: Train={len(splits_raw['Train (OOF)'])}, Val={len(splits_raw['Val'])}, Test={len(splits_raw['Test'])}, WHI={len(splits_raw['WHI'])}")
    return splits_raw, cpg_cols, clin_cols, disease_df, share_to_dbgap

# ==============================================================================
# PART A: CLINICAL BIOMARKERS DISCRIMINATION (COMMENT 11)
# ==============================================================================
def run_part_a(splits_raw, cpg_cols, clin_cols):
    logger.info("==========================================================================")
    logger.info("PART A: METABOLIC BIOMARKERS INCREMENTAL DISCRIMINATION (COMMENT 11)")
    logger.info("==========================================================================")
    
    # Check if Part A deliverables already exist
    f_delta = OUT_DIR / "c11_biomarkers_delta_cindex.csv"
    f_td = OUT_DIR / "c11_time_dependent_auroc.csv"
    f_ref = OUT_DIR / "c11_benchmark_reference_m5_m6_m7.csv"
    if f_delta.exists() and f_td.exists() and f_ref.exists():
        logger.info("Found existing Part A deliverables. Loading cached results...")
        df_delta_learners = pd.read_csv(f_delta)
        df_td_auroc = pd.read_csv(f_td)
        df_ref = pd.read_csv(f_ref)
        
        # Check if WHI TD-AUROC needs NaN fix
        if df_td_auroc["ci_lower"].isna().any() or df_td_auroc["ci_95"].astype(str).str.contains("nan").any():
            logger.info("Updating WHI TD-AUROC bootstrap CIs with finite values...")
            whi_df = splits_raw["WHI"]
            surv_whi = np.array(list(zip(whi_df["event"].astype(bool), whi_df["time"])), dtype=[("event", "?"), ("time", "<f8")])
            fc_w = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv")
            fp_w = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS_CPG.csv")
            r_comb_w = whi_df["patient_id"].map(dict(zip(fc_w["patient_id"], fc_w["risk_score"]))).values
            r_cpg_w  = whi_df["patient_id"].map(dict(zip(fp_w["patient_id"], fp_w["risk_score"]))).values
            
            rng = np.random.RandomState(SEED)
            boot_w = rng.choice(len(whi_df), size=(N_BOOT, len(whi_df)), replace=True)
            
            for w_idx, (w_name, w_grid) in enumerate([("8.0 - 14.0 years (50 pts)", np.linspace(8.0, 14.0, 50)),
                                                      ("8.0 - 20.0 years (50 pts)", np.linspace(8.0, 20.0, 50))]):
                b_diffs_w = []
                for b in range(N_BOOT):
                    idx = boot_w[b]
                    try:
                        _, m_c_b = cumulative_dynamic_auc(surv_whi, surv_whi[idx], r_comb_w[idx], w_grid)
                        _, m_p_b = cumulative_dynamic_auc(surv_whi, surv_whi[idx], r_cpg_w[idx], w_grid)
                        if not np.isnan(m_c_b) and not np.isnan(m_p_b):
                            b_diffs_w.append(m_c_b - m_p_b)
                    except Exception:
                        continue
                b_diffs_w = np.array(b_diffs_w)
                row_idx = w_idx + 1
                if len(b_diffs_w) > 0:
                    ci_l_w = np.percentile(b_diffs_w, 2.5)
                    ci_u_w = np.percentile(b_diffs_w, 97.5)
                    p_w = 2.0 * min(np.mean(b_diffs_w >= 0), np.mean(b_diffs_w <= 0))
                    p_disp_w = f"< {MIN_P_BOUND:.3f}" if p_w <= MIN_P_BOUND else f"{p_w:.3f}"
                    df_td_auroc.loc[row_idx, "ci_lower"] = round(ci_l_w, 4)
                    df_td_auroc.loc[row_idx, "ci_upper"] = round(ci_u_w, 4)
                    df_td_auroc.loc[row_idx, "ci_95"] = f"{ci_l_w:.4f} - {ci_u_w:.4f}"
                    df_td_auroc.loc[row_idx, "p_value"] = p_disp_w
            df_td_auroc.to_csv(f_td, index=False)
            logger.info("Saved updated TD-AUROC table.")
        return df_delta_learners, df_td_auroc, df_ref
        
    # --------------------------------------------------------------------------
    # A.1 Paired ΔC across 5 learners and Stacking Ensemble
    # --------------------------------------------------------------------------
    logger.info("A.1: Computing paired DeltaC for 5 learners and Stacking Ensemble...")
    learners = ["coxnet", "mtlr", "weibull_aft", "rsf", "svm", "ensemble"]
    learner_labels = {
        "coxnet": "CoxNet",
        "mtlr": "MTLR",
        "weibull_aft": "Weibull-AFT",
        "rsf": "RSF",
        "svm": "Fast Kernel SVM",
        "ensemble": "Stacking Ensemble (CoxMLP)"
    }
    
    split_file_map = {
        "Train (OOF)": "train",
        "Val": "val",
        "Test": "test",
        "WHI": "whi"
    }
    
    # Pre-generate bootstrap resample indices per split for strict paired alignment (seed 42)
    rng = np.random.RandomState(SEED)
    boot_indices = {
        s_name: rng.choice(len(s_df), size=(N_BOOT, len(s_df)), replace=True)
        for s_name, s_df in splits_raw.items()
    }
    
    delta_rows = []
    
    for s_name in ["Train (OOF)", "Val", "Test", "WHI"]:
        file_suffix = split_file_map[s_name]
        s_df = splits_raw[s_name]
        ev_arr = s_df["event"].values.astype(bool)
        ti_arr = s_df["time"].values
        
        for m in learners:
            if m == "ensemble":
                fc = pd.read_csv(PRED_DIR / f"stacking_predictions_{file_suffix}_V3_OPTUNA_SS.csv")
                fp = pd.read_csv(PRED_DIR / f"stacking_predictions_{file_suffix}_V3_OPTUNA_SS_CPG.csv")
                col_name = "ensemble"
            else:
                fc = pd.read_csv(PRED_DIR / f"{m}_{file_suffix}_V3_OPTUNA_SS.csv")
                fp = pd.read_csv(PRED_DIR / f"{m}_{file_suffix}_V3_OPTUNA_SS_CPG.csv")
                col_name = "risk_score"
                
            # Align predictions by patient_id
            map_c = dict(zip(fc["patient_id"], fc[col_name]))
            map_p = dict(zip(fp["patient_id"], fp[col_name]))
            
            r_comb = s_df["patient_id"].map(map_c).values
            r_cpg  = s_df["patient_id"].map(map_p).values
            
            c_comb = concordance_index_censored(ev_arr, ti_arr, r_comb)[0]
            c_cpg  = concordance_index_censored(ev_arr, ti_arr, r_cpg)[0]
            delta_c = c_comb - c_cpg
            
            # Bootstrap paired ΔC
            b_diffs = []
            b_c_comb = []
            b_c_cpg  = []
            for b in range(N_BOOT):
                idx = boot_indices[s_name][b]
                cb_c = concordance_index_censored(ev_arr[idx], ti_arr[idx], r_comb[idx])[0]
                cb_p = concordance_index_censored(ev_arr[idx], ti_arr[idx], r_cpg[idx])[0]
                b_diffs.append(cb_c - cb_p)
                b_c_comb.append(cb_c)
                b_c_cpg.append(cb_p)
                
            b_diffs = np.array(b_diffs)
            d_ci_l = np.percentile(b_diffs, 2.5)
            d_ci_u = np.percentile(b_diffs, 97.5)
            p_raw = 2.0 * min(np.mean(b_diffs >= 0), np.mean(b_diffs <= 0))
            p_disp = f"< {MIN_P_BOUND:.3f}" if p_raw <= MIN_P_BOUND else f"{p_raw:.3f}"
            
            c_comb_l = np.percentile(b_c_comb, 2.5)
            c_comb_u = np.percentile(b_c_comb, 97.5)
            c_cpg_l  = np.percentile(b_c_cpg, 2.5)
            c_cpg_u  = np.percentile(b_c_cpg, 97.5)
            
            delta_rows.append({
                "split": s_name,
                "learner_key": m,
                "learner_name": learner_labels[m],
                "c_comb": round(c_comb, 4),
                "c_comb_ci": f"{c_comb_l:.4f} - {c_comb_u:.4f}",
                "c_cpg": round(c_cpg, 4),
                "c_cpg_ci": f"{c_cpg_l:.4f} - {c_cpg_u:.4f}",
                "delta_c": round(delta_c, 4),
                "ci_lower": round(d_ci_l, 4),
                "ci_upper": round(d_ci_u, 4),
                "ci_95": f"{d_ci_l:.4f} - {d_ci_u:.4f}",
                "p_value": p_disp
            })
            
    df_delta_learners = pd.DataFrame(delta_rows)
    df_delta_learners.to_csv(OUT_DIR / "c11_biomarkers_delta_cindex.csv", index=False)
    logger.info(f"Saved learner DeltaC table -> {OUT_DIR / 'c11_biomarkers_delta_cindex.csv'}")
    
    # --------------------------------------------------------------------------
    # A.2 Time-dependent AUROC (8-14 years for Test, supported windows for WHI)
    # --------------------------------------------------------------------------
    logger.info("A.2: Computing Time-Dependent AUROC for CoxNet...")
    fhs_tr = splits_raw["Train (OOF)"]
    surv_train = np.array(list(zip(fhs_tr["event"].astype(bool), fhs_tr["time"])), dtype=[("event", "?"), ("time", "<f8")])
    
    td_rows = []
    
    # (a) Test split on 8-14 year grid
    grid_test = np.linspace(8.0, 14.0, 50)
    te_df = splits_raw["Test"]
    surv_test = np.array(list(zip(te_df["event"].astype(bool), te_df["time"])), dtype=[("event", "?"), ("time", "<f8")])
    
    fc_te = pd.read_csv(PRED_DIR / "coxnet_test_V3_OPTUNA_SS.csv")
    fp_te = pd.read_csv(PRED_DIR / "coxnet_test_V3_OPTUNA_SS_CPG.csv")
    r_comb_te = te_df["patient_id"].map(dict(zip(fc_te["patient_id"], fc_te["risk_score"]))).values
    r_cpg_te  = te_df["patient_id"].map(dict(zip(fp_te["patient_id"], fp_te["risk_score"]))).values
    
    _, m_comb_te = cumulative_dynamic_auc(surv_train, surv_test, r_comb_te, grid_test)
    _, m_cpg_te  = cumulative_dynamic_auc(surv_train, surv_test, r_cpg_te, grid_test)
    d_te = m_comb_te - m_cpg_te
    
    # Bootstrap paired difference in mean AUROC
    b_diffs_te = []
    for b in range(N_BOOT):
        idx = boot_indices["Test"][b]
        try:
            _, m_c_b = cumulative_dynamic_auc(surv_train, surv_test[idx], r_comb_te[idx], grid_test)
            _, m_p_b = cumulative_dynamic_auc(surv_train, surv_test[idx], r_cpg_te[idx], grid_test)
            b_diffs_te.append(m_c_b - m_p_b)
        except Exception:
            continue
    b_diffs_te = np.array(b_diffs_te)
    ci_l_te = np.percentile(b_diffs_te, 2.5)
    ci_u_te = np.percentile(b_diffs_te, 97.5)
    p_te = 2.0 * min(np.mean(b_diffs_te >= 0), np.mean(b_diffs_te <= 0))
    p_disp_te = f"< {MIN_P_BOUND:.3f}" if p_te <= MIN_P_BOUND else f"{p_te:.3f}"
    
    td_rows.append({
        "split": "Test",
        "time_window": "8.0 - 14.0 years (50 evaluation times)",
        "reference_ipcw": "FHS Train (max follow-up 14.81 years)",
        "mean_auroc_comb": round(m_comb_te, 4),
        "mean_auroc_cpg": round(m_cpg_te, 4),
        "delta_mean_auroc": round(d_te, 4),
        "ci_lower": round(ci_l_te, 4),
        "ci_upper": round(ci_u_te, 4),
        "ci_95": f"{ci_l_te:.4f} - {ci_u_te:.4f}",
        "p_value": p_disp_te,
        "notes": "Exact manuscript evaluation grid"
    })
    
    # (b) WHI split
    # WHI follow-up extends past 23 years (events: 7.52 to 21.11 years).
    # FHS Train censoring survival is 0 beyond 14.81 years, so IPCW reference must be WHI itself.
    whi_df = splits_raw["WHI"]
    surv_whi = np.array(list(zip(whi_df["event"].astype(bool), whi_df["time"])), dtype=[("event", "?"), ("time", "<f8")])
    fc_w = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv")
    fp_w = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS_CPG.csv")
    r_comb_w = whi_df["patient_id"].map(dict(zip(fc_w["patient_id"], fc_w["risk_score"]))).values
    r_cpg_w  = whi_df["patient_id"].map(dict(zip(fp_w["patient_id"], fp_w["risk_score"]))).values
    
    for w_name, w_grid in [("8.0 - 14.0 years (50 pts)", np.linspace(8.0, 14.0, 50)),
                           ("8.0 - 20.0 years (50 pts)", np.linspace(8.0, 20.0, 50))]:
        _, m_comb_w = cumulative_dynamic_auc(surv_whi, surv_whi, r_comb_w, w_grid)
        _, m_cpg_w  = cumulative_dynamic_auc(surv_whi, surv_whi, r_cpg_w, w_grid)
        d_w = m_comb_w - m_cpg_w
        
        b_diffs_w = []
        for b in range(N_BOOT):
            idx = boot_indices["WHI"][b]
            try:
                _, m_c_b = cumulative_dynamic_auc(surv_whi, surv_whi[idx], r_comb_w[idx], w_grid)
                _, m_p_b = cumulative_dynamic_auc(surv_whi, surv_whi[idx], r_cpg_w[idx], w_grid)
                b_diffs_w.append(m_c_b - m_p_b)
            except Exception:
                continue
        b_diffs_w = np.array(b_diffs_w)
        ci_l_w = np.percentile(b_diffs_w, 2.5)
        ci_u_w = np.percentile(b_diffs_w, 97.5)
        p_w = 2.0 * min(np.mean(b_diffs_w >= 0), np.mean(b_diffs_w <= 0))
        p_disp_w = f"< {MIN_P_BOUND:.3f}" if p_w <= MIN_P_BOUND else f"{p_w:.3f}"
        
        td_rows.append({
            "split": "WHI",
            "time_window": w_name,
            "reference_ipcw": "WHI cohort (supported event window 7.52 - 21.11 years)",
            "mean_auroc_comb": round(m_comb_w, 4),
            "mean_auroc_cpg": round(m_cpg_w, 4),
            "delta_mean_auroc": round(d_w, 4),
            "ci_lower": round(ci_l_w, 4),
            "ci_upper": round(ci_u_w, 4),
            "ci_95": f"{ci_l_w:.4f} - {ci_u_w:.4f}",
            "p_value": p_disp_w,
            "notes": "WHI censoring reference used due to follow-up extending to 23.19 years"
        })
        
    df_td_auroc = pd.DataFrame(td_rows)
    df_td_auroc.to_csv(OUT_DIR / "c11_time_dependent_auroc.csv", index=False)
    logger.info(f"Saved TD-AUROC table -> {OUT_DIR / 'c11_time_dependent_auroc.csv'}")
    
    # --------------------------------------------------------------------------
    # A.3 & A.4 Reference Nested Models (M5 vs M2, M6 vs M4, and M7 alone)
    # --------------------------------------------------------------------------
    logger.info("A.3 & A.4: Fitting reference nested models M2, M4, M5, M6, M7...")
    # Standardize continuous covariates on Train (apparent)
    cont_cols = ["age"] + cpg_cols + clin_cols
    scaler_ref = StandardScaler()
    scaler_ref.fit(splits_raw["Train (OOF)"][cont_cols])
    
    scaled_splits = {}
    for s_name, s_df in splits_raw.items():
        sdf = s_df.copy()
        sdf[cont_cols] = scaler_ref.transform(s_df[cont_cols])
        scaled_splits[s_name] = sdf
        
    tr_s = scaled_splits["Train (OOF)"]
    cph_m2 = CoxPHFitter().fit(tr_s[["age", "sex", "time", "event"]], duration_col="time", event_col="event")
    cph_m4 = CoxPHFitter().fit(tr_s[["age", "sex"] + cpg_cols + ["time", "event"]], duration_col="time", event_col="event")
    cph_m5 = CoxPHFitter().fit(tr_s[["age", "sex"] + clin_cols + ["time", "event"]], duration_col="time", event_col="event")
    cph_m6 = CoxPHFitter().fit(tr_s[["age", "sex"] + cpg_cols + clin_cols + ["time", "event"]], duration_col="time", event_col="event")
    cph_m7 = CoxPHFitter().fit(tr_s[clin_cols + ["time", "event"]], duration_col="time", event_col="event")
    
    models_dict = {
        "M2": (cph_m2, ["age", "sex"]),
        "M4": (cph_m4, ["age", "sex"] + cpg_cols),
        "M5": (cph_m5, ["age", "sex"] + clin_cols),
        "M6": (cph_m6, ["age", "sex"] + cpg_cols + clin_cols),
        "M7": (cph_m7, clin_cols),
    }
    
    preds_nested = {s: {} for s in splits_raw.keys()}
    for s_name, sdf in scaled_splits.items():
        is_whi = (s_name == "WHI")
        for mid, (cph, covs) in models_dict.items():
            if is_whi and ("sex" in covs):
                eval_df = sdf.copy()
                eval_df["sex"] = 0.0
                preds_nested[s_name][mid] = cph.predict_partial_hazard(eval_df[covs]).values
            else:
                preds_nested[s_name][mid] = cph.predict_partial_hazard(sdf[covs]).values
                
    ref_rows = []
    for s_name, s_df in splits_raw.items():
        ev = s_df["event"].values.astype(bool)
        ti = s_df["time"].values
        split_label = "Train (apparent)" if s_name == "Train (OOF)" else s_name
        
        # M7 alone C-index
        c7 = concordance_index_censored(ev, ti, preds_nested[s_name]["M7"])[0]
        b_c7 = [concordance_index_censored(ev[idx], ti[idx], preds_nested[s_name]["M7"][idx])[0]
                for idx in boot_indices[s_name]]
        ref_rows.append({
            "split": split_label,
            "comparison": "M7 alone (Biomarkers alone)",
            "model_full": "M7 (Total_Chol, HDL, Glucose, Triglycerides)",
            "model_reduced": "None",
            "c_index_estimate": round(c7, 4),
            "ci_lower": round(np.percentile(b_c7, 2.5), 4),
            "ci_upper": round(np.percentile(b_c7, 97.5), 4),
            "ci_95": f"{np.percentile(b_c7, 2.5):.4f} - {np.percentile(b_c7, 97.5):.4f}",
            "delta_c": "",
            "delta_ci_95": "",
            "p_value": "",
            "details": "Biomarkers alone C-index"
        })
        
        # M5 vs M2 (Biomarkers added to Age + Sex)
        c5 = concordance_index_censored(ev, ti, preds_nested[s_name]["M5"])[0]
        c2 = concordance_index_censored(ev, ti, preds_nested[s_name]["M2"])[0]
        d52 = c5 - c2
        b_d52 = [concordance_index_censored(ev[idx], ti[idx], preds_nested[s_name]["M5"][idx])[0] -
                 concordance_index_censored(ev[idx], ti[idx], preds_nested[s_name]["M2"][idx])[0]
                 for idx in boot_indices[s_name]]
        p_52 = 2.0 * min(np.mean(np.array(b_d52) >= 0), np.mean(np.array(b_d52) <= 0))
        p_disp_52 = f"< {MIN_P_BOUND:.3f}" if p_52 <= MIN_P_BOUND else f"{p_52:.3f}"
        ref_rows.append({
            "split": split_label,
            "comparison": "M5 vs M2 (Biomarkers added to Age + Sex)",
            "model_full": "M5 (Age + Sex + Clinical)",
            "model_reduced": "M2 (Age + Sex)",
            "c_index_estimate": round(c5, 4),
            "ci_lower": round(np.percentile(b_d52, 2.5), 4),
            "ci_upper": round(np.percentile(b_d52, 97.5), 4),
            "ci_95": "",
            "delta_c": round(d52, 4),
            "delta_ci_95": f"{np.percentile(b_d52, 2.5):.4f} - {np.percentile(b_d52, 97.5):.4f}",
            "p_value": p_disp_52,
            "details": "Clinical markers added to Base-A"
        })
        
        # M6 vs M4 (Biomarkers added to Age + Sex + 10 CpGs)
        c6 = concordance_index_censored(ev, ti, preds_nested[s_name]["M6"])[0]
        c4 = concordance_index_censored(ev, ti, preds_nested[s_name]["M4"])[0]
        d64 = c6 - c4
        b_d64 = [concordance_index_censored(ev[idx], ti[idx], preds_nested[s_name]["M6"][idx])[0] -
                 concordance_index_censored(ev[idx], ti[idx], preds_nested[s_name]["M4"][idx])[0]
                 for idx in boot_indices[s_name]]
        p_64 = 2.0 * min(np.mean(np.array(b_d64) >= 0), np.mean(np.array(b_d64) <= 0))
        p_disp_64 = f"< {MIN_P_BOUND:.3f}" if p_64 <= MIN_P_BOUND else f"{p_64:.3f}"
        ref_rows.append({
            "split": split_label,
            "comparison": "M6 vs M4 (Biomarkers added to Age + Sex + 10 CpGs)",
            "model_full": "M6 (Age + Sex + 10 CpGs + Clinical)",
            "model_reduced": "M4 (Age + Sex + 10 CpGs)",
            "c_index_estimate": round(c6, 4),
            "ci_lower": round(np.percentile(b_d64, 2.5), 4),
            "ci_upper": round(np.percentile(b_d64, 97.5), 4),
            "ci_95": "",
            "delta_c": round(d64, 4),
            "delta_ci_95": f"{np.percentile(b_d64, 2.5):.4f} - {np.percentile(b_d64, 97.5):.4f}",
            "p_value": p_disp_64,
            "details": "Clinical markers added to epigenetic base"
        })
        
    df_ref = pd.DataFrame(ref_rows)
    df_ref.to_csv(OUT_DIR / "c11_benchmark_reference_m5_m6_m7.csv", index=False)
    logger.info(f"Saved benchmark reference table -> {OUT_DIR / 'c11_benchmark_reference_m5_m6_m7.csv'}")
    
    return df_delta_learners, df_td_auroc, df_ref

# ==============================================================================
# PART B: KAPLAN-MEIER CURVES NOT DRIVEN BY AGE (COMMENT 12)
# ==============================================================================
def mantel_haenszel_stratified_logrank(df, time_col, event_col, group_col, strata_col, group_order):
    """Computes exact Mantel-Haenszel stratified log-rank test across K groups and S strata."""
    K = len(group_order)
    g_map = {g: i for i, g in enumerate(group_order)}
    
    O = np.zeros(K)
    E = np.zeros(K)
    V = np.zeros((K, K))
    
    for stratum, s_df in df.groupby(strata_col):
        ev_times = sorted(s_df[s_df[event_col] == 1][time_col].unique())
        t_arr = s_df[time_col].values
        e_arr = s_df[event_col].values
        g_arr = np.array([g_map[g] for g in s_df[group_col]])
        
        for t in ev_times:
            at_risk = t_arr >= t
            events  = (t_arr == t) & (e_arr == 1)
            Y = np.sum(at_risk)
            d = np.sum(events)
            if Y <= 1 or d == 0:
                continue
                
            Y_k = np.array([np.sum(at_risk & (g_arr == k)) for k in range(K)])
            d_k = np.array([np.sum(events  & (g_arr == k)) for k in range(K)])
            
            e_k = Y_k * (d / Y)
            O += d_k
            E += e_k
            
            hyper = (Y - d) / (Y - 1)
            for i in range(K):
                for j in range(K):
                    if i == j:
                        V[i, i] += hyper * d * (Y_k[i] / Y) * (1.0 - Y_k[i] / Y)
                    else:
                        V[i, j] -= hyper * d * (Y_k[i] / Y) * (Y_k[j] / Y)
                        
    diff = (O - E)[:-1]
    V_sub = V[:-1, :-1]
    stat = float(diff @ np.linalg.inv(V_sub) @ diff)
    pval = float(chi2.sf(stat, df=K-1))
    return stat, pval

def run_part_b(splits_raw, disease_df, share_to_dbgap):
    logger.info("==========================================================================")
    logger.info("PART B: KAPLAN-MEIER CURVES NOT DRIVEN BY AGE (COMMENT 12)")
    logger.info("==========================================================================")
    
    # 1. Merge CoxNet combined score and GrimAge into all splits
    pub_tr_c = pd.read_csv(PRED_DIR / "coxnet_train_V3_OPTUNA_SS.csv")
    pub_va_c = pd.read_csv(PRED_DIR / "coxnet_val_V3_OPTUNA_SS.csv")
    pub_te_c = pd.read_csv(PRED_DIR / "coxnet_test_V3_OPTUNA_SS.csv")
    pub_wh_c = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv")
    
    score_maps = {
        "Train (OOF)": dict(zip(pub_tr_c["patient_id"], pub_tr_c["risk_score"])),
        "Val": dict(zip(pub_va_c["patient_id"], pub_va_c["risk_score"])),
        "Test": dict(zip(pub_te_c["patient_id"], pub_te_c["risk_score"])),
        "WHI": dict(zip(pub_wh_c["patient_id"], pub_wh_c["risk_score"]))
    }
    
    splits_b = {}
    for s_name, s_df in splits_raw.items():
        sdf = s_df.copy()
        sdf["coxnet_score"] = sdf["patient_id"].map(score_maps[s_name])
        splits_b[s_name] = sdf
        
    tr = splits_b["Train (OOF)"]
    te = splits_b["Test"]
    wh = splits_b["WHI"]
    
    targets = [
        ("coxnet_score", "EpiClock risk score"),
        ("grimage", "GrimAge")
    ]
    
    group_order = ["Low Risk", "Medium Risk", "High Risk"]
    
    km_stats_rows = []
    
    # Storage for plot data on Test and WHI
    plot_data_test = {}
    plot_data_whi  = {}
    
    for sc_id, sc_name in targets:
        logger.info(f"Analyzing {sc_name}...")
        
        # ----------------------------------------------------------------------
        # B.1 Original Unadjusted Tertiles (Train-derived fixed cutpoints)
        # ----------------------------------------------------------------------
        q1_orig = np.percentile(tr[sc_id], 100.0 / 3.0)
        q2_orig = np.percentile(tr[sc_id], 200.0 / 3.0)
        
        for s_key, s_df, p_store in [("Test", te, plot_data_test), ("WHI", wh, plot_data_whi)]:
            sdf = s_df.copy()
            sdf["orig_tertile"] = pd.cut(
                sdf[sc_id], bins=[-np.inf, q1_orig, q2_orig, np.inf], labels=group_order
            )
            lr_orig = multivariate_logrank_test(sdf["time"], sdf["orig_tertile"], sdf["event"].astype(int))
            
            # High vs Low HR
            sub = sdf[sdf["orig_tertile"].isin(["Low Risk", "High Risk"])].copy()
            sub["is_high"] = (sub["orig_tertile"] == "High Risk").astype(float)
            
            # Unadjusted Cox
            cph_u = CoxPHFitter().fit(sub[["is_high", "time", "event"]], duration_col="time", event_col="event")
            hr_u = cph_u.hazard_ratios_["is_high"]
            ci_u_l = cph_u.summary.loc["is_high", "exp(coef) lower 95%"]
            ci_u_u = cph_u.summary.loc["is_high", "exp(coef) upper 95%"]
            p_u = cph_u.summary.loc["is_high", "p"]
            
            # Adjusted Cox
            covs = ["age"] if s_key == "WHI" else ["age", "sex"]
            cph_a = CoxPHFitter().fit(sub[["is_high"] + covs + ["time", "event"]], duration_col="time", event_col="event")
            hr_a = cph_a.hazard_ratios_["is_high"]
            ci_a_l = cph_a.summary.loc["is_high", "exp(coef) lower 95%"]
            ci_a_u = cph_a.summary.loc["is_high", "exp(coef) upper 95%"]
            p_a = cph_a.summary.loc["is_high", "p"]
            
            p_store[(sc_id, "orig")] = (sdf, "orig_tertile", lr_orig.test_statistic, lr_orig.p_value)
            
            km_stats_rows.append({
                "score_name": sc_name,
                "cohort_split": s_key,
                "analysis_type": "Original Tertiles (Unadjusted Fixed)",
                "logrank_stat": round(lr_orig.test_statistic, 4),
                "logrank_p": f"< 0.001" if lr_orig.p_value < 0.001 else f"{lr_orig.p_value:.4f}",
                "high_vs_low_hr_unadj": f"{hr_u:.2f} ({ci_u_l:.2f} - {ci_u_u:.2f})",
                "high_vs_low_p_unadj": f"< 0.001" if p_u < 0.001 else f"{p_u:.4f}",
                "high_vs_low_hr_adj": f"{hr_a:.2f} ({ci_a_l:.2f} - {ci_a_u:.2f})",
                "high_vs_low_p_adj": f"< 0.001" if p_a < 0.001 else f"{p_a:.4f}",
                "notes": f"Train cutpoints: Q1={q1_orig:.4f}, Q2={q2_orig:.4f}"
            })
            
        # ----------------------------------------------------------------------
        # B.2 Age-Residualized Tertiles
        # ----------------------------------------------------------------------
        # Regress score on age + sex in TRAIN
        reg_res = LinearRegression()
        reg_res.fit(tr[["age", "sex"]], tr[sc_id])
        
        # Residuals in Train, Test, and WHI
        tr_res = tr[sc_id] - reg_res.predict(tr[["age", "sex"]])
        q1_res = np.percentile(tr_res, 100.0 / 3.0)
        q2_res = np.percentile(tr_res, 200.0 / 3.0)
        
        for s_key, s_df, p_store in [("Test", te, plot_data_test), ("WHI", wh, plot_data_whi)]:
            sdf = s_df.copy()
            if s_key == "WHI":
                # In WHI sex is female: in FHS 2.0 = female
                eval_covs = pd.DataFrame({"age": sdf["age"], "sex": 2.0})
                sdf["residual"] = sdf[sc_id] - reg_res.predict(eval_covs)
            else:
                sdf["residual"] = sdf[sc_id] - reg_res.predict(sdf[["age", "sex"]])
                
            sdf["res_tertile"] = pd.cut(
                sdf["residual"], bins=[-np.inf, q1_res, q2_res, np.inf], labels=group_order
            )
            lr_res = multivariate_logrank_test(sdf["time"], sdf["res_tertile"], sdf["event"].astype(int))
            
            sub = sdf[sdf["res_tertile"].isin(["Low Risk", "High Risk"])].copy()
            sub["is_high"] = (sub["res_tertile"] == "High Risk").astype(float)
            
            # Unadjusted Cox
            cph_u = CoxPHFitter().fit(sub[["is_high", "time", "event"]], duration_col="time", event_col="event")
            hr_u = cph_u.hazard_ratios_["is_high"]
            ci_u_l = cph_u.summary.loc["is_high", "exp(coef) lower 95%"]
            ci_u_u = cph_u.summary.loc["is_high", "exp(coef) upper 95%"]
            p_u = cph_u.summary.loc["is_high", "p"]
            
            # Adjusted Cox
            covs = ["age"] if s_key == "WHI" else ["age", "sex"]
            cph_a = CoxPHFitter().fit(sub[["is_high"] + covs + ["time", "event"]], duration_col="time", event_col="event")
            hr_a = cph_a.hazard_ratios_["is_high"]
            ci_a_l = cph_a.summary.loc["is_high", "exp(coef) lower 95%"]
            ci_a_u = cph_a.summary.loc["is_high", "exp(coef) upper 95%"]
            p_a = cph_a.summary.loc["is_high", "p"]
            
            p_store[(sc_id, "res")] = (sdf, "res_tertile", lr_res.test_statistic, lr_res.p_value)
            
            km_stats_rows.append({
                "score_name": sc_name,
                "cohort_split": s_key,
                "analysis_type": "Age-Residualized Tertiles",
                "logrank_stat": round(lr_res.test_statistic, 4),
                "logrank_p": f"< 0.001" if lr_res.p_value < 0.001 else f"{lr_res.p_value:.4f}",
                "high_vs_low_hr_unadj": f"{hr_u:.2f} ({ci_u_l:.2f} - {ci_u_u:.2f})",
                "high_vs_low_p_unadj": f"< 0.001" if p_u < 0.001 else f"{p_u:.4f}",
                "high_vs_low_hr_adj": f"{hr_a:.2f} ({ci_a_l:.2f} - {ci_a_u:.2f})",
                "high_vs_low_p_adj": f"< 0.001" if p_a < 0.001 else f"{p_a:.4f}",
                "notes": f"Train residual cutpoints: Q1={q1_res:.4f}, Q2={q2_res:.4f}"
            })
            
        # ----------------------------------------------------------------------
        # B.3 Age-Stratified Analysis (Within-band tertiles & Stratified Log-Rank)
        # ----------------------------------------------------------------------
        # Test split age bands: <50 (4 deaths) merged with 50-64 -> <65 (23 deaths), 65-74 (42 deaths), >=75 (62 deaths)
        def get_band_test(age):
            if age < 65:
                return "<65"
            elif age < 75:
                return "65-74"
            else:
                return ">=75"
                
        te_strat = te.copy()
        te_strat["age_band"] = te_strat["age"].apply(get_band_test)
        
        te_strat_dfs = []
        for band, b_df in te_strat.groupby("age_band"):
            b_df = b_df.copy()
            q1_b = np.percentile(b_df[sc_id], 100.0 / 3.0)
            q2_b = np.percentile(b_df[sc_id], 200.0 / 3.0)
            b_df["strat_tertile"] = pd.cut(
                b_df[sc_id], bins=[-np.inf, q1_b, q2_b, np.inf], labels=group_order
            )
            te_strat_dfs.append(b_df)
        te_strat_df = pd.concat(te_strat_dfs, ignore_index=True)
        
        stat_te_s, p_te_s = mantel_haenszel_stratified_logrank(
            te_strat_df, "time", "event", "strat_tertile", "age_band", group_order
        )
        plot_data_test[(sc_id, "strat")] = (te_strat_df, "strat_tertile", stat_te_s, p_te_s)
        
        km_stats_rows.append({
            "score_name": sc_name,
            "cohort_split": "Test",
            "analysis_type": "Age-Stratified / Within-Band Tertiles",
            "logrank_stat": round(stat_te_s, 4),
            "logrank_p": f"< 0.001" if p_te_s < 0.001 else f"{p_te_s:.4f}",
            "high_vs_low_hr_unadj": "",
            "high_vs_low_p_unadj": "",
            "high_vs_low_hr_adj": "",
            "high_vs_low_p_adj": "",
            "notes": "Age bands (<65 [23 deaths], 65-74 [42 deaths], >=75 [62 deaths]); Mantel-Haenszel stratified log-rank test"
        })
        
        # WHI split age bands: <65 (20 deaths), >=65 (36 deaths)
        def get_band_whi(age):
            return "<65" if age < 65 else ">=65"
            
        wh_strat = wh.copy()
        wh_strat["age_band"] = wh_strat["age"].apply(get_band_whi)
        
        wh_strat_dfs = []
        for band, b_df in wh_strat.groupby("age_band"):
            b_df = b_df.copy()
            q1_b = np.percentile(b_df[sc_id], 100.0 / 3.0)
            q2_b = np.percentile(b_df[sc_id], 200.0 / 3.0)
            b_df["strat_tertile"] = pd.cut(
                b_df[sc_id], bins=[-np.inf, q1_b, q2_b, np.inf], labels=group_order
            )
            wh_strat_dfs.append(b_df)
        wh_strat_df = pd.concat(wh_strat_dfs, ignore_index=True)
        
        stat_wh_s, p_wh_s = mantel_haenszel_stratified_logrank(
            wh_strat_df, "time", "event", "strat_tertile", "age_band", group_order
        )
        plot_data_whi[(sc_id, "strat")] = (wh_strat_df, "strat_tertile", stat_wh_s, p_wh_s)
        
        km_stats_rows.append({
            "score_name": sc_name,
            "cohort_split": "WHI",
            "analysis_type": "Age-Stratified / Within-Band Tertiles",
            "logrank_stat": round(stat_wh_s, 4),
            "logrank_p": f"< 0.001" if p_wh_s < 0.001 else f"{p_wh_s:.4f}",
            "high_vs_low_hr_unadj": "",
            "high_vs_low_p_unadj": "",
            "high_vs_low_hr_adj": "",
            "high_vs_low_p_adj": "",
            "notes": "Age bands (<65 [20 deaths], >=65 [36 deaths]); Mantel-Haenszel stratified log-rank test"
        })
        
    df_km_stats = pd.DataFrame(km_stats_rows)
    df_km_stats.to_csv(OUT_DIR / "c12_km_tertile_cox_hr.csv", index=False)
    logger.info(f"Saved KM tertile Cox HR table -> {OUT_DIR / 'c12_km_tertile_cox_hr.csv'}")
    
    # --------------------------------------------------------------------------
    # B.4 Plotting 2 x 3 Panels for Test Split and WHI Split
    # --------------------------------------------------------------------------
    logger.info("Generating publication-ready 2 x 3 KM figures...")
    
    def generate_2x3_figure(plot_data, cohort_name, out_prefix):
        is_test = ("test" in out_prefix.lower())
        
        # 190 mm wide (7.48 in), ~165 mm tall (6.50 in)
        fig_w = 190.0 / 25.4
        fig_h = 165.0 / 25.4
        fig = plt.figure(figsize=(fig_w, fig_h))
        
        # Outer grid: 2 rows x 3 columns
        outer = gridspec.GridSpec(
            2, 3,
            top=0.94, bottom=0.06,
            left=0.10, right=0.98,
            hspace=0.52, wspace=0.38 if not is_test else 0.28
        )
        
        col_titles = [
            "A  Crude tertiles",
            "B  Age-residualized tertiles",
            "C  Within age bands (pooled)"
        ]
        
        plot_keys = ["orig", "res", "strat"]
        score_keys = ["coxnet_score", "grimage"]
        
        # Determine x-axis limits and at-risk table time points
        if is_test:
            x_max = 15.0
            x_ticks = [0, 5, 10, 15]
            times_ar = [0, 5, 10]  # Drop 15 because follow-up ends at 14.8y (all groups 0)
            lbl_x = -1.2
        else:
            x_max = 20.0
            x_ticks = [0, 5, 10, 15, 20]
            times_ar = [0, 5, 10, 15, 20]
            lbl_x = -1.2
            
        for r_idx, sc in enumerate(score_keys):
            for c_idx, pk in enumerate(plot_keys):
                inner = gridspec.GridSpecFromSubplotSpec(
                    2, 1,
                    subplot_spec=outer[r_idx, c_idx],
                    height_ratios=[3.6, 1.0],
                    hspace=0.35
                )
                ax = fig.add_subplot(inner[0])
                ax_risk = fig.add_subplot(inner[1])
                
                df_plot, tert_col, stat_val, p_val = plot_data[(sc, pk)]
                p_disp = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
                test_label = "Stratified log-rank" if pk == "strat" else "Log-rank"
                
                fitters = []
                for grp in group_order:
                    gdf = df_plot[df_plot[tert_col] == grp]
                    n_sub = len(gdf)
                    n_ev  = int(gdf["event"].sum())
                    
                    kmf = KaplanMeierFitter()
                    label_grp = grp.replace(" Risk", "")
                    legend_lbl = f"{label_grp} (n = {n_sub}, {n_ev} deaths)"
                    kmf.fit(gdf["time"], gdf["event"].astype(int), label=legend_lbl)
                    kmf.plot_survival_function(
                        ax=ax,
                        color=RISK_PALETTE[grp],
                        linestyle=RISK_STYLES[grp],
                        linewidth=1.2,
                        show_censors=True,
                        censor_styles={"marker": "|", "ms": 3.5, "mew": 0.8},
                        ci_show=False
                    )
                    fitters.append((gdf, label_grp, RISK_PALETTE[grp]))
                    
                ax.set_ylim(0.5, 1.0)
                ax.set_xlim(0.0, x_max)
                ax.set_xticks(x_ticks)
                ax.tick_params(axis="both", labelsize=7, pad=2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(False)
                
                # y-axis label on left column only
                if c_idx == 0:
                    ax.set_ylabel("Survival probability", fontsize=8, labelpad=3)
                else:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)
                    
                # x-axis label on each panel
                ax.set_xlabel("Time (years)", fontsize=8, labelpad=2)
                
                # Column titles appear once above top row only
                if r_idx == 0:
                    ax.set_title(col_titles[c_idx], fontsize=9, fontweight="bold", pad=6)
                    
                # Log-rank box in lower-left
                lr_text = f"{test_label}\np {p_disp}"
                if c_idx == 0:
                    # In Column A, curves separate widely; place log-rank box at bottom-left
                    ax.text(
                        0.04, 0.06, lr_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
                    )
                    # Legend between Medium and High curves on the right (clear of all curves)
                    ax.legend(
                        bbox_to_anchor=(0.98, 0.48), loc="center right",
                        fontsize=7, frameon=True, facecolor="white", framealpha=0.9,
                        edgecolor="none", handlelength=1.4, borderpad=0.2, labelspacing=0.25
                    )
                else:
                    # In Column B & C, stagger log-rank box at y=0.30 to avoid legend collision
                    ax.text(
                        0.04, 0.30, lr_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        va="bottom", ha="left",
                        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
                    )
                    # Legend in lower-right below curves
                    ax.legend(
                        loc="lower right",
                        fontsize=7, frameon=True, facecolor="white", framealpha=0.9,
                        edgecolor="none", handlelength=1.4, borderpad=0.2, labelspacing=0.25
                    )
                    
                # Number at risk table under panel
                ax_risk.set_xlim(0.0, x_max)
                ax_risk.set_ylim(-0.6, 2.9)
                ax_risk.axis("off")
                
                for i_g, (gdf_g, l_grp, c_grp) in enumerate(fitters):
                    y_r = 2 - i_g
                    ax_risk.text(lbl_x, y_r, l_grp, color=c_grp, fontsize=7, fontweight="bold", ha="right", va="center")
                    for t_val in times_ar:
                        n_ar = (gdf_g["time"] >= t_val).sum()
                        ax_risk.text(t_val, y_r, str(n_ar), fontsize=7, ha="center", va="center", color="black")
                        
        # Bold row labels on the left of each row
        fig.text(0.016, 0.77, "EpiClock risk score", rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
        fig.text(0.016, 0.28, "GrimAge", rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")
        
        # Save figure in PDF, TIFF (600 dpi, LZW, RGB), and PNG
        for ext, dpi in [("pdf", 600), ("tif", 600), ("png", 300)]:
            out_file = OUT_DIR / f"{out_prefix}.{ext}"
            fig_file = FIG_DIR / f"{out_prefix}.{ext}"
            
            if ext == "tif":
                # Save via PIL to guarantee 600 dpi RGB LZW
                temp_tif = FIG_DIR / f"temp_{out_prefix}.tif"
                fig.savefig(temp_tif, dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
                im = Image.open(temp_tif)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                im.save(fig_file, compression="tiff_lzw", dpi=(600, 600))
                im.save(out_file, compression="tiff_lzw", dpi=(600, 600))
                if temp_tif.exists():
                    temp_tif.unlink()
            else:
                fig.savefig(out_file, dpi=dpi)
                fig.savefig(fig_file, dpi=dpi)
                
            logger.info(f"Saved {cohort_name} figure -> {out_file} (dpi={dpi})")
            
        # Additional export for Figure 4 (FHS Test)
        if is_test:
            fig_4_pdf = FIG_DIR / "Figure_4_km_age_residualized_test.pdf"
            fig_4_tif = FIG_DIR / "Figure_4_km_age_residualized_test.tif"
            fig_4_png = FIG_DIR / "Figure_4_km_age_residualized_test.png"
            fig_4_png150 = FIG_DIR / "Figure_4_km_age_residualized_test_150dpi.png"
            
            fig.savefig(fig_4_pdf, dpi=600)
            fig.savefig(fig_4_png, dpi=300)
            fig.savefig(fig_4_png150, dpi=150)
            
            temp_tif4 = FIG_DIR / "temp_fig4.tif"
            fig.savefig(temp_tif4, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
            im4 = Image.open(temp_tif4)
            if im4.mode != "RGB":
                im4 = im4.convert("RGB")
            im4.save(fig_4_tif, compression="tiff_lzw", dpi=(600, 600))
            if temp_tif4.exists():
                temp_tif4.unlink()
                
            # Copy PDF to paper_work/Paper_submission/review_2/Figures/Figure_4.pdf
            dest_fig4 = BASE_DIR / "paper_work" / "Paper_submission" / "review_2" / "Figures" / "Figure_4.pdf"
            if dest_fig4.parent.exists():
                shutil.copy2(fig_4_pdf, dest_fig4)
                logger.info(f"Copied updated Figure 4 PDF -> {dest_fig4}")
                
        plt.close(fig)
        
    generate_2x3_figure(plot_data_test, "FHS Test Split (n=699, deaths=127)", "reviewer1_km_age_residualized_test")
    generate_2x3_figure(plot_data_whi, "WHI External Cohort (n=196, deaths=56)", "reviewer1_km_age_residualized_whi")
    
    # --------------------------------------------------------------------------
    # B.5 Cardiovascular Cumulative-Incidence by Residual Tertile
    # --------------------------------------------------------------------------
    logger.info("B.5: Evaluating cardiovascular outcomes (MI, CVA, CHF) by residual tertile...")
    
    # Link dbGaP_Subject_ID with disease data
    te_m = te.copy()
    te_m["dbGaP_Subject_ID"] = te_m["patient_id"].map(share_to_dbgap)
    disease_cols_to_use = [
        "dbGaP_Subject_ID", "MI", "MI_time", "AP", "AP_time", 
        "CVA", "CVA_time", "IC", "IC_time", "CHF", "CHF_time"
    ]
    te_dis = te_m.merge(disease_df[disease_cols_to_use], on="dbGaP_Subject_ID")
    
    # Regress CoxNet score on age + sex in Train
    reg_dis = LinearRegression().fit(tr[["age", "sex"]], tr["coxnet_score"])
    te_dis["residual"] = te_dis["coxnet_score"] - reg_dis.predict(te_dis[["age", "sex"]])
    
    tr_res_dis = tr["coxnet_score"] - reg_dis.predict(tr[["age", "sex"]])
    q1_dis = np.percentile(tr_res_dis, 100.0 / 3.0)
    q2_dis = np.percentile(tr_res_dis, 200.0 / 3.0)
    te_dis["res_tertile"] = pd.cut(
        te_dis["residual"], bins=[-np.inf, q1_dis, q2_dis, np.inf], labels=group_order
    )
    
    cv_targets = [
        ("MI", "MI_time", "Myocardial Infarction (MI)"),
        ("CVA", "CVA_time", "Stroke / Cerebrovascular Accident (CVA)"),
        ("CHF", "CHF_time", "Congestive Heart Failure (CHF)")
    ]
    
    cv_rows = []
    
    fig_cv_w = 190.0 / 25.4  # 7.48 in
    fig_cv_h = 95.0 / 25.4   # 3.74 in
    fig_cv = plt.figure(figsize=(fig_cv_w, fig_cv_h))
    
    outer_cv = gridspec.GridSpec(
        1, 3,
        top=0.88, bottom=0.10,
        left=0.10, right=0.98,
        wspace=0.28
    )
    
    for idx_cv, (d_col, t_col, d_label) in enumerate(cv_targets):
        inner_cv = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_cv[0, idx_cv],
            height_ratios=[3.6, 1.0],
            hspace=0.35
        )
        ax = fig_cv.add_subplot(inner_cv[0])
        ax_risk = fig_cv.add_subplot(inner_cv[1])
        
        # Exclude prevalent cases at baseline (time < 0)
        inc = te_dis[te_dis[t_col] >= 0].copy()
        n_inc = len(inc)
        n_ev_inc = int(inc[d_col].sum())
        
        lr_cv = multivariate_logrank_test(inc[t_col], inc["res_tertile"], inc[d_col].astype(int))
        p_disp_cv = f"< 0.001" if lr_cv.p_value < 0.001 else f"= {lr_cv.p_value:.3f}"
        
        # Cause-specific Cox model: High vs Low residual tertile adjusted for age + sex
        sub_cv = inc[inc["res_tertile"].isin(["Low Risk", "High Risk"])].copy()
        sub_cv["is_high"] = (sub_cv["res_tertile"] == "High Risk").astype(float)
        
        cph_cv = CoxPHFitter().fit(
            sub_cv[["is_high", "age", "sex", t_col, d_col]],
            duration_col=t_col,
            event_col=d_col
        )
        hr_cv = cph_cv.hazard_ratios_["is_high"]
        ci_l_cv = cph_cv.summary.loc["is_high", "exp(coef) lower 95%"]
        ci_u_cv = cph_cv.summary.loc["is_high", "exp(coef) upper 95%"]
        p_hr_cv = cph_cv.summary.loc["is_high", "p"]
        p_hr_disp = f"< 0.001" if p_hr_cv < 0.001 else f"{p_hr_cv:.4f}"
        
        cv_rows.append({
            "endpoint": d_label,
            "incident_sample_size": n_inc,
            "incident_events": n_ev_inc,
            "cause_specific_logrank_stat": round(lr_cv.test_statistic, 4),
            "cause_specific_logrank_p": p_disp_cv,
            "high_vs_low_hr_adj_age_sex": f"{hr_cv:.2f} ({ci_l_cv:.2f} - {ci_u_cv:.2f})",
            "high_vs_low_p_value": p_hr_disp,
            "notes": "Prevalent cases at baseline (time < 0) excluded per protocol"
        })
        
        # Plot Cumulative Incidence (1 - S(t))
        fitters_cv = []
        max_y = 0.05
        for grp in group_order:
            gdf = inc[inc["res_tertile"] == grp]
            kmf = KaplanMeierFitter()
            lbl = f"{grp.replace(' Risk', '')} (n = {len(gdf)}, {int(gdf[d_col].sum())} events)"
            kmf.fit(gdf[t_col], gdf[d_col].astype(int), label=lbl)
            kmf.plot_cumulative_density(
                ax=ax,
                color=RISK_PALETTE[grp],
                linestyle=RISK_STYLES[grp],
                linewidth=1.2,
                show_censors=True,
                censor_styles={"marker": "|", "ms": 3.5, "mew": 0.8},
                ci_show=False
            )
            fitters_cv.append((gdf, grp.replace(" Risk", ""), RISK_PALETTE[grp]))
            if len(kmf.cumulative_density_) > 0:
                max_y = max(max_y, kmf.cumulative_density_.values[-1][0])
                
        ax.set_ylim(0.0, min(1.0, max(0.25, max_y * 1.35)))
        ax.set_xlim(0.0, 15.0)
        ax.set_xticks([0, 5, 10, 15])
        ax.tick_params(axis="both", labelsize=7, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        
        if idx_cv == 0:
            ax.set_ylabel("Cumulative incidence", fontsize=8, labelpad=3)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
            
        ax.set_xlabel("Time (years)", fontsize=8, labelpad=2)
        ax.set_title(f"{d_label}\n(n = {n_inc}, {n_ev_inc} events)", fontsize=9, fontweight="bold", pad=6)
        
        info_text = f"Cause-specific log-rank p {p_disp_cv}\nHigh vs Low HR: {hr_cv:.2f} ({ci_l_cv:.2f}–{ci_u_cv:.2f})"
        ax.text(
            0.04, 0.96, info_text,
            transform=ax.transAxes,
            fontsize=6.8,
            va="top", ha="left",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
        )
        
        ax.legend(
            loc="center right",
            bbox_to_anchor=(0.98, 0.58),
            fontsize=6.8, frameon=True, facecolor="white", framealpha=0.9,
            edgecolor="none", handlelength=1.4, borderpad=0.2, labelspacing=0.25
        )
        
        # Number at risk table
        ax_risk.set_xlim(0.0, 15.0)
        ax_risk.set_ylim(-0.6, 2.9)
        ax_risk.axis("off")
        
        times_ar = [0, 5, 10]
        for i_g, (gdf_g, l_grp, c_grp) in enumerate(fitters_cv):
            y_r = 2 - i_g
            ax_risk.text(-1.2, y_r, l_grp, color=c_grp, fontsize=7, fontweight="bold", ha="right", va="center")
            for t_val in times_ar:
                n_ar = (gdf_g[t_col] >= t_val).sum()
                ax_risk.text(t_val, y_r, str(n_ar), fontsize=7, ha="center", va="center", color="black")
                
    for ext, dpi in [("pdf", 600), ("tif", 600), ("png", 300)]:
        out_cv_file = OUT_DIR / f"reviewer1_cardiovascular_residual_km.{ext}"
        fig_cv_file = FIG_DIR / f"reviewer1_cardiovascular_residual_km.{ext}"
        
        if ext == "tif":
            temp_cv_tif = FIG_DIR / "temp_cv.tif"
            fig_cv.savefig(temp_cv_tif, dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
            im_cv = Image.open(temp_cv_tif)
            if im_cv.mode != "RGB":
                im_cv = im_cv.convert("RGB")
            im_cv.save(fig_cv_file, compression="tiff_lzw", dpi=(600, 600))
            im_cv.save(out_cv_file, compression="tiff_lzw", dpi=(600, 600))
            if temp_cv_tif.exists():
                temp_cv_tif.unlink()
        else:
            fig_cv.savefig(out_cv_file, dpi=dpi)
            fig_cv.savefig(fig_cv_file, dpi=dpi)
            
        logger.info(f"Saved cardiovascular figure -> {out_cv_file} (dpi={dpi})")
    plt.close(fig_cv)
    
    df_cv = pd.DataFrame(cv_rows)
    df_cv.to_csv(OUT_DIR / "c12_cardiovascular_residual_tertiles.csv", index=False)
    logger.info(f"Saved cardiovascular residual tertile table -> {OUT_DIR / 'c12_cardiovascular_residual_tertiles.csv'}")
    
    return df_km_stats, df_cv

# ==============================================================================
# PART C: EPIGENETIC AGE ACCELERATION (COMMENT 13)
# ==============================================================================
def run_part_c(splits_raw, cpg_cols):
    logger.info("==========================================================================")
    logger.info("PART C: EPIGENETIC AGE ACCELERATION (COMMENT 13)")
    logger.info("==========================================================================")
    
    tr = splits_raw["Train (OOF)"].copy()
    va = splits_raw["Val"].copy()
    te = splits_raw["Test"].copy()
    wh = splits_raw["WHI"].copy()
    
    splits_c = {"Train (apparent)": tr, "Val": va, "Test": te, "WHI": wh}
    
    # --------------------------------------------------------------------------
    # C.1 Fit EpiClock 10-CpG Epigenetic Age Model on Train only
    # --------------------------------------------------------------------------
    logger.info("C.1: Fitting 10-CpG Epigenetic Age Predictor on Train split...")
    
    # OLS Model
    ols_age = LinearRegression()
    ols_age.fit(tr[cpg_cols], tr["age"])
    
    # ElasticNetCV Model
    enet_age = ElasticNetCV(cv=5, random_state=SEED)
    enet_age.fit(tr[cpg_cols], tr["age"])
    
    age_model_rows = []
    for model_name, m_obj in [("LinearRegression (OLS)", ols_age), ("ElasticNetCV", enet_age)]:
        for s_name, s_df in splits_c.items():
            pred_age = m_obj.predict(s_df[cpg_cols])
            r_age, p_age = pearsonr(pred_age, s_df["age"])
            mae = float(np.mean(np.abs(pred_age - s_df["age"])))
            rmse = float(np.sqrt(np.mean((pred_age - s_df["age"]) ** 2)))
            r2 = float(1.0 - np.sum((s_df["age"] - pred_age) ** 2) / np.sum((s_df["age"] - s_df["age"].mean()) ** 2))
            
            p_disp = f"< 0.001" if p_age < 0.001 else f"{p_age:.4f}"
            age_model_rows.append({
                "model": model_name,
                "split": s_name,
                "pearson_r": round(r_age, 4),
                "pearson_p": p_disp,
                "r_squared": round(r2, 4),
                "mae_years": round(mae, 2),
                "rmse_years": round(rmse, 2),
                "n_samples": len(s_df)
            })
            
    df_age_models = pd.DataFrame(age_model_rows)
    df_age_models.to_csv(OUT_DIR / "c13_epigenetic_age_models.csv", index=False)
    logger.info(f"Saved 10-CpG age model table -> {OUT_DIR / 'c13_epigenetic_age_models.csv'}")
    
    # We use OLS age predictor as primary EpiClock epigenetic age
    for s_name, s_df in splits_c.items():
        s_df["epiclock_age"] = ols_age.predict(s_df[cpg_cols])
        
    # --------------------------------------------------------------------------
    # C.2 Compute AgeAccel across 5 clocks (regressed on chronological age in TRAIN)
    # --------------------------------------------------------------------------
    logger.info("C.2: Regressing epigenetic clocks on age in Train to obtain AgeAccel...")
    clocks = ["epiclock_age", "horvath2013", "hannum", "dnamphenoage", "grimage"]
    clock_labels = {
        "epiclock_age": "EpiClock (10 CpGs)",
        "horvath2013": "Horvath",
        "hannum": "Hannum",
        "dnamphenoage": "PhenoAge",
        "grimage": "GrimAge"
    }
    
    accel_stats_rows = []
    
    # Pre-generate bootstrap resample indices per split (seed 42)
    rng = np.random.RandomState(SEED)
    boot_indices_c = {
        s_name: rng.choice(len(s_df), size=(N_BOOT, len(s_df)), replace=True)
        for s_name, s_df in splits_c.items()
    }
    
    for c_id in clocks:
        c_label = clock_labels[c_id]
        
        # Regress clock on chronological age in TRAIN
        reg_acc = LinearRegression().fit(tr[["age"]], tr[c_id])
        alpha_acc = float(reg_acc.intercept_)
        beta_acc  = float(reg_acc.coef_[0])
        
        # Compute residual AgeAccel
        for s_name, s_df in splits_c.items():
            s_df[f"{c_id}_accel"] = s_df[c_id] - reg_acc.predict(s_df[["age"]])
            
        train_sd = float(tr[f"{c_id}_accel"].std())
        
        # Standardize by TRAIN SD
        for s_name, s_df in splits_c.items():
            s_df[f"{c_id}_accel_std"] = s_df[f"{c_id}_accel"] / train_sd
            
        # Standardize continuous variables on Train for lifelines nested model
        ss_nest = StandardScaler().fit(tr[["age", f"{c_id}_accel_std"]])
        scaled_c_splits = {}
        for s_name, s_df in splits_c.items():
            sc_df = s_df.copy()
            sc_df[["age", f"{c_id}_accel_std"]] = ss_nest.transform(s_df[["age", f"{c_id}_accel_std"]])
            scaled_c_splits[s_name] = sc_df
            
        # Fit base model M2 (Age + Sex) on Train (apparent)
        tr_sc = scaled_c_splits["Train (apparent)"]
        cph_m2 = CoxPHFitter().fit(tr_sc[["age", "sex", "time", "event"]], duration_col="time", event_col="event")
        # Fit nested model (Age + Sex + AgeAccel) on Train (apparent)
        cph_nest = CoxPHFitter().fit(tr_sc[["age", "sex", f"{c_id}_accel_std", "time", "event"]], duration_col="time", event_col="event")
        
        for s_name, s_df in splits_c.items():
            is_whi = (s_name == "WHI")
            ev = s_df["event"].values.astype(bool)
            ti = s_df["time"].values
            
            # 1. Correlation of AgeAccel with chronological age
            r_age, p_age = pearsonr(s_df[f"{c_id}_accel"], s_df["age"])
            p_age_disp = f"< 0.001" if p_age < 0.001 else f"{p_age:.4f}"
            
            # 2. Cox HR per 1 SD (Unadjusted)
            cph_u = CoxPHFitter().fit(
                s_df[[f"{c_id}_accel_std", "time", "event"]],
                duration_col="time",
                event_col="event"
            )
            hr_u = cph_u.hazard_ratios_[f"{c_id}_accel_std"]
            ci_u_l = cph_u.summary.loc[f"{c_id}_accel_std", "exp(coef) lower 95%"]
            ci_u_u = cph_u.summary.loc[f"{c_id}_accel_std", "exp(coef) upper 95%"]
            p_u = cph_u.summary.loc[f"{c_id}_accel_std", "p"]
            p_u_disp = f"< 0.001" if p_u < 0.001 else f"{p_u:.4f}"
            
            # 3. Cox HR per 1 SD (Adjusted for Age + Sex)
            covs_adj = ["age"] if is_whi else ["age", "sex"]
            cph_a = CoxPHFitter().fit(
                s_df[[f"{c_id}_accel_std"] + covs_adj + ["time", "event"]],
                duration_col="time",
                event_col="event"
            )
            hr_a = cph_a.hazard_ratios_[f"{c_id}_accel_std"]
            ci_a_l = cph_a.summary.loc[f"{c_id}_accel_std", "exp(coef) lower 95%"]
            ci_a_u = cph_a.summary.loc[f"{c_id}_accel_std", "exp(coef) upper 95%"]
            p_a = cph_a.summary.loc[f"{c_id}_accel_std", "p"]
            p_a_disp = f"< 0.001" if p_a < 0.001 else f"{p_a:.4f}"
            
            # 4. Discrimination: Harrell's C of Age + Sex + AgeAccel and paired ΔC vs Age + Sex
            sc_df = scaled_c_splits[s_name]
            if is_whi:
                eval_df = sc_df.copy()
                eval_df["sex"] = 0.0
                pred_m2 = cph_m2.predict_partial_hazard(eval_df[["age", "sex"]]).values
                pred_nest = cph_nest.predict_partial_hazard(eval_df[["age", "sex", f"{c_id}_accel_std"]]).values
            else:
                pred_m2 = cph_m2.predict_partial_hazard(sc_df[["age", "sex"]]).values
                pred_nest = cph_nest.predict_partial_hazard(sc_df[["age", "sex", f"{c_id}_accel_std"]]).values
                
            c_nest = concordance_index_censored(ev, ti, pred_nest)[0]
            c_m2   = concordance_index_censored(ev, ti, pred_m2)[0]
            delta_c = c_nest - c_m2
            
            b_c_nest = []
            b_diffs  = []
            for b in range(N_BOOT):
                idx = boot_indices_c[s_name][b]
                cb_n = concordance_index_censored(ev[idx], ti[idx], pred_nest[idx])[0]
                cb_2 = concordance_index_censored(ev[idx], ti[idx], pred_m2[idx])[0]
                b_c_nest.append(cb_n)
                b_diffs.append(cb_n - cb_2)
                
            b_c_nest = np.array(b_c_nest)
            b_diffs  = np.array(b_diffs)
            c_ci_l = np.percentile(b_c_nest, 2.5)
            c_ci_u = np.percentile(b_c_nest, 97.5)
            d_ci_l = np.percentile(b_diffs, 2.5)
            d_ci_u = np.percentile(b_diffs, 97.5)
            p_dc = 2.0 * min(np.mean(b_diffs >= 0), np.mean(b_diffs <= 0))
            p_dc_disp = f"< {MIN_P_BOUND:.3f}" if p_dc <= MIN_P_BOUND else f"{p_dc:.3f}"
            
            accel_stats_rows.append({
                "clock": c_label,
                "split": s_name,
                "train_sd_years": round(train_sd, 4),
                "pearson_r_age": round(r_age, 4),
                "pearson_r_p": p_age_disp,
                "unadj_hr_per_1sd": f"{hr_u:.2f} ({ci_u_l:.2f} - {ci_u_u:.2f})",
                "unadj_p_value": p_u_disp,
                "adj_hr_per_1sd": f"{hr_a:.2f} ({ci_a_l:.2f} - {ci_a_u:.2f})",
                "adj_p_value": p_a_disp,
                "c_index_age_sex_accel": round(c_nest, 4),
                "c_index_ci_95": f"{c_ci_l:.4f} - {c_ci_u:.4f}",
                "delta_c_vs_m2": round(delta_c, 4),
                "delta_c_ci_95": f"{d_ci_l:.4f} - {d_ci_u:.4f}",
                "delta_c_p_value": p_dc_disp
            })
            
    df_accel_stats = pd.DataFrame(accel_stats_rows)
    df_accel_stats.to_csv(OUT_DIR / "c13_age_acceleration_metrics.csv", index=False)
    logger.info(f"Saved AgeAccel metrics table -> {OUT_DIR / 'c13_age_acceleration_metrics.csv'}")
    
    return df_age_models, df_accel_stats

# ==============================================================================
# SUMMARY MARKDOWN
# ==============================================================================
def write_summary_markdown(df_delta_learners, df_td_auroc, df_ref, df_km_stats, df_cv, df_age_models, df_accel_stats):
    logger.info("Writing reviewer1_c11_c12_c13_summary.md...")
    md_path = OUT_DIR / "reviewer1_c11_c12_c13_summary.md"
    
    lines = [
        "# Reviewer 1, Comments 11, 12, 13: Sensitivity Analysis Summary",
        "",
        "> **Manuscript Reference:** `AGG-D-26-04256`  ",
        "> **Pipeline Script:** [`reviewer1_c11_c12_c13.py`](file:///C:/F/base_3/reviewer1_c11_c12_c13.py)  ",
        "> **Execution Log:** [`logs/reviewer1_c11_c12_c13.log`](file:///C:/F/base_3/logs/reviewer1_c11_c12_c13.log)  ",
        "> **Reporting Policy:** Plain numbers only (estimates, 95% CIs, p-values, counts).  ",
        "",
        "---",
        "",
        "## 1. Part A (Comment 11): Do the Four Metabolic Biomarkers Add Discrimination?",
        "",
        "### 1.1 Paired ΔC Across Five Learners and Stacking Ensemble",
        "Comparison: $C(\\text{CpG + 4 Clinical Biomarkers}) - C(\\text{CpG-only})$ evaluated using saved prediction files with 1,000 paired bootstrap resamples (seed 42).",
        "",
        "| Split | Learner | Comb C-index (95% CI) | CpG C-index (95% CI) | Paired ΔC (95% CI) | p-value |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    
    for _, r in df_delta_learners.iterrows():
        lines.append(
            f"| {r['split']} | **{r['learner_name']}** | {r['c_comb']:.4f} ({r['c_comb_ci']}) | "
            f"{r['c_cpg']:.4f} ({r['c_cpg_ci']}) | {r['delta_c']:+.4f} ({r['ci_95']}) | {r['p_value']} |"
        )
        
    lines.extend([
        "",
        "### 1.2 Time-Dependent AUROC for CoxNet",
        "",
        "| Split | Evaluation Window | Reference IPCW Cohort | Mean AUROC (Comb) | Mean AUROC (CpG) | Paired ΔAUROC (95% CI) | p-value | Notes |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |"
    ])
    
    for _, r in df_td_auroc.iterrows():
        lines.append(
            f"| **{r['split']}** | {r['time_window']} | {r['reference_ipcw']} | {r['mean_auroc_comb']:.4f} | "
            f"{r['mean_auroc_cpg']:.4f} | {r['delta_mean_auroc']:+.4f} ({r['ci_95']}) | {r['p_value']} | {r['notes']} |"
        )
        
    lines.extend([
        "",
        "### 1.3 Benchmark Reference Nested Models and Biomarkers Alone (M7)",
        "",
        "| Split | Comparison | Full Model | Reduced Model | Estimate (95% CI) | p-value | Description |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :--- |"
    ])
    
    for _, r in df_ref.iterrows():
        is_m7 = pd.isna(r["delta_c"]) or str(r["delta_c"]).strip() == "" or "M7 alone" in str(r["comparison"])
        if is_m7:
            est_str = f"C = {r['c_index_estimate']:.4f} ({r['ci_95']})"
            p_str = "-"
            m_red = "-"
        else:
            est_str = f"ΔC = {float(r['delta_c']):+.4f} ({r['delta_ci_95']})"
            p_str = str(r['p_value'])
            m_red = str(r['model_reduced'])
        lines.append(
            f"| **{r['split']}** | {r['comparison']} | {r['model_full']} | {m_red} | "
            f"{est_str} | {p_str} | {r['details']} |"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "## 2. Part B (Comment 12): Kaplan–Meier Curves Not Driven by Age",
        "",
        "### 2.1 Tertile Survival Separation and Cox Hazard Ratios (High vs Low Tertile)",
        "",
        "| Score | Cohort Split | Analysis Type | Log-rank Stat | Log-rank p | High vs Low HR Unadj (95% CI) | p | High vs Low HR Adj Age+Sex (95% CI) | p | Notes |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ])
    
    for _, r in df_km_stats.iterrows():
        lines.append(
            f"| **{r['score_name']}** | {r['cohort_split']} | {r['analysis_type']} | {r['logrank_stat']} | "
            f"{r['logrank_p']} | {r['high_vs_low_hr_unadj']} | {r['high_vs_low_p_unadj']} | "
            f"{r['high_vs_low_hr_adj']} | {r['high_vs_low_p_adj']} | {r['notes']} |"
        )
        
    lines.extend([
        "",
        "### 2.2 Cardiovascular Endpoints (MI, CVA, CHF) by Age-Residualized Tertiles (FHS Test)",
        "Prevalent cases at baseline ($t < 0$) excluded. Evaluated using Train-derived fixed cutpoints applied to age-residualized score.",
        "",
        "| Endpoint | Incident N | Incident Deaths | Cause-Specific Log-rank Stat | p-value | High vs Low HR Adj Age+Sex (95% CI) | p-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])
    
    for _, r in df_cv.iterrows():
        lines.append(
            f"| **{r['endpoint']}** | {r['incident_sample_size']} | {r['incident_events']} | {r['cause_specific_logrank_stat']:.4f} | "
            f"{r['cause_specific_logrank_p']} | {r['high_vs_low_hr_adj_age_sex']} | {r['high_vs_low_p_value']} |"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "## 3. Part C (Comment 13): Epigenetic Age Acceleration",
        "",
        "### 3.1 Examination of Existing Code & 10-CpG Epigenetic Age Model",
        "Review of `code/coxnet_age_predictor.py` and `code/coxnet_age_acceleration_survival.ipynb`:",
        "- Neither file contains the original 2022 regression coefficients from Vijayakumar & Cho (2022).",
        "- Instead, existing scripts fit a post-hoc linear calibrator `chronological_age = alpha + beta * risk_score` on the survival linear predictor.",
        "- Per protocol, a linear/elastic-net age predictor was fitted strictly on the 10 CpGs on FHS Train only ($n=2,095$) and evaluated across all splits.",
        "",
        "| Model | Split | Pearson r | p-value | R² | MAE (Years) | RMSE (Years) | N |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])
    
    for _, r in df_age_models.iterrows():
        lines.append(
            f"| **{r['model']}** | {r['split']} | {r['pearson_r']:.4f} | {r['pearson_p']} | "
            f"{r['r_squared']:.4f} | {r['mae_years']:.2f} | {r['rmse_years']:.2f} | {r['n_samples']} |"
        )
        
    lines.extend([
        "",
        "### 3.2 Age Acceleration Association with Mortality Across Clocks",
        "AgeAccel = residual of epigenetic age regressed on chronological age in Train. Evaluated per 1 SD (Train SD).",
        "",
        "| Clock | Split | Train SD (Yrs) | r with Age (p) | Unadjusted HR per 1 SD (95% CI) | p | Adjusted HR per 1 SD (95% CI) | p | Age+Sex+Accel C-index (95% CI) | Paired ΔC vs M2 (95% CI) | p |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])
    
    for _, r in df_accel_stats.iterrows():
        lines.append(
            f"| **{r['clock']}** | {r['split']} | {r['train_sd_years']:.4f} | {r['pearson_r_age']:+.4f} ({r['pearson_r_p']}) | "
            f"{r['unadj_hr_per_1sd']} | {r['unadj_p_value']} | {r['adj_hr_per_1sd']} | {r['adj_p_value']} | "
            f"{r['c_index_age_sex_accel']:.4f} ({r['c_index_ci_95']}) | {r['delta_c_vs_m2']:+.4f} ({r['delta_c_ci_95']}) | {r['delta_c_p_value']} |"
        )
        
    lines.extend([
        "",
        "---",
        "",
        "## 4. Figures Generated",
        "",
        "The following figures were generated at 600 DPI (PDF, TIFF, PNG) with number-at-risk tables:",
        "1. **Main KM Figure (Test Split, 2 × 3 panel):** [`figures/reviewer1_km_age_residualized_test.pdf`](file:///C:/F/base_3/figures/reviewer1_km_age_residualized_test.pdf) (TIFF: [`reviewer1_km_age_residualized_test.tif`](file:///C:/F/base_3/figures/reviewer1_km_age_residualized_test.tif))",
        "   - Rows: EpiClock risk score, GrimAge",
        "   - Columns: Original Tertiles, Age-Residualized Tertiles, Age-Stratified / Pooled Within-Band Tertiles",
        "2. **Supplementary KM Figure (WHI Cohort, 2 × 3 panel):** [`figures/reviewer1_km_age_residualized_whi.pdf`](file:///C:/F/base_3/figures/reviewer1_km_age_residualized_whi.pdf) (TIFF: [`reviewer1_km_age_residualized_whi.tif`](file:///C:/F/base_3/figures/reviewer1_km_age_residualized_whi.tif))",
        "3. **Cardiovascular Endpoints Cumulative Incidence (FHS Test, 1 × 3 panel):** [`figures/reviewer1_cardiovascular_residual_km.pdf`](file:///C:/F/base_3/figures/reviewer1_cardiovascular_residual_km.pdf) (TIFF: [`reviewer1_cardiovascular_residual_km.tif`](file:///C:/F/base_3/figures/reviewer1_cardiovascular_residual_km.tif))",
        "",
        "---",
        "",
        "## 5. Items Not Done & Methodological Boundaries",
        "",
        "1. **WHI Cardiovascular Endpoints:** Cardiovascular endpoints (MI, CVA, CHF) in `data/Disease_data_with_censoring.csv` are available only for the 3,493 FHS participants; WHI has no linked cause-specific disease records in that file.",
        "2. **Prevalent Case Exclusion in WHI:** Prevalent cardiovascular case exclusion was performed for FHS where time-to-disease < 0 indicates pre-baseline occurrence.",
        "3. **Sex Adjustment in WHI:** WHI participants are 100% female; sex was omitted from WHI regression and survival models.",
        "4. **Re-tuning / Retraining:** No learners were retuned or retrained; all calculations used exact published prediction files and benchmark splits.",
        ""
    ])
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved executive markdown summary -> {md_path}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    logger.info("Starting Reviewer 1 Comments 11, 12, 13 Analysis Pipeline...")
    
    # 1. Load data
    splits_raw, cpg_cols, clin_cols, disease_df, share_to_dbgap = load_and_prepare_data()
    
    # 2. Part A: Metabolic biomarkers ΔC and TD-AUROC
    df_delta_learners, df_td_auroc, df_ref = run_part_a(splits_raw, cpg_cols, clin_cols)
    
    # 3. Part B: KM curves not driven by age & cardiovascular endpoints
    df_km_stats, df_cv = run_part_b(splits_raw, disease_df, share_to_dbgap)
    
    # 4. Part C: Epigenetic age acceleration
    df_age_models, df_accel_stats = run_part_c(splits_raw, cpg_cols)
    
    # 5. Summary Markdown
    write_summary_markdown(df_delta_learners, df_td_auroc, df_ref, df_km_stats, df_cv, df_age_models, df_accel_stats)
    
    logger.info("==========================================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"All deliverables written to: {OUT_DIR}")
    logger.info("==========================================================================")

if __name__ == "__main__":
    main()
