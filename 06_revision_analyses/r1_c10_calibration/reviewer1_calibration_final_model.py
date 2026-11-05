import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent
while _repo_root != _repo_root.parent:
    if (_repo_root / "config.py").exists():
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        break
    _repo_root = _repo_root.parent
import config

#!/usr/bin/env python
# coding: utf-8

"""
Reviewer 1, Comment 10: Calibration Analysis of the Final CoxNet Model
====================================================================
Manuscript Revision: AGG-D-26-04256
Model Evaluated: Manuscript Final CoxNet (10 EpiClock CpGs + 4 Clinical Biomarkers)
Cohorts: Framingham Heart Study (FHS Test, FHS Apparent/Train, FHS Val) & Women's Health Initiative (WHI)

Key Deliverables:
1. reviewer1_calibration_main.csv:
   - Primary 10-year metrics with 2,000-resample bootstrap 95% CIs:
     O_E, cal_slope, cal_intercept, ICI, E50, E90, brier, ibs, ipa, each with lower/upper CI.
   - Evaluated across FHS Test (primary), WHI (transported), WHI (recalibrated-in-the-large), FHS Train (apparent), and FHS Val.
2. Number at risk log at 5, 8, and 10 years.
3. Publication-quality Calibration Plot (300-DPI PNG and TIFF):
   - FHS Test and WHI as separate panels.
   - Deciles of predicted risk vs Kaplan-Meier observed risk with 95% CIs.
   - Austin et al. (2020) smoothed calibration curve: pseudo-observations regressed on cloglog(predicted risk) with restricted cubic spline (3 knots).
   - Identity line and marginal histogram of predicted risks underneath.
4. Cross-check reproduction with R riskRegression::Score() confirming agreement to within 0.005.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from lifelines import KaplanMeierFitter

# -----------------------------------------------------------------------------
# Configuration & Logging
# -----------------------------------------------------------------------------
SEED = 42
N_BOOT = 2000
HORIZON = 10.0
np.random.seed(SEED)

BASE_DIR = config.PROJECT_ROOT
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "predictions"
FIG_DIR  = BASE_DIR / "figures"
LOG_DIR  = BASE_DIR / "logs"

FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / "reviewer1_calibration_final_model.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CalibrationFinalModel")

# -----------------------------------------------------------------------------
# 1. Data Loading & Breslow Baseline Hazard Estimation
# -----------------------------------------------------------------------------
def load_data():
    """
    Loads existing CoxNet predictions for all splits.
    Does not refit or alter splits or feature scaling.
    """
    logger.info("Loading cohort splits and CoxNet prediction files...")
    splits = {
        "FHS Train (apparent)": pd.read_csv(PRED_DIR / "coxnet_train_V3_OPTUNA_SS.csv"),
        "FHS Val": pd.read_csv(PRED_DIR / "coxnet_val_V3_OPTUNA_SS.csv"),
        "FHS Test": pd.read_csv(PRED_DIR / "coxnet_test_V3_OPTUNA_SS.csv"),
        "WHI": pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv"),
    }
    for name, df in splits.items():
        # Ensure event is integer 0/1
        df['event'] = (df['event'] == True) | (df['event'] == 1) | (df['event'] == "True")
        df['event'] = df['event'].astype(int)
        logger.info(f"Loaded {name:22} | n = {len(df):4d} | events = {int(df['event'].sum()):3d}")
    return splits

def compute_breslow_cumulative_hazard(times, events, linear_predictors):
    """
    Computes Breslow baseline cumulative hazard H0(t) given times, events, and linear predictors.
    dH0(t_j) = d_j / sum_{k in R_j} exp(LP_k)
    """
    w = np.exp(linear_predictors)
    order = np.argsort(times)
    t_s, e_s, w_s = times[order], events[order], w[order]
    r_sum = np.cumsum(w_s[::-1])[::-1]
    ev_mask = (e_s == 1)
    t_ev = t_s[ev_mask]
    r_ev = r_sum[ev_mask]
    dh = np.zeros_like(r_ev, dtype=float)
    dh[r_ev > 0] = 1.0 / r_ev[r_ev > 0]
    H0_cum = np.cumsum(dh)
    return t_ev, H0_cum

def get_h0_at_t(t_ev, H0_cum, eval_t):
    """Evaluates cumulative hazard at target horizon eval_t."""
    idx = np.searchsorted(t_ev, eval_t, side='right') - 1
    return float(H0_cum[idx]) if idx >= 0 else 0.0

# -----------------------------------------------------------------------------
# 2. Number at Risk and Follow-up Audit
# -----------------------------------------------------------------------------
def log_number_at_risk(splits):
    """
    Audits and logs the number of participants at risk and cumulative deaths
    at 5, 8, and 10 years across all splits.
    """
    logger.info("==========================================================================")
    logger.info("LOG OF NUMBER AT RISK AT 5, 8, AND 10 YEARS")
    logger.info("==========================================================================")
    
    audit_records = []
    horizons = [5.0, 8.0, 10.0]
    
    for name, df in splits.items():
        t = df['time'].values
        e = df['event'].values
        n = len(df)
        
        row = {"split": name, "N_total": n, "total_deaths": int(e.sum())}
        logger.info(f"Cohort Split: {name} (N = {n}, Total Deaths = {int(e.sum())})")
        for h in horizons:
            at_risk = int((t >= h).sum())
            deaths = int(((t <= h) & (e == 1)).sum())
            pct_risk = at_risk / n * 100.0
            row[f"at_risk_{int(h)}y"] = at_risk
            row[f"pct_at_risk_{int(h)}y"] = round(pct_risk, 1)
            row[f"deaths_by_{int(h)}y"] = deaths
            logger.info(f"  At {int(h):2d} years: At Risk = {at_risk:4d} ({pct_risk:5.1f}%) | Deaths by {int(h):2d}y = {deaths:3d}")
        audit_records.append(row)
        
    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(BASE_DIR / "calibration_number_at_risk_audit.csv", index=False)
    logger.info("Saved calibration_number_at_risk_audit.csv.")
    return df_audit

# -----------------------------------------------------------------------------
# 3. Austin et al. (2020) Pseudo-Observations & Spline Calibration
# -----------------------------------------------------------------------------
def fast_km_survival_at_t(times, events, eval_t):
    """Fast Kaplan-Meier survival probability S(eval_t)."""
    ev_times = times[events == 1]
    ev_times = ev_times[ev_times <= eval_t]
    if len(ev_times) == 0:
        return 1.0
    unique_t, d_j = np.unique(ev_times, return_counts=True)
    sorted_times = np.sort(times)
    n_j = len(times) - np.searchsorted(sorted_times, unique_t, side='left')
    return float(np.prod(1.0 - d_j / np.maximum(n_j, 1)))

def compute_jackknife_pseudo_observations(times, events, eval_t):
    """
    Computes jackknife pseudo-observations for cumulative incidence / risk at horizon eval_t.
    theta_i = n * S(t) - (n - 1) * S^{-i}(t)
    Y_i = 1 - theta_i (risk scale)
    """
    n = len(times)
    s_all = fast_km_survival_at_t(times, events, eval_t)
    pseudos_surv = np.zeros(n)
    for i in range(n):
        idx = np.delete(np.arange(n), i)
        pseudos_surv[i] = n * s_all - (n - 1) * fast_km_survival_at_t(times[idx], events[idx], eval_t)
    return 1.0 - pseudos_surv

def harrell_rcs_3knots(x, knots=None):
    """
    Computes Harrell's restricted cubic spline (RCS) basis with 3 knots (at 10th, 50th, 90th percentiles).
    Basis terms: s1 = x, s2 = non-linear cubic spline term.
    """
    if knots is None:
        knots = np.percentile(x, [10, 50, 90])
    t1, t2, t3 = knots
    s1 = x
    denom = (t3 - t1)**2
    if denom == 0:
        denom = 1e-6
    term1 = np.maximum(0, x - t1)**3
    term2 = np.maximum(0, x - t2)**3
    term3 = np.maximum(0, x - t3)**3
    scale2 = (t3 - t1) / max(t3 - t2, 1e-6)
    scale3 = (t2 - t1) / max(t3 - t2, 1e-6)
    s2 = (term1 - scale2 * term2 + scale3 * term3) / denom
    return s1, s2, knots

def fit_austin_pseudo_spline(pred_risk, pseudo_risk, knots=None):
    """
    Regresses pseudo-observations on cloglog(pred_risk) with restricted cubic spline (3 knots).
    cloglog(p) = ln(-ln(1 - p))
    Y_pseudo = beta0 + beta1 * s1 + beta2 * s2 + epsilon
    """
    p_clipped = np.clip(pred_risk, 1e-5, 1.0 - 1e-5)
    cloglog_p = np.log(-np.log(1.0 - p_clipped))
    s1, s2, knots_used = harrell_rcs_3knots(cloglog_p, knots=knots)
    X = np.column_stack([np.ones_like(s1), s1, s2])
    beta = np.linalg.lstsq(X, pseudo_risk, rcond=None)[0]
    smoothed_risk = np.clip(X @ beta, 0.0, 1.0)
    return beta, knots_used, smoothed_risk

def predict_austin_pseudo_spline(eval_p, beta, knots):
    """Evaluates the fitted Austin spline curve over a dense grid of predicted risks."""
    p_clipped = np.clip(eval_p, 1e-5, 1.0 - 1e-5)
    cloglog_p = np.log(-np.log(1.0 - p_clipped))
    s1, s2, _ = harrell_rcs_3knots(cloglog_p, knots=knots)
    X = np.column_stack([np.ones_like(s1), s1, s2])
    return np.clip(X @ beta, 0.0, 1.0)

# -----------------------------------------------------------------------------
# 4. Calibration Metrics & Accuracy Functions
# -----------------------------------------------------------------------------
def fast_univariate_cox_slope(times, events, lp_vals):
    """
    Computes calibration slope: coefficient of the linear predictor in a univariate Cox model
    fitted in the evaluation set, using exact Newton-Raphson optimization.
    """
    order = np.argsort(times)
    t_s, e_s, x_s = times[order], events[order], lp_vals[order]
    ev_idx = np.where(e_s == 1)[0]
    if len(ev_idx) == 0:
        return 0.0
    beta = 1.0
    for _ in range(8):
        exp_bx = np.exp(beta * x_s)
        s0 = np.cumsum(exp_bx[::-1])[::-1]
        s1 = np.cumsum((x_s * exp_bx)[::-1])[::-1]
        s2 = np.cumsum(((x_s**2) * exp_bx)[::-1])[::-1]
        s0_ev, s1_ev, s2_ev = s0[ev_idx], s1[ev_idx], s2[ev_idx]
        grad = np.sum(x_s[ev_idx] - s1_ev / s0_ev)
        hess = -np.sum(s2_ev / s0_ev - (s1_ev / s0_ev)**2)
        step = grad / max(-hess, 1e-6)
        beta += step
        if abs(step) < 1e-6:
            break
    return float(beta)

def precompute_ipcw_structures(times, events, lp, t_ev_tr, H0_cum_tr, s_null_10, horizon=10.0, grid_1_13=None):
    """
    Precomputes individual IPCW loss vectors and loss matrix across grid_1_13
    for lightning-fast vectorization in bootstrapping.
    """
    n = len(times)
    km_c = KaplanMeierFitter().fit(times, 1 - events)
    G_ti = np.maximum(km_c.predict(times).values, 1e-6)
    G_h = max(float(km_c.predict(horizon)), 1e-6)
    
    # Baseline hazard at 10y
    H0_10 = get_h0_at_t(t_ev_tr, H0_cum_tr, horizon)
    surv_10 = np.exp(-H0_10 * np.exp(lp))
    
    # 1. 10-year model loss vector
    loss_mod_10 = np.zeros(n, dtype=float)
    died_10 = (times <= horizon) & (events == 1)
    alive_10 = (times > horizon)
    loss_mod_10[died_10] = (surv_10[died_10] ** 2) / G_ti[died_10]
    loss_mod_10[alive_10] = ((1.0 - surv_10[alive_10]) ** 2) / G_h
    
    # 2. 10-year null loss vector (training KM at 10y)
    loss_null_10 = np.zeros(n, dtype=float)
    loss_null_10[died_10] = (s_null_10 ** 2) / G_ti[died_10]
    loss_null_10[alive_10] = ((1.0 - s_null_10) ** 2) / G_h
    
    # 3. Grid loss matrix for IBS (1-13y)
    if grid_1_13 is None:
        grid_1_13 = np.linspace(1.0, 13.0, 50)
    G_grid = np.maximum(km_c.predict(grid_1_13).values, 1e-6)
    
    L_grid = np.zeros((n, len(grid_1_13)), dtype=float)
    for j, gt in enumerate(grid_1_13):
        H0_gt = get_h0_at_t(t_ev_tr, H0_cum_tr, gt)
        surv_gt = np.exp(-H0_gt * np.exp(lp))
        died_gt = (times <= gt) & (events == 1)
        alive_gt = (times > gt)
        L_grid[died_gt, j] = (surv_gt[died_gt] ** 2) / G_ti[died_gt]
        L_grid[alive_gt, j] = ((1.0 - surv_gt[alive_gt]) ** 2) / G_grid[j]
        
    return loss_mod_10, loss_null_10, L_grid, grid_1_13

def compute_all_metrics(times, events, lp, p, surv, t_ev_tr, H0_cum_tr, s_null_10, horizon=10.0, grid_1_13=None):
    """
    Computes point estimates of all calibration and accuracy metrics at horizon = 10 years:
    - O/E, slope, intercept
    - ICI, E50, E90 via Austin pseudo-splines
    - Brier score (10y), IBS (1-13y), IPA (10y)
    """
    # 1. Observed / Expected Ratio & Intercept
    O = 1.0 - fast_km_survival_at_t(times, events, horizon)
    E = float(np.mean(p))
    oe_ratio = O / E if E > 0 else np.nan
    intercept = float(np.log(oe_ratio)) if oe_ratio > 0 else np.nan
    
    # 2. Calibration Slope
    slope = fast_univariate_cox_slope(times, events, lp)
    
    # 3. Austin et al. (2020) Pseudo-Observations & ICI, E50, E90
    pseudo_risk = compute_jackknife_pseudo_observations(times, events, horizon)
    beta_spline, knots_used, smooth_risk = fit_austin_pseudo_spline(p, pseudo_risk)
    abs_errors = np.abs(smooth_risk - p)
    ici = float(np.mean(abs_errors))
    e50 = float(np.median(abs_errors))
    e90 = float(np.percentile(abs_errors, 90.0))
    
    # 4. IPCW Brier Score & IPA at 10 years
    loss_mod_10, loss_null_10, L_grid, grid_1_13 = precompute_ipcw_structures(
        times, events, lp, t_ev_tr, H0_cum_tr, s_null_10, horizon, grid_1_13
    )
    brier_val = float(np.mean(loss_mod_10))
    brier_null = float(np.mean(loss_null_10))
    ipa_val = float(1.0 - brier_val / brier_null) if brier_null > 0 else np.nan
    
    # 5. Integrated Brier Score over 1-13 years on common grid
    brier_curve = np.mean(L_grid, axis=0)
    ibs_val = float(np.trapezoid(brier_curve, grid_1_13) / (grid_1_13[-1] - grid_1_13[0]))
    
    return {
        "O": O, "E": E, "O_E": oe_ratio,
        "cal_slope": slope, "cal_intercept": intercept,
        "ICI": ici, "E50": e50, "E90": e90,
        "brier": brier_val, "brier_null": brier_null, "ipa": ipa_val,
        "ibs": ibs_val, "pseudo_risk": pseudo_risk,
        "beta_spline": beta_spline, "knots": knots_used,
        "smooth_risk": smooth_risk, "loss_mod_10": loss_mod_10,
        "loss_null_10": loss_null_10, "L_grid": L_grid, "grid_1_13": grid_1_13
    }

# -----------------------------------------------------------------------------
# 5. Bootstrap Inference (2,000 Resamples)
# -----------------------------------------------------------------------------
def run_bootstrap_metrics(times, events, lp, p, surv, t_ev_tr, H0_cum_tr, s_null_10, horizon=10.0, grid_1_13=None, n_boot=2000):
    """
    Computes 2,000 bootstrap resamples with fixed seed to obtain 95% confidence intervals
    for all calibration and accuracy metrics.
    """
    pt = compute_all_metrics(times, events, lp, p, surv, t_ev_tr, H0_cum_tr, s_null_10, horizon, grid_1_13)
    n = len(times)
    
    rng = np.random.RandomState(SEED)
    boot_indices = rng.choice(n, size=(n_boot, n), replace=True)
    
    boot_oe, boot_slope, boot_int = [], [], []
    boot_ici, boot_e50, boot_e90 = [], [], []
    boot_brier, boot_ipa, boot_ibs = [], [], []
    
    loss_mod_10 = pt["loss_mod_10"]
    loss_null_10 = pt["loss_null_10"]
    L_grid = pt["L_grid"]
    grid_times = pt["grid_1_13"]
    pseudo_risk = pt["pseudo_risk"]
    knots = pt["knots"]
    
    cloglog_p = np.log(-np.log(1.0 - np.clip(p, 1e-5, 1.0 - 1e-5)))
    s1, s2, _ = harrell_rcs_3knots(cloglog_p, knots=knots)
    X_spline = np.column_stack([np.ones_like(s1), s1, s2])
    
    logger.info(f"Running {n_boot} vectorized bootstrap resamples...")
    for idx_b in boot_indices:
        tb, eb, lpb, pb = times[idx_b], events[idx_b], lp[idx_b], p[idx_b]
        
        # O/E & Intercept
        ob = 1.0 - fast_km_survival_at_t(tb, eb, horizon)
        eb_mean = float(np.mean(pb))
        o_e_b = ob / eb_mean if eb_mean > 0 else np.nan
        boot_oe.append(o_e_b)
        boot_int.append(np.log(o_e_b) if o_e_b > 0 else np.nan)
        
        # Slope
        boot_slope.append(fast_univariate_cox_slope(tb, eb, lpb))
        
        # Brier & IPA
        bs_b = float(np.mean(loss_mod_10[idx_b]))
        bs_null_b = float(np.mean(loss_null_10[idx_b]))
        boot_brier.append(bs_b)
        boot_ipa.append(1.0 - bs_b / bs_null_b if bs_null_b > 0 else np.nan)
        
        # Fast vectorized IBS
        b_grid_b = np.mean(L_grid[idx_b, :], axis=0)
        ibs_b = float(np.trapezoid(b_grid_b, grid_times) / (grid_times[-1] - grid_times[0]))
        boot_ibs.append(ibs_b)
        
        # Austin pseudo-spline ICI, E50, E90
        X_b = X_spline[idx_b]
        y_b = pseudo_risk[idx_b]
        try:
            beta_b = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
            sm_b = np.clip(X_b @ beta_b, 0.0, 1.0)
            err_b = np.abs(sm_b - pb)
            boot_ici.append(float(np.mean(err_b)))
            boot_e50.append(float(np.median(err_b)))
            boot_e90.append(float(np.percentile(err_b, 90.0)))
        except Exception:
            boot_ici.append(pt["ICI"])
            boot_e50.append(pt["E50"])
            boot_e90.append(pt["E90"])
            
    # Calculate 95% Percentile CIs
    def get_ci(arr):
        v = np.array(arr)
        v = v[~np.isnan(v)]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        
    oe_l, oe_u = get_ci(boot_oe)
    sl_l, sl_u = get_ci(boot_slope)
    int_l, int_u = get_ci(boot_int)
    ici_l, ici_u = get_ci(boot_ici)
    e50_l, e50_u = get_ci(boot_e50)
    e90_l, e90_u = get_ci(boot_e90)
    bs_l, bs_u = get_ci(boot_brier)
    ipa_l, ipa_u = get_ci(boot_ipa)
    ibs_l, ibs_u = get_ci(boot_ibs)
    
    ci_dict = {
        "O_E_lower": oe_l, "O_E_upper": oe_u,
        "cal_slope_lower": sl_l, "cal_slope_upper": sl_u,
        "cal_intercept_lower": int_l, "cal_intercept_upper": int_u,
        "ICI_lower": ici_l, "ICI_upper": ici_u,
        "E50_lower": e50_l, "E50_upper": e50_u,
        "E90_lower": e90_l, "E90_upper": e90_u,
        "brier_lower": bs_l, "brier_upper": bs_u,
        "ibs_lower": ibs_l, "ibs_upper": ibs_u,
        "ipa_lower": ipa_l, "ipa_upper": ipa_u
    }
    return pt, ci_dict

# -----------------------------------------------------------------------------
# 6. Primary Publication Calibration Figure (300 DPI PNG & TIFF)
# -----------------------------------------------------------------------------
def plot_calibration_figure(fhs_data, whi_trans_data, whi_recal_data):
    """
    Generates a 300-DPI publication-grade, colorblind-friendly calibration figure:
    - Panel A: FHS Test Set (Primary) [10-year horizon]
    - Panel B: WHI External Cohort (Transported & Recalibrated) [10-year horizon]
    - Decile points with KM 95% error bars, Austin et al. (2020) smoothed pseudo-spline curve,
      identity line, annotation summary, and marginal histograms underneath.
    """
    logger.info("Generating publication calibration figure (300 DPI TIFF & PNG)...")
    
    # Colorblind-friendly palette (Okabe-Ito / ColorBrewer)
    C_TEST_PTS   = "#0072B2"  # Deep Blue
    C_TEST_LINE  = "#D55E00"  # Vermilion
    C_TRANS_PTS  = "#E69F00"  # Orange
    C_TRANS_LINE = "#E69F00"
    C_RECAL_PTS  = "#009E73"  # Bluish Green
    C_RECAL_LINE = "#009E73"
    C_PERFECT    = "#555555"  # Dark Gray
    C_HIST       = "#90A4AE"  # Soft Blue-Gray
    
    fig = plt.figure(figsize=(13.5, 6.8), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[4.2, 1.0], hspace=0.18, wspace=0.22)
    
    ax_test = fig.add_subplot(gs[0, 0])
    ax_hist_test = fig.add_subplot(gs[1, 0], sharex=ax_test)
    
    ax_whi = fig.add_subplot(gs[0, 1])
    ax_hist_whi = fig.add_subplot(gs[1, 1], sharex=ax_whi)
    
    # -------------------------------------------------------------------------
    # Panel A: FHS Test Set
    # -------------------------------------------------------------------------
    p_te = fhs_data["p"]
    t_te = fhs_data["time"]
    e_te = fhs_data["event"]
    pt_te = fhs_data["pt"]
    
    # 1. Identity line
    ax_test.plot([0, 0.72], [0, 0.72], linestyle='--', color=C_PERFECT, linewidth=1.5, zorder=1, label="Ideal Calibration ($y = x$)")
    
    # 2. Deciles of predicted risk vs KM observed
    df_te_dec = pd.DataFrame({"p": p_te, "time": t_te, "event": e_te})
    df_te_dec["bin"] = pd.qcut(df_te_dec["p"], q=10, duplicates="drop")
    
    dx_te, dy_te, dy_l_te, dy_u_te = [], [], [], []
    for _, grp in df_te_dec.groupby("bin", observed=True):
        if len(grp) == 0:
            continue
        dx_te.append(float(grp["p"].mean()))
        km = KaplanMeierFitter().fit(grp["time"], grp["event"])
        surv_10 = float(km.predict(10.0))
        obs_10 = 1.0 - surv_10
        dy_te.append(obs_10)
        
        ci_df = km.confidence_interval_
        sub_ci = ci_df.index[ci_df.index <= 10.0]
        if len(sub_ci) > 0:
            last_t = sub_ci[-1]
            s_lo = float(ci_df.loc[last_t].iloc[0])
            s_hi = float(ci_df.loc[last_t].iloc[1])
            dy_l_te.append(max(0.0, obs_10 - (1.0 - s_hi)))
            dy_u_te.append(max(0.0, (1.0 - s_lo) - obs_10))
        else:
            dy_l_te.append(0.0)
            dy_u_te.append(0.0)
            
    ax_test.errorbar(
        dx_te, dy_te, yerr=[dy_l_te, dy_u_te], fmt='o',
        color=C_TEST_PTS, ecolor=C_TEST_PTS, elinewidth=1.6, capsize=3.5, markersize=6.5,
        zorder=4, label="Deciles of Risk (KM observed $\\pm$ 95% CI)"
    )
    
    # 3. Austin et al. (2020) Pseudo-spline smoothed curve
    dense_p = np.linspace(p_te.min() * 1.01, min(p_te.max() * 0.99, 0.72), 150)
    dense_sm = predict_austin_pseudo_spline(dense_p, pt_te["beta_spline"], pt_te["knots"])
    ax_test.plot(dense_p, dense_sm, color=C_TEST_LINE, linewidth=2.4, zorder=3, label="Austin Smoothed Curve (Pseudo-RCS)")
    
    # Marginal Histogram underneath
    ax_hist_test.hist(p_te, bins=25, color=C_HIST, edgecolor="white", linewidth=0.8, density=True)
    ax_hist_test.set_xlabel("Predicted 10-Year Mortality Risk", fontsize=11, fontweight="bold")
    ax_hist_test.set_ylabel("Density", fontsize=9)
    ax_hist_test.set_yticks([])
    
    ax_test.set_xlim(-0.02, 0.72)
    ax_test.set_ylim(-0.02, 0.72)
    ax_test.set_ylabel("Observed 10-Year Mortality (Kaplan–Meier)", fontsize=11, fontweight="bold")
    ax_test.set_title("(A) FHS Test Cohort (Primary External Benchmark, N = 699)", fontsize=12, fontweight="bold", pad=8)
    ax_test.grid(True, linestyle=":", alpha=0.55)
    ax_test.tick_params(labelbottom=False)
    
    # Annotation box for FHS Test
    annot_test = (
        f"$\\mathbf{{FHS\\ Test\\ (10\\text{{-}}Year\\ Horizon)}}$\n"
        f"O/E Ratio: {pt_te['O_E']:.3f} [{fhs_data['ci']['O_E_lower']:.3f}–{fhs_data['ci']['O_E_upper']:.3f}]\n"
        f"Cal. Slope: {pt_te['cal_slope']:.3f} [{fhs_data['ci']['cal_slope_lower']:.3f}–{fhs_data['ci']['cal_slope_upper']:.3f}]\n"
        f"Intercept:  {pt_te['cal_intercept']:+.3f} [{fhs_data['ci']['cal_intercept_lower']:+.3f}–{fhs_data['ci']['cal_intercept_upper']:+.3f}]\n"
        f"ICI: {pt_te['ICI']:.4f} | E50: {pt_te['E50']:.4f}\n"
        f"IPCW Brier: {pt_te['brier']:.4f} | IPA: {pt_te['ipa']*100:+.1f}%\n"
        f"IBS (1–13y): {pt_te['ibs']:.4f}"
    )
    ax_test.text(
        0.04, 0.96, annot_test, transform=ax_test.transAxes, fontsize=8.8,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CFD8DC', alpha=0.92)
    )
    ax_test.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel B: WHI External Cohort (Transported & Recalibrated)
    # -------------------------------------------------------------------------
    p_whi_tr = whi_trans_data["p"]
    p_whi_rc = whi_recal_data["p"]
    t_whi = whi_trans_data["time"]
    e_whi = whi_trans_data["event"]
    pt_whi_tr = whi_trans_data["pt"]
    pt_whi_rc = whi_recal_data["pt"]
    
    ax_whi.plot([0, 0.45], [0, 0.45], linestyle='--', color=C_PERFECT, linewidth=1.5, zorder=1, label="Ideal Calibration ($y = x$)")
    
    # 1. Transported groups
    df_whi_tr = pd.DataFrame({"p": p_whi_tr, "time": t_whi, "event": e_whi})
    df_whi_tr["bin"] = pd.qcut(df_whi_tr["p"], q=5, duplicates="drop")
    
    dx_w_tr, dy_w_tr, dy_l_w_tr, dy_u_w_tr = [], [], [], []
    for _, grp in df_whi_tr.groupby("bin", observed=True):
        if len(grp) == 0:
            continue
        dx_w_tr.append(float(grp["p"].mean()))
        km = KaplanMeierFitter().fit(grp["time"], grp["event"])
        surv_10 = float(km.predict(10.0))
        obs_10 = 1.0 - surv_10
        dy_w_tr.append(obs_10)
        
        ci_df = km.confidence_interval_
        sub_ci = ci_df.index[ci_df.index <= 10.0]
        if len(sub_ci) > 0:
            last_t = sub_ci[-1]
            s_lo = float(ci_df.loc[last_t].iloc[0])
            s_hi = float(ci_df.loc[last_t].iloc[1])
            dy_l_w_tr.append(max(0.0, obs_10 - (1.0 - s_hi)))
            dy_u_w_tr.append(max(0.0, (1.0 - s_lo) - obs_10))
        else:
            dy_l_w_tr.append(0.0)
            dy_u_w_tr.append(0.0)
            
    ax_whi.errorbar(
        dx_w_tr, dy_w_tr, yerr=[dy_l_w_tr, dy_u_w_tr], fmt='s',
        color=C_TRANS_PTS, ecolor=C_TRANS_PTS, elinewidth=1.6, capsize=3.5, markersize=6.0,
        zorder=4, label="Transported Groups (KM observed $\\pm$ 95% CI)"
    )
    
    dense_p_w = np.linspace(p_whi_tr.min() * 1.01, min(p_whi_tr.max() * 0.99, 0.45), 150)
    dense_sm_w_tr = predict_austin_pseudo_spline(dense_p_w, pt_whi_tr["beta_spline"], pt_whi_tr["knots"])
    ax_whi.plot(dense_p_w, dense_sm_w_tr, color=C_TRANS_LINE, linestyle='-', linewidth=2.2, zorder=3, label="Transported Smoothed Curve")
    
    # 2. Recalibrated in-the-large groups
    df_whi_rc = pd.DataFrame({"p": p_whi_rc, "time": t_whi, "event": e_whi})
    df_whi_rc["bin"] = pd.qcut(df_whi_rc["p"], q=5, duplicates="drop")
    dx_w_rc, dy_w_rc, dy_l_w_rc, dy_u_w_rc = [], [], [], []
    for _, grp in df_whi_rc.groupby("bin", observed=True):
        if len(grp) == 0:
            continue
        dx_w_rc.append(float(grp["p"].mean()))
        km = KaplanMeierFitter().fit(grp["time"], grp["event"])
        surv_10 = float(km.predict(10.0))
        obs_10 = 1.0 - surv_10
        dy_w_rc.append(obs_10)
        ci_df = km.confidence_interval_
        sub_ci = ci_df.index[ci_df.index <= 10.0]
        if len(sub_ci) > 0:
            last_t = sub_ci[-1]
            s_lo = float(ci_df.loc[last_t].iloc[0])
            s_hi = float(ci_df.loc[last_t].iloc[1])
            dy_l_w_rc.append(max(0.0, obs_10 - (1.0 - s_hi)))
            dy_u_w_rc.append(max(0.0, (1.0 - s_lo) - obs_10))
        else:
            dy_l_w_rc.append(0.0)
            dy_u_w_rc.append(0.0)
            
    ax_whi.errorbar(
        dx_w_rc, dy_w_rc, yerr=[dy_l_w_rc, dy_u_w_rc], fmt='o',
        color=C_RECAL_PTS, ecolor=C_RECAL_PTS, elinewidth=1.6, capsize=3.5, markersize=6.5,
        zorder=5, label="Recalibrated Groups (KM observed $\\pm$ 95% CI)"
    )
    
    dense_p_w_rc = np.linspace(p_whi_rc.min() * 1.01, min(p_whi_rc.max() * 0.99, 0.45), 150)
    dense_sm_w_rc = predict_austin_pseudo_spline(dense_p_w_rc, pt_whi_rc["beta_spline"], pt_whi_rc["knots"])
    ax_whi.plot(dense_p_w_rc, dense_sm_w_rc, color=C_RECAL_LINE, linestyle='-', linewidth=2.4, zorder=4, label="Recalibrated Smoothed Curve")
    
    # Histogram underneath WHI
    ax_hist_whi.hist(p_whi_tr, bins=20, color=C_HIST, edgecolor="white", linewidth=0.8, density=True, alpha=0.75, label="Transported")
    ax_hist_whi.set_xlabel("Predicted 10-Year Mortality Risk", fontsize=11, fontweight="bold")
    ax_hist_whi.set_yticks([])
    
    ax_whi.set_xlim(-0.01, 0.45)
    ax_whi.set_ylim(-0.01, 0.45)
    ax_whi.set_ylabel("Observed 10-Year Mortality (Kaplan–Meier)", fontsize=11, fontweight="bold")
    ax_whi.set_title("(B) WHI External Validation Cohort (All-Female, N = 196)", fontsize=12, fontweight="bold", pad=8)
    ax_whi.grid(True, linestyle=":", alpha=0.55)
    ax_whi.tick_params(labelbottom=False)
    
    # Annotation box for WHI
    annot_whi = (
        f"$\\mathbf{{WHI\\ (10\\text{{-}}Year\\ Horizon)}}$\n"
        f"Transported (as-is):\n"
        f"  O/E: {pt_whi_tr['O_E']:.3f} [{whi_trans_data['ci']['O_E_lower']:.3f}–{whi_trans_data['ci']['O_E_upper']:.3f}]\n"
        f"  Slope: {pt_whi_tr['cal_slope']:.3f} | Intercept: {pt_whi_tr['cal_intercept']:+.3f}\n"
        f"  ICI: {pt_whi_tr['ICI']:.4f} | Brier: {pt_whi_tr['brier']:.4f}\n"
        f"Recalibrated-in-the-Large:\n"
        f"  O/E: {pt_whi_rc['O_E']:.3f} [{whi_recal_data['ci']['O_E_lower']:.3f}–{whi_recal_data['ci']['O_E_upper']:.3f}]\n"
        f"  Slope: {pt_whi_rc['cal_slope']:.3f} | Intercept: {pt_whi_rc['cal_intercept']:+.3f}\n"
        f"  ICI: {pt_whi_rc['ICI']:.4f} | Brier: {pt_whi_rc['brier']:.4f}"
    )
    ax_whi.text(
        0.04, 0.96, annot_whi, transform=ax_whi.transAxes, fontsize=8.4,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CFD8DC', alpha=0.92)
    )
    ax_whi.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.2)
    
    # Save formats
    png_path = FIG_DIR / "reviewer1_calibration_final_model.png"
    tif_path = FIG_DIR / "reviewer1_calibration_final_model.tif"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(tif_path, dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    
    # Also save to calibration_plots.png / .tif for backward compatibility
    plt.savefig(FIG_DIR / "calibration_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / "calibration_plots.tif", dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    
    plt.close()
    logger.info(f"Saved PNG calibration figure to: {png_path}")
    logger.info(f"Saved 300-DPI TIFF calibration figure to: {tif_path}")

# -----------------------------------------------------------------------------
# 7. Cross-Check with R riskRegression::Score()
# -----------------------------------------------------------------------------
def run_r_crosscheck(fhs_test_df, surv_test_10):
    """
    Executes cross-check in R using riskRegression::Score() and prints comparison table.
    Confirms agreement of Brier score, IPA, and calibration curve to within about 0.005.
    """
    logger.info("==========================================================================")
    logger.info("STEP 5: R riskRegression::Score() CROSS-CHECK VALIDATION")
    logger.info("==========================================================================")
    
    r_script_path = BASE_DIR / "scratch" / "r_calibration_crosscheck.R"
    r_script_content = f"""
library(riskRegression)
library(survival)

test_df <- read.csv("predictions/coxnet_test_V3_OPTUNA_SS.csv")
train_df <- read.csv("predictions/coxnet_train_V3_OPTUNA_SS.csv")

# Baseline cumulative hazard
train_df <- train_df[order(train_df$time), ]
w_tr <- exp(train_df$risk_score)
r_sum <- rev(cumsum(rev(w_tr)))
ev_mask <- (train_df$event == "True" | train_df$event == 1 | train_df$event == TRUE)
t_ev <- train_df$time[ev_mask]
r_ev <- r_sum[ev_mask]
dh <- 1 / r_ev
H0_cum <- cumsum(dh)

get_surv <- function(eval_t, lp) {{
  H0 <- sapply(eval_t, function(tt) {{
    idx <- max(which(t_ev <= tt))
    if (is.infinite(idx) || length(idx) == 0 || idx <= 0) return(0)
    return(H0_cum[idx])
  }})
  outer(exp(lp), H0, function(w, h) exp(-w * h))
}}

test_df$event_num <- as.numeric(test_df$event == "True" | test_df$event == 1 | test_df$event == TRUE)

p_10 <- 1.0 - get_surv(10.0, test_df$risk_score)
p_mat_10 <- matrix(p_10, ncol = 1)

# Fit Score at 10 years
sc_10 <- Score(list("coxnet"=p_mat_10), formula=Surv(time, event_num)~1, data=test_df, times=10.0, metrics=c("brier"), summary="ipa", plots="calibration")
brier_res <- sc_10$Brier$score
cox_brier <- brier_res[brier_res$model == "coxnet", ]
null_brier <- brier_res[brier_res$model == "Null model", ]

r_brier <- cox_brier$Brier
r_null_brier <- null_brier$Brier
r_ipa <- 1.0 - (r_brier / r_null_brier)

cat(sprintf("R_BRIER_10Y=%.6f\n", r_brier))
cat(sprintf("R_NULL_BRIER_10Y=%.6f\n", r_null_brier))
cat(sprintf("R_IPA_10Y=%.6f\n", r_ipa))

# Generate R calibration plot for visual cross-check
png("figures/r_calibration_crosscheck.png", width=1800, height=1800, res=300)
plotCalibration(sc_10, times=10.0, models="coxnet", xlab="Predicted 10-Year Mortality Risk", ylab="Observed Mortality")
dev.off()
cat("R calibration plot saved to figures/r_calibration_crosscheck.png\n")
"""
    with open(r_script_path, "w", encoding="utf-8") as f:
        f.write(r_script_content)
        
    import shutil
    rscript_exe = shutil.which("Rscript") or os.environ.get("RSCRIPT_PATH", r"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe")
    res = subprocess.run([rscript_exe, str(r_script_path)], capture_output=True, text=True)
    logger.info("R script output:\n" + res.stdout)
    if res.stderr:
        logger.warning("R script warnings/errors:\n" + res.stderr)
        
    r_brier, r_ipa = None, None
    for line in res.stdout.splitlines():
        if line.startswith("R_BRIER_10Y="):
            r_brier = float(line.split("=")[1])
        elif line.startswith("R_IPA_10Y="):
            r_ipa = float(line.split("=")[1])
            
    return r_brier, r_ipa

# -----------------------------------------------------------------------------
# 8. Main Execution Orchestrator
# -----------------------------------------------------------------------------
def main():
    logger.info("==========================================================================")
    logger.info("STARTING CALIBRATION ANALYSIS FOR REVIEWER 1, COMMENT 10 (FINAL MODEL)")
    logger.info("==========================================================================")
    
    # Step 1: Load Cohort Splits
    splits = load_data()
    
    # Step 2: Log Number at Risk at 5, 8, 10 years
    df_audit = log_number_at_risk(splits)
    
    # Step 3: Breslow Baseline Hazard Estimation on FHS Train
    train_df = splits["FHS Train (apparent)"]
    t_tr = train_df['time'].values
    e_tr = train_df['event'].values
    lp_tr = train_df['risk_score'].values
    
    t_ev_tr, H0_cum_tr = compute_breslow_cumulative_hazard(t_tr, e_tr, lp_tr)
    H0_tr_10 = get_h0_at_t(t_ev_tr, H0_cum_tr, HORIZON)
    S0_tr_10 = float(np.exp(-H0_tr_10))
    
    km_train = KaplanMeierFitter().fit(t_tr, e_tr)
    s_null_10 = float(km_train.predict(HORIZON))
    
    logger.info(f"FHS Training Split Baseline Hazard at 10y: H0(10) = {H0_tr_10:.6f} | S0(10) = {S0_tr_10:.6f}")
    logger.info(f"FHS Training Split Null Model KM S(10y):   S_null(10) = {s_null_10:.6f}")
    
    # Common 1-13 year grid for IBS
    grid_1_13 = np.linspace(1.0, 13.0, 50)
    
    # Step 4: Evaluate Each Target Split
    eval_configs = [
        ("FHS Test", "none", splits["FHS Test"]),
        ("WHI", "none", splits["WHI"]),
        ("WHI", "in-the-large", splits["WHI"]),
        ("FHS Train (apparent)", "none", splits["FHS Train (apparent)"]),
        ("FHS Val", "none", splits["FHS Val"])
    ]
    
    csv_rows = []
    plot_data_dict = {}
    
    for split_name, recal_type, df_curr in eval_configs:
        logger.info(f"--- Evaluating {split_name} (Recalibration: {recal_type}) ---")
        t_curr = df_curr['time'].values
        e_curr = df_curr['event'].values
        lp_curr = df_curr['risk_score'].values
        
        if recal_type == "in-the-large":
            # Re-estimate only 10-year baseline survival in WHI, keeping coefficients fixed
            t_ev_whi, H0_cum_whi = compute_breslow_cumulative_hazard(t_curr, e_curr, lp_curr)
            H0_eval_10 = get_h0_at_t(t_ev_whi, H0_cum_whi, HORIZON)
            S0_eval_10 = float(np.exp(-H0_eval_10))
            logger.info(f"WHI Recalibration-in-the-large: H0_whi(10) = {H0_eval_10:.6f} | S0_whi(10) = {S0_eval_10:.6f}")
        else:
            H0_eval_10 = H0_tr_10
            S0_eval_10 = S0_tr_10
            
        # Predicted absolute risks and survival at 10 years
        surv_curr = np.exp(-H0_eval_10 * np.exp(lp_curr))
        p_curr = np.clip(1.0 - surv_curr, 1e-5, 1.0 - 1e-5)
        
        # Point estimates & 2,000 bootstrap CIs
        pt, ci = run_bootstrap_metrics(
            t_curr, e_curr, lp_curr, p_curr, surv_curr,
            t_ev_tr, H0_cum_tr, s_null_10, horizon=HORIZON,
            grid_1_13=grid_1_13, n_boot=N_BOOT
        )
        
        row = {
            "split": split_name,
            "horizon": HORIZON,
            "recalibration": recal_type,
            "O_E": round(pt["O_E"], 3),
            "O_E_lower": round(ci["O_E_lower"], 3),
            "O_E_upper": round(ci["O_E_upper"], 3),
            "cal_slope": round(pt["cal_slope"], 3),
            "cal_slope_lower": round(ci["cal_slope_lower"], 3),
            "cal_slope_upper": round(ci["cal_slope_upper"], 3),
            "cal_intercept": round(pt["cal_intercept"], 3),
            "cal_intercept_lower": round(ci["cal_intercept_lower"], 3),
            "cal_intercept_upper": round(ci["cal_intercept_upper"], 3),
            "ICI": round(pt["ICI"], 4),
            "ICI_lower": round(ci["ICI_lower"], 4),
            "ICI_upper": round(ci["ICI_upper"], 4),
            "E50": round(pt["E50"], 4),
            "E50_lower": round(ci["E50_lower"], 4),
            "E50_upper": round(ci["E50_upper"], 4),
            "E90": round(pt["E90"], 4),
            "E90_lower": round(ci["E90_lower"], 4),
            "E90_upper": round(ci["E90_upper"], 4),
            "brier": round(pt["brier"], 4),
            "brier_lower": round(ci["brier_lower"], 4),
            "brier_upper": round(ci["brier_upper"], 4),
            "ibs": round(pt["ibs"], 4),
            "ibs_lower": round(ci["ibs_lower"], 4),
            "ibs_upper": round(ci["ibs_upper"], 4),
            "ipa": round(pt["ipa"], 4),
            "ipa_lower": round(ci["ipa_lower"], 4),
            "ipa_upper": round(ci["ipa_upper"], 4),
            # Formatted presentation strings
            "O_E_95_ci": f"{pt['O_E']:.3f} ({ci['O_E_lower']:.3f} - {ci['O_E_upper']:.3f})",
            "cal_slope_95_ci": f"{pt['cal_slope']:.3f} ({ci['cal_slope_lower']:.3f} - {ci['cal_slope_upper']:.3f})",
            "cal_intercept_95_ci": f"{pt['cal_intercept']:+.3f} ({ci['cal_intercept_lower']:+.3f} - {ci['cal_intercept_upper']:+.3f})",
            "ICI_95_ci": f"{pt['ICI']:.4f} ({ci['ICI_lower']:.4f} - {ci['ICI_upper']:.4f})",
            "brier_95_ci": f"{pt['brier']:.4f} ({ci['brier_lower']:.4f} - {ci['brier_upper']:.4f})",
            "ibs_95_ci": f"{pt['ibs']:.4f} ({ci['ibs_lower']:.4f} - {ci['ibs_upper']:.4f})",
            "ipa_95_ci": f"{pt['ipa']*100:.2f}% ({ci['ipa_lower']*100:.2f}% - {ci['ipa_upper']*100:.2f}%)"
        }
        csv_rows.append(row)
        
        plot_data_dict[(split_name, recal_type)] = {
            "p": p_curr, "time": t_curr, "event": e_curr, "pt": pt, "ci": ci
        }
        
    df_main = pd.DataFrame(csv_rows)
    main_csv_path = BASE_DIR / "reviewer1_calibration_main.csv"
    df_main.to_csv(main_csv_path, index=False)
    logger.info(f"Saved primary metrics table to: {main_csv_path}")
    
    # Step 5: Generate Calibration Plot
    plot_calibration_figure(
        plot_data_dict[("FHS Test", "none")],
        plot_data_dict[("WHI", "none")],
        plot_data_dict[("WHI", "in-the-large")]
    )
    
    # Step 6: R Cross-Check
    fhs_test_surv_10 = plot_data_dict[("FHS Test", "none")]["pt"]["loss_mod_10"]
    r_brier, r_ipa = run_r_crosscheck(splits["FHS Test"], fhs_test_surv_10)
    
    py_brier = df_main.loc[df_main['split'] == 'FHS Test', 'brier'].values[0]
    py_ipa = df_main.loc[df_main['split'] == 'FHS Test', 'ipa'].values[0]
    
    logger.info("==========================================================================")
    logger.info("CROSS-CHECK COMPARISON SUMMARY (PYTHON vs R riskRegression::Score)")
    logger.info("==========================================================================")
    logger.info(f"Python FHS Test 10y IPCW Brier: {py_brier:.5f} | R: {r_brier:.5f} | Diff: {abs(py_brier - r_brier):.5f}")
    logger.info(f"Python FHS Test 10y IPA:        {py_ipa:.5f} | R: {r_ipa:.5f} | Diff: {abs(py_ipa - r_ipa):.5f}")
    
    agreement_brier = abs(py_brier - r_brier) < 0.005
    agreement_ipa = abs(py_ipa - r_ipa) < 0.005
    logger.info(f"Brier score agreement within 0.005: {agreement_brier}")
    logger.info(f"IPA agreement within 0.005:         {agreement_ipa}")
    
    logger.info("==========================================================================")
    logger.info("CALIBRATION PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("==========================================================================")

if __name__ == "__main__":
    main()
