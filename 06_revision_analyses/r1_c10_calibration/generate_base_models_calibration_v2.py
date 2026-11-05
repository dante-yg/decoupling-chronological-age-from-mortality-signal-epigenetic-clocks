"""
Generate Refined Publication-Quality Calibration Plots for Base Models
Manuscript: AGG-D-26-04256
Evaluates:
  1. CpG-Only Model (10 EpiClock CpGs)
  2. CpG + Clinical Model (10 CpGs + 4 Clinical Biomarkers)
Across:
  1. FHS Held-Out Test Set (N = 699, 127 total events)
  2. WHI External Validation Cohort (N = 196, 56 total events)
At:
  Primary: 14.0 Years (End of Follow-Up Time Point)
  Secondary: 10.0 Years (Standard 10-Year Horizon)
"""
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



import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.metrics import brier_score

# Paths
BASE_DIR = config.PROJECT_ROOT
PRED_DIR = BASE_DIR / "predictions"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
train_ss = pd.read_csv(PRED_DIR / "coxnet_train_V3_OPTUNA_SS.csv")
train_cpg = pd.read_csv(PRED_DIR / "coxnet_train_V3_OPTUNA_SS_CPG.csv")
test_ss = pd.read_csv(PRED_DIR / "coxnet_test_V3_OPTUNA_SS.csv")
test_cpg = pd.read_csv(PRED_DIR / "coxnet_test_V3_OPTUNA_SS_CPG.csv")
whi_ss = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv")
whi_cpg = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS_CPG.csv")

for df in [train_ss, train_cpg, test_ss, test_cpg, whi_ss, whi_cpg]:
    df['event'] = (df['event'] == True) | (df['event'] == 1) | (df['event'] == "True")
    df['event'] = df['event'].astype(int)

# 2. Breslow Cumulative Baseline Hazard Function on Training Split
def compute_breslow(times, events, lp):
    w = np.exp(lp)
    order = np.argsort(times)
    t_s, e_s, w_s = times[order], events[order], w[order]
    r_sum = np.cumsum(w_s[::-1])[::-1]
    ev_mask = (e_s == 1)
    dh = np.zeros_like(r_sum, dtype=float)
    dh[ev_mask] = 1.0 / r_sum[ev_mask]
    return t_s[ev_mask], np.cumsum(dh[ev_mask])

t_ev_cpg, H0_cum_cpg = compute_breslow(train_cpg['time'].values, train_cpg['event'].values, train_cpg['risk_score'].values)
t_ev_ss, H0_cum_ss = compute_breslow(train_ss['time'].values, train_ss['event'].values, train_ss['risk_score'].values)

# Harrell 3-Knot Restricted Cubic Spline Basis
def harrell_rcs_3knots(x, knots=None):
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
    return pd.DataFrame({'rcs1': s1, 'rcs2': s2}), knots

# 3. Comprehensive Calibration Function
def evaluate_calibration(train_df, eval_df, lp_col, t_ev, H0_cum, t_horizon, cohort_name, model_name, n_bins=10):
    idx = np.searchsorted(t_ev, t_horizon, side='right') - 1
    h0 = H0_cum[idx]
    
    lp = eval_df[lp_col].values
    p = 1.0 - np.exp(-h0 * np.exp(lp))
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    
    df_eval = pd.DataFrame({
        "time": eval_df['time'].values,
        "event": eval_df['event'].values,
        "p": p,
        "lp": lp
    })
    
    # Stratify into risk deciles
    try:
        df_eval['bin'] = pd.qcut(df_eval['p'], q=n_bins, labels=False, duplicates='drop') + 1
    except Exception:
        df_eval['bin'] = pd.cut(df_eval['p'], bins=n_bins, labels=False) + 1
        
    decile_records = []
    for b in sorted(df_eval['bin'].unique()):
        grp = df_eval[df_eval['bin'] == b]
        n_b = len(grp)
        ev_t = int(((grp['time'] <= t_horizon) & (grp['event'] == 1)).sum())
        mean_p = float(grp['p'].mean())
        
        km = KaplanMeierFitter().fit(grp['time'], grp['event'])
        surv_t = float(km.predict(t_horizon))
        obs_t = 1.0 - surv_t
        
        ci_df = km.confidence_interval_
        sub_ci = ci_df.index[ci_df.index <= t_horizon]
        if len(sub_ci) > 0 and ev_t > 0:
            last_t = sub_ci[-1]
            s_lo = float(ci_df.loc[last_t].iloc[0])
            s_hi = float(ci_df.loc[last_t].iloc[1])
            lower_ci = max(0.0, 1.0 - s_hi)
            upper_ci = min(1.0, 1.0 - s_lo)
        else:
            lower_ci, upper_ci = obs_t, obs_t
            
        decile_records.append({
            "cohort": cohort_name,
            "model": model_name,
            "horizon_years": t_horizon,
            "bin": int(b),
            "n": int(n_b),
            "events": int(ev_t),
            "mean_pred_risk": mean_p,
            "observed_risk": obs_t,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci
        })
        
    df_dec = pd.DataFrame(decile_records)
    
    # Linear decile fit
    slope, intercept = np.polyfit(df_dec['mean_pred_risk'], df_dec['observed_risk'], 1)
    r = np.corrcoef(df_dec['mean_pred_risk'], df_dec['observed_risk'])[0, 1]
    
    # Overall summary stats
    km_all = KaplanMeierFitter().fit(eval_df['time'], eval_df['event'])
    obs_all = 1.0 - float(km_all.predict(t_horizon))
    pred_all = float(p.mean())
    oe_ratio = obs_all / pred_all if pred_all > 0 else np.nan
    total_events = int(((eval_df['time'] <= t_horizon) & (eval_df['event'] == 1)).sum())
    
    # Cox calibration slope
    cph = CoxPHFitter().fit(df_eval[['time', 'event', 'lp']], duration_col='time', event_col='event')
    cox_slope = float(cph.params_['lp'])
    cox_se = float(cph.standard_errors_['lp'])
    cox_slope_ci = (cox_slope - 1.96*cox_se, cox_slope + 1.96*cox_se)
    
    # sksurv IPCW Brier score
    surv_tr = np.array([(e, t) for e, t in zip(train_df['event'], train_df['time'])], dtype=[('event', bool), ('time', float)])
    surv_ev = np.array([(e, t) for e, t in zip(eval_df['event'], eval_df['time'])], dtype=[('event', bool), ('time', float)])
    s_pred = 1.0 - p
    try:
        _, bs_arr = brier_score(surv_tr, surv_ev, s_pred, times=[t_horizon])
        brier = float(bs_arr[0])
    except Exception:
        brier = np.nan
        
    # Austin et al. (2020) Smoothed Calibration Curve via RCS
    cloglog_p = np.log(-np.log(np.maximum(1.0 - p, 1e-6)))
    df_rcs, knots = harrell_rcs_3knots(cloglog_p)
    df_rcs['time'] = df_eval['time'].values
    df_rcs['event'] = df_eval['event'].values
    try:
        cph_sm = CoxPHFitter(penalizer=1e-3).fit(df_rcs, duration_col='time', event_col='event')
        
        # Grid of predicted risks for smoothing curve
        p_grid = np.linspace(min(p)*1.02, min(max(p)*0.98, 0.95), 100)
        cloglog_grid = np.log(-np.log(np.maximum(1.0 - p_grid, 1e-6)))
        grid_rcs, _ = harrell_rcs_3knots(cloglog_grid, knots=knots)
        sm_surv = cph_sm.predict_survival_function(grid_rcs[['rcs1', 'rcs2']], times=[t_horizon]).values.flatten()
        sm_risk = np.clip(1.0 - sm_surv, 0.0, 1.0)
        
        # ICI across participants
        sm_surv_ind = cph_sm.predict_survival_function(df_rcs[['rcs1', 'rcs2']], times=[t_horizon]).values.flatten()
        ici = float(np.mean(np.abs(p - (1.0 - sm_surv_ind))))
    except Exception:
        p_grid = np.linspace(min(p), max(p), 100)
        sm_risk = slope * p_grid + intercept
        ici = float(np.mean(np.abs(df_dec['observed_risk'] - df_dec['mean_pred_risk'])))
        
    stats = {
        "cohort": cohort_name,
        "model": model_name,
        "horizon_years": t_horizon,
        "total_n": len(eval_df),
        "total_events": total_events,
        "mean_predicted": pred_all,
        "overall_observed": obs_all,
        "oe_ratio": oe_ratio,
        "fit_slope": slope,
        "fit_intercept": intercept,
        "correlation_r": r,
        "cox_slope": cox_slope,
        "cox_slope_lower": cox_slope_ci[0],
        "cox_slope_upper": cox_slope_ci[1],
        "ici": ici,
        "brier": brier
    }
    
    smooth_curve = (p_grid, sm_risk)
    
    return df_dec, stats, p, smooth_curve

# 4. Multi-Panel Figure Generator
def generate_calibration_figure(t_horizon, horizon_label, filename_base):
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.1

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 12.5), dpi=300)
    
    configs = [
        # (row, col, tr_df, ev_df, lp_col, t_ev, H0_cum, cohort, model, color, title, ax_lim, box_loc)
        (0, 0, train_cpg, test_cpg, 'risk_score', t_ev_cpg, H0_cum_cpg,
         "FHS Test", "CpG-Only Model (10 CpGs)", "#0072B2",
         f"(a) FHS Test — CpG-Only ({int(t_horizon)}y)",
         (-0.02, 0.60 if t_horizon==14.0 else 0.48), (0.04, 0.96)),
         
        (0, 1, train_ss, test_ss, 'risk_score', t_ev_ss, H0_cum_ss,
         "FHS Test", "CpG + Clinical Model (14 Features)", "#D55E00",
         f"(b) FHS Test — CpG + Clinical ({int(t_horizon)}y)",
         (-0.02, 0.60 if t_horizon==14.0 else 0.48), (0.04, 0.96)),
         
        (1, 0, train_cpg, whi_cpg, 'risk_score', t_ev_cpg, H0_cum_cpg,
         "WHI External", "CpG-Only Model (10 CpGs)", "#0072B2",
         f"(c) WHI External — CpG-Only ({int(t_horizon)}y)",
         (-0.015, 0.44 if t_horizon==14.0 else 0.24), (0.50, 0.96)),
         
        (1, 1, train_ss, whi_ss, 'risk_score', t_ev_ss, H0_cum_ss,
         "WHI External", "CpG + Clinical Model (14 Features)", "#D55E00",
         f"(d) WHI External — CpG + Clinical ({int(t_horizon)}y)",
         (-0.015, 0.44 if t_horizon==14.0 else 0.24), (0.50, 0.96))
    ]
    
    decile_dfs = []
    stat_records = []
    
    for r, c, tr_d, ev_d, lp_col, t_ev, H0_cum, coh, mod, col, panel_title, (ax_min, ax_max), (bx, by) in configs:
        ax = axes[r, c]
        df_dec, stats, p_arr, (p_grid, sm_risk) = evaluate_calibration(
            tr_d, ev_d, lp_col, t_ev, H0_cum, t_horizon, coh, mod, n_bins=10
        )
        decile_dfs.append(df_dec)
        stat_records.append(stats)
        
        # 1. 45° Ideal Line
        ax.plot([0, 1], [0, 1], linestyle='--', color='#90A4AE', linewidth=1.6, zorder=1, label='Perfect Calibration ($y = x$)')
        
        # 2. Austin et al. (2020) Smoothed Calibration Curve
        ax.plot(p_grid, sm_risk, linestyle='-', color='#C62828', linewidth=2.3, alpha=0.90, zorder=3,
                label='Smoothed Calibration Curve (RCS)')
                
        # 3. Decile Points with Greenwood 95% Confidence Intervals
        y_err_lo = np.maximum(0, df_dec['observed_risk'] - df_dec['lower_ci'])
        y_err_hi = np.maximum(0, df_dec['upper_ci'] - df_dec['observed_risk'])
        ax.errorbar(
            df_dec['mean_pred_risk'], df_dec['observed_risk'],
            yerr=[y_err_lo, y_err_hi],
            fmt='o', color=col, ecolor=col, elinewidth=1.6, capsize=4.0, capthick=1.4,
            markersize=7.5, markeredgecolor='white', markeredgewidth=1.2, zorder=4,
            label='Risk Deciles (KM Observed ± 95% CI)'
        )
        
        # 4. Rug Plot of individual predicted risks
        ax.scatter(p_arr, np.full_like(p_arr, ax_min + 0.005), marker='|', color=col, alpha=0.45, s=28, zorder=2)
        
        # 5. Annotation Box (Clean, Professional Typography)
        box_text = (
            f"N = {stats['total_n']} (Events = {stats['total_events']})\n"
            f"Mean Pred. Risk: {stats['mean_predicted']*100:.1f}%\n"
            f"Observed KM Risk: {stats['overall_observed']*100:.1f}%\n"
            f"O/E Ratio:       {stats['oe_ratio']:.2f}\n"
            f"Calib. Slope:    {stats['fit_slope']:.2f}\n"
            f"Cox Hazard Slope: {stats['cox_slope']:.2f} ({stats['cox_slope_lower']:.2f}–{stats['cox_slope_upper']:.2f})\n"
            f"Calib. Intercept: {stats['fit_intercept']:+.3f}\n"
            f"ICI (Mean Error): {stats['ici']*100:.2f}%\n"
            f"IPCW Brier Score: {stats['brier']:.4f}"
        )
        ax.text(
            bx, by, box_text, transform=ax.transAxes, fontsize=8.8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#F8F9FA', edgecolor='#CFD8DC', alpha=0.94),
            family='monospace'
        )
        
        ax.set_title(panel_title, fontsize=12.0, fontweight='bold', pad=8, loc='left')
        ax.set_xlabel(f"Predicted {int(t_horizon)}-Year Mortality Risk", fontsize=10.5, fontweight='bold', labelpad=7)
        ax.set_ylabel(f"Observed {int(t_horizon)}-Year Mortality Risk (KM)", fontsize=10.5, fontweight='bold', labelpad=7)
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.55, color='#CFD8DC')
        ax.legend(loc='lower right', fontsize=8.5, frameon=True, facecolor='white', framealpha=0.92, edgecolor='#CFD8DC')
        
    plt.suptitle(
        f"Prognostic Calibration of Epigenetic Base Models at {horizon_label}\n"
        f"Direct Comparison of CpG-Only vs. CpG + Clinical Models in FHS Test and External WHI Cohorts",
        fontsize=13.5, fontweight='bold', y=0.992
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    out_png = FIG_DIR / f"{filename_base}.png"
    out_tif = FIG_DIR / f"{filename_base}.tif"
    out_pdf = FIG_DIR / f"{filename_base}.pdf"
    
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_tif, dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Successfully generated: {out_png}")
    print(f"Successfully generated: {out_tif}")
    print(f"Successfully generated: {out_pdf}")
    
    return pd.concat(decile_dfs, ignore_index=True), pd.DataFrame(stat_records)

# Execute for 14-Year End Point
df_dec_14, df_stat_14 = generate_calibration_figure(14.0, "End of Follow-Up (14-Year Horizon)", "calibration_base_models_14y")

# Execute for 10-Year Horizon
df_dec_10, df_stat_10 = generate_calibration_figure(10.0, "Conventional 10-Year Horizon", "calibration_base_models_10y")

# Save Combined Outputs
pd.concat([df_dec_14, df_dec_10], ignore_index=True).to_csv(FIG_DIR / "calibration_base_models_deciles.csv", index=False)
pd.concat([df_stat_14, df_stat_10], ignore_index=True).to_csv(FIG_DIR / "calibration_base_models_summary.csv", index=False)
print("\nCompleted base models calibration analysis!")
