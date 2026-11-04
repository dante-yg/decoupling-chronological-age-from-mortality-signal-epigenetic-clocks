"""
Script to generate side-by-side calibration curves:
'Data We Have' (WHI N = 196) vs 'All The Data' (WHI-BAA23 N = 1,818)

Produces:
1. figures/whi_calibration_196_vs_1818_side_by_side_14y.png & .tif (300 dpi):
   - Panel (a): Data We Have (N = 196, Final CoxNet 14-feature model, Events = 20)
   - Panel (b): All Available Data (N = 1,818, CpG CoxNet 10-probe model, Events = 195)
2. figures/whi_calibration_cpg_196_vs_1818_side_by_side_14y.png & .tif (300 dpi):
   - Panel (a): Data We Have (N = 196, CpG CoxNet model, Events = 20)
   - Panel (b): All Available Data (N = 1,818, CpG CoxNet model, Events = 195)
3. CSV tables of deciles and calibration summary.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import joblib

# Paths
BASE_DIR = Path(r"C:\F\base_3")
PRED_DIR = BASE_DIR / "predictions"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
whi196 = pd.read_csv(BASE_DIR / "data/WHI_non_normalized.csv")
p10_all = pd.read_csv(r"C:\Neural_network_survival_WHI\WHI_10_probes.csv")
pred_final_196 = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv")

# Ensure integer events
whi196['event'] = (whi196['event'] == True) | (whi196['event'] == 1) | (whi196['event'] == "True")
whi196['event'] = whi196['event'].astype(int)
p10_all['event'] = (p10_all['event'] == True) | (p10_all['event'] == 1) | (p10_all['event'] == "True")
p10_all['event'] = p10_all['event'].astype(int)

whi196['risk_final'] = pred_final_196['risk_score'].values

# Load models and scalers
scaler_cpg = joblib.load(BASE_DIR / "scalers/coxnet_scaler_V3_OPTUNA_SS_CPG.joblib")
model_cpg = joblib.load(BASE_DIR / "saved_models/coxnet_V3_OPTUNA_SS_CPG/coxnet_V3_OPTUNA_SS_CPG.model")

features = ['cg04604946', 'cg06639320', 'cg07955995', 'cg10246448', 'cg10501210',
            'cg11617964', 'cg16867657', 'cg20591472', 'cg24724428', 'cg25590826']

whi196['risk_cpg'] = model_cpg.predict(scaler_cpg.transform(whi196[features]))
p10_all['risk_cpg'] = model_cpg.predict(scaler_cpg.transform(p10_all[features]))

# Breslow H0 function
def compute_breslow(times, events, lp):
    w = np.exp(lp)
    order = np.argsort(times)
    t_s, e_s, w_s = times[order], events[order], w[order]
    r_sum = np.cumsum(w_s[::-1])[::-1]
    ev_mask = (e_s == 1)
    dh = np.zeros_like(r_sum, dtype=float)
    dh[ev_mask] = 1.0 / r_sum[ev_mask]
    return t_s[ev_mask], np.cumsum(dh[ev_mask])

# Baseline hazards from FHS train
fhs_train_final = pd.read_csv(PRED_DIR / "coxnet_train_V3_OPTUNA_SS.csv")
fhs_train_final['event'] = fhs_train_final['event'].astype(int)
t_ev_fin, H0_cum_fin = compute_breslow(fhs_train_final['time'].values, fhs_train_final['event'].values, fhs_train_final['risk_score'].values)

fhs_train_cpg = pd.read_csv(PRED_DIR / "coxnet_train_V3_OPTUNA_SS_CPG.csv")
fhs_train_cpg['event'] = fhs_train_cpg['event'].astype(int)
t_ev_cpg, H0_cum_cpg = compute_breslow(fhs_train_cpg['time'].values, fhs_train_cpg['event'].values, fhs_train_cpg['risk_score'].values)

def get_calibration_deciles(df, risk_col, H0_val, t_horizon, cohort_name):
    lp = df[risk_col].values
    p = 1.0 - np.exp(-H0_val * np.exp(lp))
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    
    df_eval = pd.DataFrame({
        "time": df['time'].values,
        "event": df['event'].values,
        "p": p
    })
    
    df_eval['decile'] = pd.qcut(df_eval['p'], q=10, labels=False, duplicates='drop') + 1
    
    rows = []
    for d in sorted(df_eval['decile'].unique()):
        grp = df_eval[df_eval['decile'] == d]
        n_d = len(grp)
        ev_t = int(((grp['time'] <= t_horizon) & (grp['event'] == 1)).sum())
        mean_p = float(grp['p'].mean())
        
        km = KaplanMeierFitter().fit(grp['time'], grp['event'])
        obs_t = 1.0 - float(km.predict(t_horizon))
        
        ci_df = km.confidence_interval_
        sub_ci = ci_df.index[ci_df.index <= t_horizon]
        if len(sub_ci) > 0 and ev_t > 0:
            last_t = sub_ci[-1]
            s_lo = float(ci_df.loc[last_t].iloc[0])
            s_hi = float(ci_df.loc[last_t].iloc[1])
            lower_ci = max(0.0, 1.0 - s_hi)
            upper_ci = min(1.0, 1.0 - s_lo)
        else:
            lower_ci, upper_ci = 0.0, 0.0
            
        rows.append({
            "cohort": cohort_name,
            "horizon_years": t_horizon,
            "decile": int(d),
            "n": int(n_d),
            "events": int(ev_t),
            "mean predicted risk": mean_p,
            "observed risk": obs_t,
            "lower CI": lower_ci,
            "upper CI": upper_ci
        })
        
    res_df = pd.DataFrame(rows)
    slope, intercept = np.polyfit(res_df['mean predicted risk'], res_df['observed risk'], 1)
    r = np.corrcoef(res_df['mean predicted risk'], res_df['observed risk'])[0, 1]
    
    km_all = KaplanMeierFitter().fit(df['time'], df['event'])
    overall_obs = 1.0 - float(km_all.predict(t_horizon))
    overall_pred = p.mean()
    total_events = int(((df['time'] <= t_horizon) & (df['event'] == 1)).sum())
    
    stats = {
        "cohort": cohort_name,
        "horizon_years": t_horizon,
        "total_n": len(df),
        "total_events": total_events,
        "mean_predicted": overall_pred,
        "overall_observed": overall_obs,
        "slope": slope,
        "intercept": intercept,
        "correlation_r": r
    }
    return res_df, stats

# Compute for horizon t = 14.0 years
t_h = 14.0
idx_fin = np.searchsorted(t_ev_fin, t_h, side='right') - 1
h0_fin_14 = H0_cum_fin[idx_fin]

idx_cpg = np.searchsorted(t_ev_cpg, t_h, side='right') - 1
h0_cpg_14 = H0_cum_cpg[idx_cpg]

# 1. WHI 196 Final Model
df_dec_196_fin, stats_196_fin = get_calibration_deciles(
    whi196, 'risk_final', h0_fin_14, t_h, "WHI Validation (N=196, Final 14-Feature Model)"
)

# 2. WHI 196 CpG Model
df_dec_196_cpg, stats_196_cpg = get_calibration_deciles(
    whi196, 'risk_cpg', h0_cpg_14, t_h, "WHI Validation (N=196, CpG Model)"
)

# 3. WHI 1,818 All Data CpG Model
df_dec_all_cpg, stats_all_cpg = get_calibration_deciles(
    p10_all, 'risk_cpg', h0_cpg_14, t_h, "WHI All Data (N=1,818, CpG Model)"
)

# Save master deciles CSV
combined_deciles = pd.concat([df_dec_196_fin, df_dec_196_cpg, df_dec_all_cpg], ignore_index=True)
combined_deciles.to_csv(BASE_DIR / "reviewer1_calibration_side_by_side_deciles.csv", index=False)
summary_stats = pd.DataFrame([stats_196_fin, stats_196_cpg, stats_all_cpg])
summary_stats.to_csv(BASE_DIR / "reviewer1_calibration_side_by_side_summary.csv", index=False)

print("=== Summary Stats for Side-by-Side Calibration (t = 14 Years) ===")
print(summary_stats.to_string(index=False))

# Function to plot 2-panel figure
def plot_side_by_side(panel_left_data, panel_right_data, filename_base, title_main):
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), dpi=300)
    axis_lims = [-0.02, 0.42]

    # Panel Left
    df_l, st_l, color_l, title_l = panel_left_data
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle='--', color='#777777', linewidth=1.5, zorder=1, label='Ideal (y = x)')
    x_l = np.linspace(0, 0.45, 100)
    ax.plot(x_l, st_l['slope'] * x_l + st_l['intercept'], linestyle='-', color=color_l, alpha=0.8, linewidth=1.8, zorder=2,
            label=f"Fit: y = {st_l['slope']:.2f}x + {st_l['intercept']:.3f} (r = {st_l['correlation_r']:.2f})")
    
    y_err_lo = np.maximum(0, df_l['observed risk'] - df_l['lower CI'])
    y_err_hi = np.maximum(0, df_l['upper CI'] - df_l['observed risk'])
    ax.errorbar(
        df_l['mean predicted risk'], df_l['observed risk'],
        yerr=[y_err_lo, y_err_hi],
        fmt='o', color=color_l, ecolor=color_l, elinewidth=1.3, capsize=3.5, capthick=1.3,
        markersize=7.5, markeredgecolor='white', markeredgewidth=1.0, zorder=4,
        label=f"Deciles (n = {st_l['total_n']}, events = {st_l['total_events']})"
    )
    ax.set_title(title_l, fontsize=12.5, fontweight='bold', pad=10, loc='left')
    ax.set_xlabel("Predicted 14-year mortality risk", fontsize=11, labelpad=8)
    ax.set_ylabel("Observed 14-year mortality risk (KM)", fontsize=11, labelpad=8)
    ax.set_xlim(axis_lims)
    ax.set_ylim(axis_lims)
    ax.set_aspect('equal', adjustable='box')
    box_l = (
        f"Mean Pred: {st_l['mean_predicted']*100:.1f}%\n"
        f"Observed:   {st_l['overall_observed']*100:.1f}%\n"
        f"Slope:      {st_l['slope']:.2f}\n"
        f"Intercept:  {st_l['intercept']:+.3f}\n"
        f"Corr (r):   {st_l['correlation_r']:.2f}"
    )
    ax.text(0.04, 0.96, box_l, transform=ax.transAxes, fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#D0D5DD', alpha=0.95), family='monospace')
    ax.grid(True, linestyle=':', alpha=0.5, color='#B0BEC5')
    ax.legend(loc='lower right', fontsize=9.0, frameon=True, facecolor='white', framealpha=0.9)

    # Panel Right
    df_r, st_r, color_r, title_r = panel_right_data
    ax = axes[1]
    ax.plot([0, 1], [0, 1], linestyle='--', color='#777777', linewidth=1.5, zorder=1, label='Ideal (y = x)')
    x_r = np.linspace(0, 0.45, 100)
    ax.plot(x_r, st_r['slope'] * x_r + st_r['intercept'], linestyle='-', color=color_r, alpha=0.8, linewidth=1.8, zorder=2,
            label=f"Fit: y = {st_r['slope']:.2f}x + {st_r['intercept']:.3f} (r = {st_r['correlation_r']:.2f})")
    
    y_err_lo = np.maximum(0, df_r['observed risk'] - df_r['lower CI'])
    y_err_hi = np.maximum(0, df_r['upper CI'] - df_r['observed risk'])
    ax.errorbar(
        df_r['mean predicted risk'], df_r['observed risk'],
        yerr=[y_err_lo, y_err_hi],
        fmt='o', color=color_r, ecolor=color_r, elinewidth=1.3, capsize=3.5, capthick=1.3,
        markersize=7.5, markeredgecolor='white', markeredgewidth=1.0, zorder=4,
        label=f"Deciles (n = {st_r['total_n']}, events = {st_r['total_events']})"
    )
    ax.set_title(title_r, fontsize=12.5, fontweight='bold', pad=10, loc='left')
    ax.set_xlabel("Predicted 14-year mortality risk", fontsize=11, labelpad=8)
    ax.set_ylabel("Observed 14-year mortality risk (KM)", fontsize=11, labelpad=8)
    ax.set_xlim(axis_lims)
    ax.set_ylim(axis_lims)
    ax.set_aspect('equal', adjustable='box')
    box_r = (
        f"Mean Pred: {st_r['mean_predicted']*100:.1f}%\n"
        f"Observed:   {st_r['overall_observed']*100:.1f}%\n"
        f"Slope:      {st_r['slope']:.2f}\n"
        f"Intercept:  {st_r['intercept']:+.3f}\n"
        f"Corr (r):   {st_r['correlation_r']:.2f}"
    )
    ax.text(0.04, 0.96, box_r, transform=ax.transAxes, fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#D0D5DD', alpha=0.95), family='monospace')
    ax.grid(True, linestyle=':', alpha=0.5, color='#B0BEC5')
    ax.legend(loc='lower right', fontsize=9.0, frameon=True, facecolor='white', framealpha=0.9)

    plt.suptitle(title_main, fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_png = FIG_DIR / f"{filename_base}.png"
    out_tif = FIG_DIR / f"{filename_base}.tif"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_tif, dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_tif}")

# Generate Figure 1: Final Model (196) vs All Data (1,818)
plot_side_by_side(
    panel_left_data=(df_dec_196_fin, stats_196_fin, "#D55E00", "(a) Only Data We Have: WHI Validation (N = 196)\nFinal CoxNet Model (10 CpGs + 4 Clinical Markers)"),
    panel_right_data=(df_dec_all_cpg, stats_all_cpg, "#0072B2", "(b) All Data: WHI-BAA23 Epigenome Cohort (N = 1,818)\nCpG-Only CoxNet Model (10 CpGs)"),
    filename_base="whi_calibration_196_vs_1818_side_by_side_14y",
    title_main="External Calibration at Enlarged 14-Year Timeline: Validation Sample vs. Full BAA23 Cohort"
)

# Generate Figure 2: Like-for-Like CpG Model (196) vs CpG Model (1,818)
plot_side_by_side(
    panel_left_data=(df_dec_196_cpg, stats_196_cpg, "#CC79A7", "(a) Only Data We Have: WHI Validation (N = 196)\nCpG-Only CoxNet Model (10 CpGs)"),
    panel_right_data=(df_dec_all_cpg, stats_all_cpg, "#0072B2", "(b) All Data: WHI-BAA23 Epigenome Cohort (N = 1,818)\nCpG-Only CoxNet Model (10 CpGs)"),
    filename_base="whi_calibration_cpg_196_vs_1818_side_by_side_14y",
    title_main="Direct CpG Model Calibration at Enlarged 14-Year Timeline: N = 196 vs. N = 1,818"
)
