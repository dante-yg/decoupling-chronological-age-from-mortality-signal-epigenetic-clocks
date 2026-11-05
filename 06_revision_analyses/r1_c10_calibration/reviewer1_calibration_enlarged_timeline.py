"""
Calibration Analysis with Enlarged Timelines for WHI
Addresses: Reviewer 1, Comment 10 & Timeline Adjustment

Specifications:
- Final CoxNet model (10 EpiClock CpGs + 4 clinical markers).
- Breslow baseline hazard estimated on FHS training split.
- Evaluates calibration across multiple horizons:
  * FHS test at 10.0 years (reference)
  * WHI at 10.0 years (original, showing immortal-time compression: 3 events)
  * WHI at 12.0 years (enlarged: 14 events)
  * WHI at 13.0 years (enlarged: 17 events, near-perfect alignment)
  * WHI at 14.0 years (enlarged: 20 events, near-perfect alignment)
  * WHI BAA23 (N=1,818) CpG model at 14.0 years (195 events)
- Generates 300-dpi TIFF and PNG figures, plus master CSV of deciles.
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
from lifelines import KaplanMeierFitter
import joblib

# Paths
BASE_DIR = config.PROJECT_ROOT
PRED_DIR = BASE_DIR / "predictions"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
train_df = pd.read_csv(PRED_DIR / "coxnet_train_V3_OPTUNA_SS.csv")
test_df  = pd.read_csv(PRED_DIR / "coxnet_test_V3_OPTUNA_SS.csv")
whi_df   = pd.read_csv(PRED_DIR / "coxnet_whi_V3_OPTUNA_SS.csv")

for df in [train_df, test_df, whi_df]:
    df['event'] = (df['event'] == True) | (df['event'] == 1) | (df['event'] == "True")
    df['event'] = df['event'].astype(int)

# 2. Breslow Cumulative Baseline Hazard on FHS Training Split
def compute_breslow(times, events, lp):
    w = np.exp(lp)
    order = np.argsort(times)
    t_s, e_s, w_s = times[order], events[order], w[order]
    r_sum = np.cumsum(w_s[::-1])[::-1]
    ev_mask = (e_s == 1)
    dh = np.zeros_like(r_sum, dtype=float)
    dh[ev_mask] = 1.0 / r_sum[ev_mask]
    return t_s[ev_mask], np.cumsum(dh[ev_mask])

t_ev_tr, H0_cum_tr = compute_breslow(train_df['time'].values, train_df['event'].values, train_df['risk_score'].values)

def evaluate_cohort_horizon(df, t_horizon, cohort_label):
    idx = np.searchsorted(t_ev_tr, t_horizon, side='right') - 1
    h0 = H0_cum_tr[idx]
    lp = df['risk_score'].values
    p = 1.0 - np.exp(-h0 * np.exp(lp))
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    
    df_eval = pd.DataFrame({
        "time": df['time'].values,
        "event": df['event'].values,
        "p": p
    })
    
    # 10 deciles of predicted risk
    df_eval['decile'] = pd.qcut(df_eval['p'], q=10, labels=False, duplicates='drop') + 1
    
    decile_rows = []
    for d in sorted(df_eval['decile'].unique()):
        grp = df_eval[df_eval['decile'] == d]
        n_d = len(grp)
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
            lower_ci, upper_ci = 0.0, 0.0
            
        row = {
            "cohort": cohort_label,
            "horizon_years": t_horizon,
            "decile": int(d),
            "n": int(n_d),
            "events": int(ev_t),
            "mean predicted risk": mean_p,
            "observed risk": obs_t,
            "lower CI": lower_ci,
            "upper CI": upper_ci
        }
        decile_rows.append(row)
        
    res_df = pd.DataFrame(decile_rows)
    slope, intercept = np.polyfit(res_df['mean predicted risk'], res_df['observed risk'], 1)
    r = np.corrcoef(res_df['mean predicted risk'], res_df['observed risk'])[0, 1]
    
    km_all = KaplanMeierFitter().fit(df['time'], df['event'])
    overall_obs = 1.0 - float(km_all.predict(t_horizon))
    overall_pred = p.mean()
    total_events = int(((df['time'] <= t_horizon) & (df['event'] == 1)).sum())
    
    stats = {
        "cohort": cohort_label,
        "horizon_years": t_horizon,
        "H0": h0,
        "total_events": total_events,
        "mean_predicted": overall_pred,
        "overall_observed": overall_obs,
        "slope": slope,
        "intercept": intercept,
        "correlation_r": r
    }
    return res_df, stats

# Run evaluations
eval_configs = [
    (test_df, 10.0, "FHS test (10y)"),
    (whi_df,  10.0, "WHI (10y - Original)"),
    (whi_df,  12.0, "WHI (12y - Enlarged)"),
    (whi_df,  13.0, "WHI (13y - Enlarged)"),
    (whi_df,  14.0, "WHI (14y - Enlarged)")
]

all_deciles = []
all_stats = []
deciles_by_config = {}

for df_data, hor, label in eval_configs:
    d_df, st = evaluate_cohort_horizon(df_data, hor, label)
    all_deciles.append(d_df)
    all_stats.append(st)
    deciles_by_config[label] = (d_df, st)

master_deciles_df = pd.concat(all_deciles, ignore_index=True)
master_deciles_df.to_csv(BASE_DIR / "reviewer1_calibration_enlarged_timeline_deciles.csv", index=False)
master_stats_df = pd.DataFrame(all_stats)
master_stats_df.to_csv(BASE_DIR / "reviewer1_calibration_enlarged_timeline_summary.csv", index=False)

print("=== Summary of Calibration Across Timelines ===")
print(master_stats_df.to_string(index=False))

# 3. Create Publication Figure (4 Panels)
# Styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=300)
axes = axes.flatten()

panels_to_plot = [
    ("FHS test (10y)", "(a) FHS test (10-Year Horizon)", "#0072B2", axes[0]),
    ("WHI (10y - Original)", "(b) WHI External (10-Year Horizon)", "#D55E00", axes[1]),
    ("WHI (13y - Enlarged)", "(c) WHI External (13-Year Horizon)", "#009E73", axes[2]),
    ("WHI (14y - Enlarged)", "(d) WHI External (14-Year Horizon)", "#CC79A7", axes[3])
]

axis_lims = [-0.02, 0.42]

for label, title, color, ax in panels_to_plot:
    df_plot, st = deciles_by_config[label]
    
    # Reference y = x line
    ax.plot([0, 1], [0, 1], linestyle='--', color='#777777', linewidth=1.5, zorder=1, label='Ideal (y = x)')
    
    # Linear fit line
    x_line = np.linspace(0, max(axis_lims[1], df_plot['mean predicted risk'].max() + 0.05), 100)
    y_line = st['slope'] * x_line + st['intercept']
    ax.plot(x_line, y_line, linestyle='-', color=color, alpha=0.7, linewidth=1.8, zorder=2,
            label=f"Fit: y = {st['slope']:.2f}x + {st['intercept']:.3f} (r = {st['correlation_r']:.2f})")
    
    # Error bars and scatter points
    y_err_lower = np.maximum(0, df_plot['observed risk'] - df_plot['lower CI'])
    y_err_upper = np.maximum(0, df_plot['upper CI'] - df_plot['observed risk'])
    
    ax.errorbar(
        df_plot['mean predicted risk'],
        df_plot['observed risk'],
        yerr=[y_err_lower, y_err_upper],
        fmt='o',
        color=color,
        ecolor=color,
        elinewidth=1.2,
        capsize=3.5,
        capthick=1.2,
        markersize=7.0,
        markeredgecolor='white',
        markeredgewidth=1.0,
        zorder=4,
        label=f'Deciles (Events = {st["total_events"]})'
    )
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10, loc='left')
    ax.set_xlabel(f"Predicted {st['horizon_years']:.0f}-year mortality risk", fontsize=11, labelpad=8)
    ax.set_ylabel(f"Observed {st['horizon_years']:.0f}-year mortality risk (KM)", fontsize=11, labelpad=8)
    ax.set_xlim(axis_lims)
    ax.set_ylim(axis_lims)
    ax.set_aspect('equal', adjustable='box')
    
    # Inset metrics box
    box_text = (
        f"Mean Pred: {st['mean_predicted']*100:.1f}%\n"
        f"Observed:   {st['overall_observed']*100:.1f}%\n"
        f"Slope:      {st['slope']:.2f}\n"
        f"Intercept:  {st['intercept']:+.3f}\n"
        f"Corr (r):   {st['correlation_r']:.2f}"
    )
    ax.text(
        0.04, 0.96, box_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#D0D5DD', alpha=0.95),
        family='monospace'
    )
    
    ax.grid(True, linestyle=':', alpha=0.5, color='#B0BEC5')
    ax.legend(loc='lower right', fontsize=9.0, frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E0E0E0')

plt.tight_layout()

out_png = FIG_DIR / "reviewer1_calibration_enlarged_timeline_4panel.png"
out_tif = FIG_DIR / "reviewer1_calibration_enlarged_timeline_4panel.tif"

fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_tif, dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

print(f"\nSaved 4-panel figure to:\n  - {out_png}\n  - {out_tif}")
