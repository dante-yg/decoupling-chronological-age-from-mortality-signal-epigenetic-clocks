"""
Calibration Plot for Full BAA23 Methylation Cohort (N=1,818) at 14-Year Horizon
Evaluates the CpG-only CoxNet model (10 CpGs) on all 1,818 BAA23 participants.
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



from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import joblib

BASE_DIR = config.PROJECT_ROOT
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Model, Scaler, and FHS Train Baseline Hazard
scaler = joblib.load(BASE_DIR / "scalers/coxnet_scaler_V3_OPTUNA_SS_CPG.joblib")
cpg_model = joblib.load(BASE_DIR / "saved_models/coxnet_V3_OPTUNA_SS_CPG/coxnet_V3_OPTUNA_SS_CPG.model")
fhs_train = pd.read_csv(BASE_DIR / "predictions/coxnet_train_V3_OPTUNA_SS_CPG.csv")
p10 = pd.read_csv(config.DATA_DIR / "WHI_10_probes.csv" if (config.DATA_DIR / "WHI_10_probes.csv").exists() else config.RAW_DIR / "WHI_10_probes.csv")

features = ['cg04604946', 'cg06639320', 'cg07955995', 'cg10246448', 'cg10501210',
            'cg11617964', 'cg16867657', 'cg20591472', 'cg24724428', 'cg25590826']
X_scaled = scaler.transform(p10[features])
p10['risk_cpg'] = cpg_model.predict(X_scaled)

def compute_breslow(times, events, lp):
    w = np.exp(lp)
    order = np.argsort(times)
    t_s, e_s, w_s = times[order], events[order], w[order]
    r_sum = np.cumsum(w_s[::-1])[::-1]
    ev_mask = (e_s == 1)
    dh = np.zeros_like(r_sum, dtype=float)
    dh[ev_mask] = 1.0 / r_sum[ev_mask]
    return t_s[ev_mask], np.cumsum(dh[ev_mask])

t_ev_tr, H0_cum_tr = compute_breslow(fhs_train['time'].values, fhs_train['event'].values.astype(int), fhs_train['risk_score'].values)
idx = np.searchsorted(t_ev_tr, 14.0, side='right') - 1
h0_14 = H0_cum_tr[idx]

p_pred = 1.0 - np.exp(-h0_14 * np.exp(p10['risk_cpg'].values))
p10['p_14'] = p_pred
p10['decile'] = pd.qcut(p10['p_14'], q=10, labels=False) + 1

rows = []
for d in range(1, 11):
    grp = p10[p10['decile'] == d]
    km = KaplanMeierFitter().fit(grp['time'], grp['event'].astype(int))
    obs = 1.0 - float(km.predict(14.0))
    ci = km.confidence_interval_
    sub_ci = ci.index[ci.index <= 14.0]
    last_t = sub_ci[-1]
    lo = max(0.0, 1.0 - float(ci.loc[last_t].iloc[1]))
    hi = min(1.0, 1.0 - float(ci.loc[last_t].iloc[0]))
    ev = int(((grp['time'] <= 14.0) & (grp['event'] == 1)).sum())
    rows.append({
        'decile': d,
        'n': len(grp),
        'events': ev,
        'mean_pred': grp['p_14'].mean(),
        'obs': obs,
        'lo': lo,
        'hi': hi
    })

res_df = pd.DataFrame(rows)
slope, intercept = np.polyfit(res_df['mean_pred'], res_df['obs'], 1)
r = np.corrcoef(res_df['mean_pred'], res_df['obs'])[0, 1]

km_all = KaplanMeierFitter().fit(p10['time'], p10['event'].astype(int))
overall_obs = 1.0 - float(km_all.predict(14.0))
overall_pred = p_pred.mean()
total_events = int(((p10['time'] <= 14.0) & (p10['event'] == 1)).sum())

# Plot
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

axis_lims = [-0.02, 0.42]
ax.plot([0, 1], [0, 1], linestyle='--', color='#777777', linewidth=1.5, zorder=1, label='Ideal (y = x)')

x_line = np.linspace(0, 0.45, 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, linestyle='-', color='#0072B2', alpha=0.8, linewidth=2.0, zorder=2,
        label=f"Fit: y = {slope:.2f}x + {intercept:.3f} (r = {r:.2f})")

y_err_lower = np.maximum(0, res_df['obs'] - res_df['lo'])
y_err_upper = np.maximum(0, res_df['hi'] - res_df['obs'])

ax.errorbar(
    res_df['mean_pred'],
    res_df['obs'],
    yerr=[y_err_lower, y_err_upper],
    fmt='o',
    color='#0072B2',
    ecolor='#0072B2',
    elinewidth=1.5,
    capsize=4.0,
    capthick=1.5,
    markersize=8.0,
    markeredgecolor='white',
    markeredgewidth=1.0,
    zorder=4,
    label=f'Deciles (N = 1,818, Events = {total_events})'
)

ax.set_title("WHI-BAA23 Epigenome Cohort (N = 1,818)\n10-CpG CoxNet Model Calibration at 14-Year Horizon",
             fontsize=12, fontweight='bold', pad=12, loc='left')
ax.set_xlabel("Predicted 14-year mortality risk", fontsize=11, labelpad=8)
ax.set_ylabel("Observed 14-year mortality risk (Kaplan-Meier)", fontsize=11, labelpad=8)
ax.set_xlim(axis_lims)
ax.set_ylim(axis_lims)
ax.set_aspect('equal', adjustable='box')

box_text = (
    f"Mean Pred: {overall_pred*100:.1f}%\n"
    f"Observed:   {overall_obs*100:.1f}%\n"
    f"Slope:      {slope:.2f}\n"
    f"Intercept:  {intercept:+.3f}\n"
    f"Corr (r):   {r:.2f}"
)
ax.text(
    0.04, 0.96, box_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#D0D5DD', alpha=0.95),
    family='monospace'
)

ax.grid(True, linestyle=':', alpha=0.5, color='#B0BEC5')
ax.legend(loc='lower right', fontsize=9.5, frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
out_png = FIG_DIR / "whi_baa23_1818_calibration_14y.png"
fig.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_png}")
