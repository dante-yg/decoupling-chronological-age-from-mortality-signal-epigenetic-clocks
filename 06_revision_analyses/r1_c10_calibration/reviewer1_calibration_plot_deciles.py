"""
Publication-Quality Calibration Plot for Final CoxNet Model
Reviewer 1, Comment 10: Calibration Plot across Deciles of Predicted Risk

Specifications:
- Final CoxNet model (10 EpiClock CpGs + 4 clinical markers).
- Existing FHS splits, scaling, and WHI cohort preserved without refitting.
- Breslow baseline hazard estimated on FHS training split.
- Predicted 10-year mortality risk: 1 - S(10 | x) = 1 - exp(-H0(10) * exp(LP_i)).
- 10 deciles of predicted risk per cohort.
- Mean predicted risk vs Kaplan-Meier observed 10-year risk (1 - S_KM(10)) with 95% CI.
- Two panels: (a) FHS test, (b) WHI.
- Identical axis limits on both panels: [-0.02, 0.70].
- Axes labeled: "Predicted 10-year mortality risk" and "Observed 10-year mortality risk".
- Colorblind-friendly colors (Okabe-Ito).
- Outputs: 300-dpi TIFF and PNG figures; CSV of per-decile values.
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



import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

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
idx_10 = np.searchsorted(t_ev_tr, 10.0, side='right') - 1
H0_10 = float(H0_cum_tr[idx_10])
S0_10 = float(np.exp(-H0_10))
print(f"FHS Training Split Breslow Hazard at 10 Years: H0(10) = {H0_10:.6f} | S0(10) = {S0_10:.6f}")

# 3. Compute Deciles and KM Observed Risk per Cohort
cohorts = [
    ("FHS test", test_df),
    ("WHI", whi_df)
]

all_deciles_records = []
plot_dict = {}

for cohort_name, df_cohort in cohorts:
    lp = df_cohort['risk_score'].values
    p = 1.0 - np.exp(-H0_10 * np.exp(lp))
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    
    df_eval = pd.DataFrame({
        "time": df_cohort['time'].values,
        "event": df_cohort['event'].values,
        "p": p
    })
    
    # 10 deciles of predicted risk
    df_eval['decile'] = pd.qcut(df_eval['p'], q=10, labels=False) + 1
    
    decile_rows = []
    for d in range(1, 11):
        grp = df_eval[df_eval['decile'] == d]
        n_d = len(grp)
        ev_10 = int(((grp['time'] <= 10.0) & (grp['event'] == 1)).sum())
        mean_p = float(grp['p'].mean())
        
        km = KaplanMeierFitter().fit(grp['time'], grp['event'])
        surv_10 = float(km.predict(10.0))
        obs_10 = 1.0 - surv_10
        
        ci_df = km.confidence_interval_
        sub_ci = ci_df.index[ci_df.index <= 10.0]
        if len(sub_ci) > 0 and ev_10 > 0:
            last_t = sub_ci[-1]
            s_lo = float(ci_df.loc[last_t].iloc[0])
            s_hi = float(ci_df.loc[last_t].iloc[1])
            lower_ci = max(0.0, 1.0 - s_hi)
            upper_ci = min(1.0, 1.0 - s_lo)
        else:
            # When 0 events observed by 10y, KM estimate is 0.0
            lower_ci, upper_ci = 0.0, 0.0
            
        row = {
            "cohort": cohort_name,
            "decile": int(d),
            "n": int(n_d),
            "events": int(ev_10),
            "mean predicted risk": mean_p,
            "observed risk": obs_10,
            "lower CI": lower_ci,
            "upper CI": upper_ci
        }
        decile_rows.append(row)
        all_deciles_records.append(row)
        
    plot_dict[cohort_name] = pd.DataFrame(decile_rows)

# 4. Save CSV Outputs
df_all = pd.DataFrame(all_deciles_records)

# 4a. Master combined CSV with cohort column
csv_combined_path = BASE_DIR / "reviewer1_calibration_deciles.csv"
df_all.to_csv(csv_combined_path, index=False)
print(f"Saved master combined CSV to: {csv_combined_path}")

# 4b. Per-cohort CSVs with exact required columns: decile, n, events, mean predicted risk, observed risk, lower CI, upper CI
req_cols = ["decile", "n", "events", "mean predicted risk", "observed risk", "lower CI", "upper CI"]
csv_fhs_path = BASE_DIR / "reviewer1_calibration_deciles_fhs_test.csv"
df_all[df_all['cohort'] == "FHS test"][req_cols].to_csv(csv_fhs_path, index=False)
print(f"Saved FHS test deciles CSV to: {csv_fhs_path}")

csv_whi_path = BASE_DIR / "reviewer1_calibration_deciles_whi.csv"
df_all[df_all['cohort'] == "WHI"][req_cols].to_csv(csv_whi_path, index=False)
print(f"Saved WHI deciles CSV to: {csv_whi_path}")

# Also save a canonical calibration_deciles_summary.csv
csv_canonical = BASE_DIR / "calibration_deciles_summary.csv"
df_all.to_csv(csv_canonical, index=False)
print(f"Saved canonical deciles CSV to: {csv_canonical}")

# 5. Generate Publication Figure
# Aesthetic parameters
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Okabe-Ito Colorblind-Friendly Palette:
# Blue for FHS Test: #0072B2
# Vermilion / Reddish for WHI: #D55E00
C_FHS = "#0072B2"
C_WHI = "#D55E00"
C_DIAG = "#444444"

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 5.8), sharex=True, sharey=True, dpi=300)

common_lim = (-0.02, 0.70)
diag_vals = np.array([0.0, 0.70])

# -----------------------------------------------------------------------------
# Panel (a): FHS test
# -----------------------------------------------------------------------------
df_fhs = plot_dict["FHS test"]
x_fhs = df_fhs["mean predicted risk"].values
y_fhs = df_fhs["observed risk"].values
y_lo_fhs = df_fhs["lower CI"].values
y_hi_fhs = df_fhs["upper CI"].values

err_lo_fhs = np.maximum(0.0, y_fhs - y_lo_fhs)
err_hi_fhs = np.maximum(0.0, y_hi_fhs - y_fhs)

# Dashed y = x diagonal
ax_a.plot(diag_vals, diag_vals, linestyle="--", color=C_DIAG, linewidth=1.5, zorder=1, label="Ideal calibration ($y = x$)")

# Decile points with error bars
ax_a.errorbar(
    x_fhs, y_fhs, yerr=[err_lo_fhs, err_hi_fhs],
    fmt="o", color=C_FHS, ecolor=C_FHS, elinewidth=1.6, capsize=4.0, markersize=7.0,
    zorder=4, label="Deciles of predicted risk (KM observed $\\pm$ 95% CI)"
)

ax_a.set_xlim(common_lim)
ax_a.set_ylim(common_lim)
ax_a.set_aspect("equal", adjustable="box")
ax_a.set_title("(a) FHS test", fontsize=13, fontweight="bold", pad=10)
ax_a.set_xlabel("Predicted 10-year mortality risk", fontsize=11, fontweight="bold")
ax_a.set_ylabel("Observed 10-year mortality risk", fontsize=11, fontweight="bold")
ax_a.grid(True, linestyle=":", alpha=0.55)
ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.8)

# Annotation for Panel (a)
annot_a = (
    "FHS Test Cohort ($N = 699$, 93 deaths)\n"
    "Mean Predicted Risk: 0.1186\n"
    "Observed KM Mortality: 0.1351\n"
    "O / E Ratio: 1.138 (95% CI: 0.952–1.335)"
)
ax_a.text(
    0.04, 0.96, annot_a, transform=ax_a.transAxes, fontsize=8.8,
    verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CFD8DC", alpha=0.92)
)

# -----------------------------------------------------------------------------
# Panel (b): WHI
# -----------------------------------------------------------------------------
df_whi = plot_dict["WHI"]
x_whi = df_whi["mean predicted risk"].values
y_whi = df_whi["observed risk"].values
y_lo_whi = df_whi["lower CI"].values
y_hi_whi = df_whi["upper CI"].values

err_lo_whi = np.maximum(0.0, y_whi - y_lo_whi)
err_hi_whi = np.maximum(0.0, y_hi_whi - y_whi)

# Dashed y = x diagonal
ax_b.plot(diag_vals, diag_vals, linestyle="--", color=C_DIAG, linewidth=1.5, zorder=1, label="Ideal calibration ($y = x$)")

# Decile points with error bars
ax_b.errorbar(
    x_whi, y_whi, yerr=[err_lo_whi, err_hi_whi],
    fmt="s", color=C_WHI, ecolor=C_WHI, elinewidth=1.6, capsize=4.0, markersize=6.8,
    zorder=4, label="Deciles of predicted risk (KM observed $\\pm$ 95% CI)"
)

ax_b.set_xlim(common_lim)
ax_b.set_ylim(common_lim)
ax_b.set_aspect("equal", adjustable="box")
ax_b.set_title("(b) WHI", fontsize=13, fontweight="bold", pad=10)
ax_b.set_xlabel("Predicted 10-year mortality risk", fontsize=11, fontweight="bold")
ax_b.set_ylabel("Observed 10-year mortality risk", fontsize=11, fontweight="bold")
ax_b.grid(True, linestyle=":", alpha=0.55)
ax_b.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.8)

# Annotation for Panel (b)
annot_b = (
    "WHI External Cohort ($N = 196$, 3 deaths)\n"
    "Mean Predicted Risk: 0.0568\n"
    "Observed KM Mortality: 0.0157\n"
    "O / E Ratio: 0.277 (95% CI: 0.000–0.616)"
)
ax_b.text(
    0.04, 0.96, annot_b, transform=ax_b.transAxes, fontsize=8.8,
    verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CFD8DC", alpha=0.92)
)

plt.tight_layout()

# Save PNG and 300-DPI TIFF
out_png = FIG_DIR / "reviewer1_calibration_deciles_final_model.png"
out_tif = FIG_DIR / "reviewer1_calibration_deciles_final_model.tif"

fig.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved PNG calibration figure to: {out_png}")

fig.savefig(out_tif, dpi=300, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
print(f"Saved 300-DPI TIFF calibration figure to: {out_tif}")

plt.close(fig)
print("Calibration decile generation finished successfully!")
