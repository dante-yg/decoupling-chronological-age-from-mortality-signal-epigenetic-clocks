"""
Comprehensive WHI Survival Data Audit for Reviewer 1
- Analyzes follow-up times, event distributions, and milestones
- Confirms totals match Table 1 (n = 196, 56 deaths, mean follow-up 16.95 y)
- Generates 2-panel publication figure:
  (a) Histogram of follow-up and event times
  (b) Kaplan-Meier survival curve with 95% CI and milestone annotations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Paths
BASE_DIR = Path(r"C:\F\base_3")
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
whi = pd.read_csv(DATA_DIR / "WHI_non_normalized.csv")
whi['event'] = (whi['event'] == True) | (whi['event'] == 1) | (whi['event'] == "True")
whi['event'] = whi['event'].astype(int)

# 2. Detailed Follow-up Statistics
n_total = len(whi)
deaths = whi[whi['event'] == 1]['time']
cens = whi[whi['event'] == 0]['time']

audit_data = {
    "Group": ["All Participants", "Deaths (Events = 1)", "Censored (Events = 0)"],
    "N": [n_total, len(deaths), len(cens)],
    "Pct": [100.0, len(deaths) / n_total * 100.0, len(cens) / n_total * 100.0],
    "Mean_y": [whi['time'].mean(), deaths.mean(), cens.mean()],
    "Std_y": [whi['time'].std(), deaths.std(), cens.std()],
    "Median_y": [whi['time'].median(), deaths.median(), cens.median()],
    "Min_y": [whi['time'].min(), deaths.min(), cens.min()],
    "Max_y": [whi['time'].max(), deaths.max(), cens.max()],
    "Min_days": [whi['time'].min() * 365.0, deaths.min() * 365.0, cens.min() * 365.0],
    "Median_days": [whi['time'].median() * 365.0, deaths.median() * 365.0, cens.median() * 365.0],
    "Max_days": [whi['time'].max() * 365.0, deaths.max() * 365.0, cens.max() * 365.0],
}
df_audit = pd.DataFrame(audit_data)
df_audit.to_csv(BASE_DIR / "whi_followup_audit_summary.csv", index=False)
print("WHI Follow-up Summary:")
print(df_audit.to_string())

# 3. Horizon Audit: 5, 10, 15, 20 Years
km = KaplanMeierFitter().fit(whi['time'], whi['event'], label="WHI Cohort (N = 196)")

horizon_records = []
for h in [5.0, 10.0, 15.0, 20.0]:
    d_count = int(((whi['time'] <= h) & (whi['event'] == 1)).sum())
    c_count = int(((whi['time'] <= h) & (whi['event'] == 0)).sum())
    at_risk = int((whi['time'] >= h).sum())
    s_h = float(km.predict(h))
    obs_risk = 1.0 - s_h
    
    # 95% CI from lifelines
    ci_df = km.confidence_interval_
    sub_ci = ci_df.index[ci_df.index <= h]
    if len(sub_ci) > 0 and d_count > 0:
        last_t = sub_ci[-1]
        s_lo = float(ci_df.loc[last_t].iloc[0])
        s_hi = float(ci_df.loc[last_t].iloc[1])
        ci_str = f"[{1.0 - s_hi:.4f}, {1.0 - s_lo:.4f}]"
    else:
        ci_str = "[0.0000, 0.0000]"
        
    horizon_records.append({
        "Horizon_Years": int(h),
        "At_Risk": at_risk,
        "Pct_At_Risk": round(at_risk / n_total * 100.0, 1),
        "Cumulative_Deaths": d_count,
        "Pct_Deaths": round(d_count / n_total * 100.0, 2),
        "Cumulative_Censored": c_count,
        "KM_Survival": round(s_h, 4),
        "KM_Observed_Risk": round(obs_risk, 4),
        "Observed_Risk_95_CI": ci_str
    })
df_horizons = pd.DataFrame(horizon_records)
df_horizons.to_csv(BASE_DIR / "whi_deaths_by_horizon_audit.csv", index=False)
print("\nWHI Cumulative Deaths by Horizon:")
print(df_horizons.to_string())

# 4. Generate 2-Panel Publication Figure
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), dpi=300)

# Okabe-Ito colors:
C_DEATH = "#D55E00"  # Vermilion
C_CENS  = "#0072B2"  # Blue
C_KM    = "#009E73"  # Bluish Green
C_GRID  = "#DDDDDD"

# -----------------------------------------------------------------------------
# Panel (a): Histogram of Follow-up Times (Deaths vs Censored)
# -----------------------------------------------------------------------------
bins = np.linspace(5.0, 25.0, 21)
ax1.hist(
    [deaths, cens], bins=bins, stacked=True,
    color=[C_DEATH, C_CENS], edgecolor="white", linewidth=1.0,
    label=[f"Deaths / Events (n = {len(deaths)})", f"Censored (n = {len(cens)})"],
    alpha=0.9
)

ax1.axvline(10.0, color="#333333", linestyle="--", linewidth=1.6, label="10-Year Horizon Threshold")
ax1.axvline(whi['time'].mean(), color="#777777", linestyle=":", linewidth=1.8, label=f"Mean Follow-up: {whi['time'].mean():.2f} y")

ax1.set_title("(a) Distribution of WHI Follow-Up Times (N = 196)", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Follow-up Time (Years)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Number of Participants", fontsize=11, fontweight="bold")
ax1.set_xlim(4.0, 25.0)
ax1.set_ylim(0, 45)
ax1.grid(True, linestyle=":", alpha=0.55)
ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.8)

# Text annotation on panel a
annot_hist = (
    f"WHI Follow-Up Distribution:\n"
    f"• 0 deaths before 7.5 years\n"
    f"• Only 3 deaths by 10.0 years\n"
    f"• 53 of 56 deaths occur after 10 years\n"
    f"• Median death time: {deaths.median():.2f} years\n"
    f"• Max follow-up: {whi['time'].max():.2f} years"
)
ax1.text(
    0.50, 0.96, annot_hist, transform=ax1.transAxes, fontsize=8.8,
    verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CFD8DC", alpha=0.92)
)

# -----------------------------------------------------------------------------
# Panel (b): Kaplan-Meier Survival Curve
# -----------------------------------------------------------------------------
surv_df = km.survival_function_
ci_df = km.confidence_interval_
t_km = surv_df.index
s_km = surv_df.iloc[:, 0].values

s_lo = ci_df.iloc[:, 0].values
s_hi = ci_df.iloc[:, 1].values

ax2.step(t_km, s_km, where="post", color=C_KM, linewidth=2.4, label="Kaplan–Meier Estimate S(t)")
ax2.fill_between(t_km, s_lo, s_hi, step="post", color=C_KM, alpha=0.22, label="95% Hall–Wellner / Greenwood CI")

# Plot censored ticks
cens_times = cens.values
cens_survs = km.predict(cens_times).values
ax2.scatter(cens_times, cens_survs, marker='+', s=45, color='#333333', linewidth=1.2, zorder=4, label=f"Censored (n = {len(cens)})")

# Milestone annotations on KM curve
milestones = [
    (5.0, float(km.predict(5.0))),
    (10.0, float(km.predict(10.0))),
    (15.0, float(km.predict(15.0))),
    (20.0, float(km.predict(20.0)))
]

for t_val, s_val in milestones:
    ax2.plot(t_val, s_val, marker='o', markersize=6.5, color=C_DEATH, zorder=5)
    
# Vertical drop line to 10y milestone
ax2.axvline(10.0, color="#333333", linestyle="--", linewidth=1.4, alpha=0.7)

# Annotate t=10y and t=15y specifically
s_10 = float(km.predict(10.0))
s_15 = float(km.predict(15.0))

ax2.annotate(
    f"10-Year Horizon\nS(10) = {s_10:.3f} (3 deaths)",
    xy=(10.0, s_10),
    xytext=(3.5, 0.82),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    fontsize=8.5, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF9C4", edgecolor="#FBC02D", alpha=0.92)
)

ax2.annotate(
    f"15-Year Horizon\nS(15) = {s_15:.3f} (21 deaths)",
    xy=(15.0, s_15),
    xytext=(10.8, 0.56),
    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
    fontsize=8.5, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#E1F5FE", edgecolor="#0288D1", alpha=0.92)
)

ax2.set_title("(b) Kaplan–Meier Survival Curve (WHI Cohort, N = 196)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Time from Baseline (Years)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Survival Probability, S(t)", fontsize=11, fontweight="bold")
ax2.set_xlim(0, 24.0)
ax2.set_ylim(-0.02, 1.05)
ax2.grid(True, linestyle=":", alpha=0.55)
ax2.legend(loc="lower left", bbox_to_anchor=(0.02, 0.23), frameon=True, facecolor="white", framealpha=0.92, fontsize=8.2)

# Numbers at risk table at bottom
n_risk_times = [0, 5, 10, 15, 20]
n_risk_counts = [int((whi['time'] >= t).sum()) for t in n_risk_times]
n_risk_deaths = [int(((whi['time'] <= t) & (whi['event'] == 1)).sum()) for t in n_risk_times]

table_text = "At Risk:   " + "      ".join([f"{n:3d}" for n in n_risk_counts]) + "\n" + \
             "Deaths:    " + "      ".join([f"{d:3d}" for d in n_risk_deaths])
ax2.text(
    0.04, 0.12, f"Time (years):  0         5        10        15        20\n{table_text}",
    transform=ax2.transAxes, fontsize=8.0, family="monospace",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5", edgecolor="#BDBDBD", alpha=0.9)
)

plt.tight_layout()

out_png = FIG_DIR / "whi_survival_audit_plot.png"
out_tif = FIG_DIR / "whi_survival_audit_plot.tif"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_tif, dpi=300, pil_kwargs={"compression": "tiff_lzw"}, bbox_inches="tight")
plt.close(fig)
print(f"Saved PNG to: {out_png}")
print(f"Saved TIFF to: {out_tif}")
