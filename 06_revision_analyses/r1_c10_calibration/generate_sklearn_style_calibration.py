"""
Generate Scikit-Learn style calibration curves for FHS Test and WHI cohorts
at the 14-year end point, matching the exact visual style of the reference image:
- Blue solid connected line with circular markers
- Gray dashed diagonal line ("Ideal Calibration")
- 0.0 to 1.0 axes
- "Mean Predicted Probability" vs "Fraction of Positives"
- Clean grid lines and classic framing
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



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

deciles_file = "figures/calibration_base_models_deciles.csv"
df = pd.read_csv(deciles_file)

# Filter for 14.0 year horizon
df_14 = df[df['horizon_years'] == 14.0].copy()

out_dir = "figures/sklearn_style"
os.makedirs(out_dir, exist_ok=True)

# 1. Individual 1-to-1 Style Plots matching the user's reference image exactly
tasks = [
    ("FHS Test", "CpG-Only Model (10 CpGs)", "fhs_cpg_only", "Calibration Curve - FHS Test (CpG Alone, 14-Year)"),
    ("FHS Test", "CpG + Clinical Model (14 Features)", "fhs_cpg_clinical", "Calibration Curve - FHS Test (CpG + Clinical, 14-Year)"),
    ("WHI External", "CpG-Only Model (10 CpGs)", "whi_cpg_only", "Calibration Curve - WHI External (CpG Alone, 14-Year)"),
    ("WHI External", "CpG + Clinical Model (14 Features)", "whi_cpg_clinical", "Calibration Curve - WHI External (CpG + Clinical, 14-Year)")
]

for cohort, model_name, file_suffix, title in tasks:
    sub = df_14[(df_14['cohort'] == cohort) & (df_14['model'] == model_name)].sort_values('bin')
    x = sub['mean_pred_risk'].values
    y = sub['observed_risk'].values
    
    # Version A: Exactly 0.0 to 1.0 (matching reference image)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    ax.plot(x, y, marker='o', markersize=6.5, color='blue', linewidth=1.8, label='Calibration Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Ideal Calibration')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=13, pad=12, fontweight='medium')
    ax.grid(True, linestyle='-', alpha=0.65)
    ax.legend(loc='upper left', frameon=True, fontsize=11, facecolor='white', framealpha=0.95)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/calibration_1to1_{file_suffix}_scale10.png", dpi=300)
    plt.close(fig)

    # Version B: Focused Scale (e.g. up to 0.75 for FHS, 0.40 for WHI)
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    ax.plot(x, y, marker='o', markersize=6.5, color='blue', linewidth=1.8, label='Calibration Curve')
    max_lim = 0.75 if "FHS" in cohort else 0.40
    ax.plot([0, max_lim], [0, max_lim], linestyle='--', color='gray', linewidth=1.5, label='Ideal Calibration')
    
    ax.set_xlim(-0.02, max_lim)
    ax.set_ylim(-0.02, max_lim)
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f"{title} (Focused View)", fontsize=13, pad=12, fontweight='medium')
    ax.grid(True, linestyle='-', alpha=0.65)
    ax.legend(loc='upper left', frameon=True, fontsize=11, facecolor='white', framealpha=0.95)
    plt.tight_layout()
    fig.savefig(f"{out_dir}/calibration_focused_{file_suffix}.png", dpi=300)
    plt.close(fig)

# 2. Side-by-Side Comparison Plot (FHS vs WHI, both models on each cohort)
fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), dpi=300)

for idx, (cohort, ax) in enumerate([("FHS Test", axes[0]), ("WHI External", axes[1])]):
    sub_cpg = df_14[(df_14['cohort'] == cohort) & (df_14['model'] == "CpG-Only Model (10 CpGs)")].sort_values('bin')
    sub_clin = df_14[(df_14['cohort'] == cohort) & (df_14['model'] == "CpG + Clinical Model (14 Features)")].sort_values('bin')
    
    # Plot Ideal
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Ideal Calibration')
    
    # Plot CpG Alone
    ax.plot(sub_cpg['mean_pred_risk'].values, sub_cpg['observed_risk'].values, 
            marker='o', markersize=7, color='#0052CC', linewidth=2.0, 
            label='CpG Alone Model (10 CpGs)')
    
    # Plot CpG + Clinical
    ax.plot(sub_clin['mean_pred_risk'].values, sub_clin['observed_risk'].values, 
            marker='s', markersize=7, color='#D55E00', linewidth=2.0, 
            label='CpG + Clinical Model (14 Features)')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    cohort_title = f"{cohort} Cohort (14-Year End Point)"
    ax.set_title(cohort_title, fontsize=13, pad=12, fontweight='bold')
    ax.grid(True, linestyle='-', alpha=0.65)
    ax.legend(loc='upper left', frameon=True, fontsize=11, facecolor='white', framealpha=0.95)

plt.suptitle("Calibration Curves: Base Models at 14-Year End Point (Scikit-Learn Style)", fontsize=15, y=0.98, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{out_dir}/calibration_comparison_sklearn_style_14y.png", dpi=300)
fig.savefig(f"figures/calibration_comparison_sklearn_style_14y.png", dpi=300)
fig.savefig(f"figures/calibration_comparison_sklearn_style_14y.pdf")
plt.close(fig)

# 3. Exactly identical to the uploaded image: single plot with just title "Calibration Curve",
# showing FHS CpG + Clinical (the final base model) exactly mimicking the user's uploaded plot
sub_fhs_clin = df_14[(df_14['cohort'] == "FHS Test") & (df_14['model'] == "CpG + Clinical Model (14 Features)")].sort_values('bin')
fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
ax.plot(sub_fhs_clin['mean_pred_risk'].values, sub_fhs_clin['observed_risk'].values, 
        marker='o', markersize=6.5, color='blue', linewidth=1.8, label='Calibration Curve')
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, label='Ideal Calibration')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(np.arange(0.0, 1.1, 0.2))
ax.set_yticks(np.arange(0.0, 1.1, 0.2))
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontsize=12)
ax.set_title('Calibration Curve', fontsize=14, pad=10)
ax.grid(True, linestyle='-', alpha=0.65)
ax.legend(loc='upper left', frameon=True, fontsize=11, facecolor='white', framealpha=0.95)
plt.tight_layout()
exact_match_file = f"{out_dir}/calibration_curve_exact_match_fhs_cpg_clinical.png"
fig.savefig(exact_match_file, dpi=300)
fig.savefig("figures/calibration_curve_exact_match_fhs_cpg_clinical.png", dpi=300)
plt.close(fig)

print("All plots generated successfully!")
