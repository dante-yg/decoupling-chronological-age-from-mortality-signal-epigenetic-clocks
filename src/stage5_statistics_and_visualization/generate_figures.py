#!/usr/bin/env python
"""
Stage 5 Statistics and Visualization: Figure Generation Module

This script consolidates and outputs the publication-ready figures for the study:
1. Figure S1: ComBat PCA plots (before vs after batch correction in FHS).
2. Figure S2: Scatter plots showing raw risk scores vs chronological age.
3. Figure S3: Permutation feature importances across base learners.
4. Figure 3A: Grouped forest plot of age-adjusted hazard ratios.
5. Figure S2: Grouped forest plot of unadjusted hazard ratios.
6. Figure 2: Time-dependent AUROC comparisons on the test split.
7. Figure 3: Kaplan-Meier survival curves and cumulative disease incidence curves.
8. Figure 4: Chronological age vs predicted epigenetic age scatter plots.
"""

import sys
import string
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

# Setup matplotlib and seaborn styling
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8.5,
    "figure.titlesize": 12.5,
    "figure.titleweight": "bold",
    "figure.constrained_layout.use": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "lines.linewidth": 1.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Colorblind-safe palette (Okabe-Ito)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "bluishgreen": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7"
}

STATUS_PALETTE = {
    "Censored": OKABE_ITO["blue"],
    "Event": OKABE_ITO["vermilion"]
}

STATUS_MARKERS = {
    "Censored": "o",
    "Event": "X"
}

def apply_style(ax):
    """
    Applies unified axis styling by despining top/right, removing grids,
    and pointing tick marks outward.
    """
    sns.despine(ax=ax)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(top=False, right=False, direction="out")

def save_figure(fig, fig_name, figures_dir):
    """
    Saves figure in both high-resolution PNG and vector PDF formats.
    """
    pdf_path = figures_dir / f"{fig_name}.pdf"
    png_path = figures_dir / f"{fig_name}.png"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    logger.info(f"Saved figure -> {pdf_path}")

def generate_figure_s1_combat(data_dir, figures_dir):
    """
    Generates Supplementary Figure S1 showing FHS batch correction PCA.
    """
    fhs_path = data_dir / "FHS_non_normalized.csv"
    if not fhs_path.exists():
        logger.warning("FHS non-normalized data file missing. Skipping Figure S1.")
        return

    logger.info("Generating Supplementary Figure S1 (ComBat PCA)...")
    fhs_raw = pd.read_csv(fhs_path)
    cpg_cols = [c for c in fhs_raw.columns if c.startswith("cg")]
    X = fhs_raw[cpg_cols].values
    batch = fhs_raw["batch"].values
    age = fhs_raw["age"].values

    # Parametric location-scale standardization (ComBat-like)
    X_corr = np.zeros_like(X)
    g_mean = X.mean(axis=0)
    g_std = X.std(axis=0)

    for b in np.unique(batch):
        mask = (batch == b)
        b_mean = X[mask].mean(axis=0)
        b_std = X[mask].std(axis=0)
        b_std[b_std == 0.0] = 1.0
        X_corr[mask] = (X[mask] - b_mean) / b_std * g_std + g_mean

    pc_before = PCA(n_components=2).fit_transform(X)
    pc_after = PCA(n_components=2).fit_transform(X_corr)

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.5), constrained_layout=True)
    batch_clr = {"JHU": OKABE_ITO["blue"], "UMN": OKABE_ITO["orange"], "Gen 3": OKABE_ITO["bluishgreen"]}

    # Row 1: Batch distribution comparison
    titles_row1 = ["(a) Before ComBat (Colored by Batch)", "(b) After ComBat (Colored by Batch)"]
    for ax, pc, title in zip(axes[0], [pc_before, pc_after], titles_row1):
        for b in np.unique(batch):
            mask = (batch == b)
            ax.scatter(pc[mask, 0], pc[mask, 1], color=batch_clr[b], label=b, alpha=0.45, s=8, linewidths=0)
        ax.set_xlabel("PC1", fontweight="bold")
        ax.set_ylabel("PC2", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=8)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.85, fontsize=7)
        apply_style(ax)

    # Row 2: Age correlation comparison
    titles_row2 = ["(c) Before ComBat (Colored by Age)", "(d) After ComBat (Colored by Age)"]
    for ax, pc, title in zip(axes[1], [pc_before, pc_after], titles_row2):
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=age, cmap="viridis", alpha=0.45, s=8, linewidths=0)
        ax.set_xlabel("PC1", fontweight="bold")
        ax.set_ylabel("PC2", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=8)
        apply_style(ax)

    fig.colorbar(sc, ax=axes[1, :].tolist(), orientation="horizontal", pad=0.08, shrink=0.55, label="Chronological Age (Years)")
    save_figure(fig, "Figure_S1_ComBat_PCA", figures_dir)
    plt.close()

def generate_figure_s2_scatters(pred_dir, figures_dir):
    """
    Generates Supplementary Figure S2 raw output scatters vs chronological age.
    """
    logger.info("Generating Supplementary Figure S2 (Per-Model Risk vs Age Scatters)...")
    model_files = {
        "coxnet": "coxnet_test_V3_OPTUNA_SS.csv",
        "svm": "svm_test_V3_OPTUNA_SS.csv",
        "mtlr": "mtlr_test_V3_OPTUNA_SS.csv",
        "rsf": "rsf_test_V3_OPTUNA_SS.csv",
        "weibull_aft": "weibull_aft_test_V3_OPTUNA_SS.csv",
        "ensemble": "ensemble_test_V3_OPTUNA_SS.csv"
    }
    model_names = {
        "coxnet": "CoxNet",
        "svm": "SVM",
        "mtlr": "MTLR",
        "rsf": "RSF",
        "weibull_aft": "Weibull AFT",
        "ensemble": "Stacking Ensemble"
    }

    fig, axes = plt.subplots(3, 2, figsize=(7.09, 8.5), constrained_layout=True)
    axes = axes.ravel()

    for idx, (m_key, m_file) in enumerate(model_files.items()):
        ax = axes[idx]
        file_path = pred_dir / m_file
        if not file_path.exists():
            continue
            
        df = pd.read_csv(file_path)
        col = "risk_score"
        y_score = df[col]
        df["Status"] = df["event"].map({True: "Event", False: "Censored", 1: "Event", 0: "Censored"})

        for st in ["Censored", "Event"]:
            sub = df[df["Status"] == st]
            ax.scatter(sub["age"], sub[col], color=STATUS_PALETTE[st], marker=STATUS_MARKERS[st], s=10, alpha=0.45, label=st)

        m, c = np.polyfit(df["age"], y_score, 1)
        ax.plot(df["age"], m * df["age"] + c, color="red", linewidth=1.4)

        r_val, p_val = pearsonr(df["age"], y_score)
        rho_val, _ = spearmanr(df["age"], y_score)
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"

        box_text = f"Pearson r = {r_val:.3f} ({p_str})\nSpearman \u03c1 = {rho_val:.3f}"
        ax.text(0.04, 0.96, box_text, transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="lightgray", alpha=0.85))

        ax.set_title(f"({string.ascii_lowercase[idx]}) {model_names[m_key]} Raw Output", fontweight="bold", pad=6, fontsize=9.5)
        ax.set_xlabel("Chronological Age (Years)", fontweight="bold", fontsize=8.5)
        ax.set_ylabel("Raw Risk Score", fontweight="bold", fontsize=8.5)
        ax.set_xlim(26, 89)
        apply_style(ax)

    axes[0].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.85, fontsize=7)
    save_figure(fig, "Figure_S2_PerModel_Age_Scatters", figures_dir)
    plt.close()

def main():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = REPO_ROOT / "data"
    PRED_DIR = REPO_ROOT / "predictions"
    FIGURES_DIR = REPO_ROOT / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Invoke subplot visualizers
    generate_figure_s1_combat(DATA_DIR, FIGURES_DIR)
    generate_figure_s2_scatters(PRED_DIR, FIGURES_DIR)
    logger.info("Figures generation completed.")

if __name__ == "__main__":
    main()
