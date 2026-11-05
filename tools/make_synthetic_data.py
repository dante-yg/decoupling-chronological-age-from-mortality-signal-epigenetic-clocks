"""Generate synthetic data for pipeline smoke-testing.

IMPORTANT NOTICE:
This script generates entirely synthetic, randomized data for testing that the
code runs end-to-end without errors. The values are purely simulated from random
distributions with plausible biological ranges; NO real participant data or dbGaP
records are accessed or included.

Results obtained on synthetic data are completely meaningless scientifically.
To reproduce the published findings, researchers must request access to the real
FHS (phs000007) and WHI (phs000200) datasets through dbGaP.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# The 10 EpiClock CpGs
CPG_10 = [
    "cg04604946", "cg06639320", "cg07955995", "cg10246448", "cg10501210",
    "cg11617964", "cg16867657", "cg20591472", "cg24724428", "cg25590826"
]

CLINICAL_COLS = ["Total_Cholesterol", "Triglycerides", "Glucose", "HDL"]

# Additional CpGs commonly referenced in clocks (Horvath, Hannum, PhenoAge, GrimAge)
EXTRA_CLOCK_CPGS = [
    f"cg{i:08d}" for i in [
        1470438, 2038753, 2197541, 3607001, 3747684, 4084157, 4774697, 4941443,
        5005882, 5440620, 6083517, 6144549, 6418403, 6493994, 6874449, 7388493,
        7955995, 8312092, 8547495, 8887097, 8927692, 9886361, 10501210, 11617964,
        13001142, 14361627, 16867657, 19722847, 21572722, 22736354, 24724428, 25590826
    ]
]


def generate_synthetic_data(output_dir: Path, seed: int = 42):
    """Generate fake datasets for smoke testing."""
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic datasets in: {output_dir}")

    # --------------------------------------------------------------------------
    # 1. FHS_non_normalized.csv (~300 rows)
    # --------------------------------------------------------------------------
    n_fhs = 300
    share_ids = np.arange(1001, 1001 + n_fhs)
    dbgap_ids = np.arange(2001, 2001 + n_fhs)
    fhs_sex = np.random.choice([1, 2], size=n_fhs, p=[0.48, 0.52])
    fhs_age = np.random.uniform(40.0, 85.0, size=n_fhs).round(1)
    fhs_event = np.random.choice([0, 1], size=n_fhs, p=[0.80, 0.20])
    fhs_batch = np.random.choice(["Gen 3", "JHU", "UMN"], size=n_fhs, p=[0.40, 0.35, 0.25])
    fhs_time = np.random.uniform(0.5, 14.8, size=n_fhs).round(2)

    fhs_chol = np.random.normal(205.0, 38.0, size=n_fhs).clip(120, 350).round(1)
    fhs_trig = np.random.normal(135.0, 50.0, size=n_fhs).clip(40, 400).round(1)
    fhs_gluc = np.random.normal(98.0, 18.0, size=n_fhs).clip(65, 250).round(1)
    fhs_hdl = np.random.normal(52.0, 14.0, size=n_fhs).clip(25, 110).round(1)

    fhs_data = {
        "Share_ID": share_ids,
        "dbGaP_Subject_ID": dbgap_ids,
        "sex": fhs_sex,
        "age": fhs_age,
        "event": fhs_event,
        "batch": fhs_batch,
        "time": fhs_time,
        "Total_Cholesterol": fhs_chol,
        "Triglycerides": fhs_trig,
        "Glucose": fhs_gluc,
        "HDL": fhs_hdl,
    }
    for cpg in CPG_10:
        fhs_data[cpg] = np.random.beta(a=2.0, b=5.0, size=n_fhs).clip(0.01, 0.99).round(4)

    df_fhs = pd.DataFrame(fhs_data)
    fhs_path = output_dir / "FHS_non_normalized.csv"
    df_fhs.to_csv(fhs_path, index=False)
    print(f"  [+] Wrote {fhs_path.name}: {df_fhs.shape}")

    # --------------------------------------------------------------------------
    # 2. WHI_non_normalized.csv (~60 rows, all female)
    # --------------------------------------------------------------------------
    n_whi = 60
    whi_subj_ids = np.arange(3001, 3001 + n_whi)
    whi_ids = whi_subj_ids
    whi_age = np.random.uniform(50.0, 80.0, size=n_whi).round(1)
    whi_sex = np.full(n_whi, 2)  # All female
    whi_event = np.random.choice([0, 1], size=n_whi, p=[0.72, 0.28])
    whi_time = np.random.uniform(0.5, 20.0, size=n_whi).round(2)

    whi_chol = np.random.normal(215.0, 35.0, size=n_whi).clip(120, 350).round(1)
    whi_trig = np.random.normal(140.0, 48.0, size=n_whi).clip(40, 400).round(1)
    whi_gluc = np.random.normal(96.0, 16.0, size=n_whi).clip(65, 250).round(1)
    whi_hdl = np.random.normal(55.0, 15.0, size=n_whi).clip(25, 110).round(1)

    whi_data = {
        "SUBJECT_ID": whi_subj_ids,
        "ID": whi_ids,
        "age": whi_age,
        "sex": whi_sex,
        "event": whi_event,
        "time": whi_time,
        "Total_Cholesterol": whi_chol,
        "Triglycerides": whi_trig,
        "Glucose": whi_gluc,
        "HDL": whi_hdl,
    }
    for cpg in CPG_10:
        whi_data[cpg] = np.random.beta(a=2.0, b=5.0, size=n_whi).clip(0.01, 0.99).round(4)

    df_whi = pd.DataFrame(whi_data)
    whi_path = output_dir / "WHI_non_normalized.csv"
    df_whi.to_csv(whi_path, index=False)
    print(f"  [+] Wrote {whi_path.name}: {df_whi.shape}")

    # --------------------------------------------------------------------------
    # 3. Disease_data_with_censoring.csv (matched to FHS)
    # --------------------------------------------------------------------------
    disease_data = {
        "dbGaP_Subject_ID": dbgap_ids,
        "sex": fhs_sex,
        "age": fhs_age,
        "batch": fhs_batch,
    }
    for disease in ["MI", "AP", "CVA", "IC", "CHF"]:
        ev = np.random.choice([0, 1], size=n_fhs, p=[0.92, 0.08])
        t = np.random.uniform(0.5, 14.8, size=n_fhs).round(2)
        disease_data[disease] = ev
        disease_data[f"{disease}_time"] = t

    df_disease = pd.DataFrame(disease_data)
    disease_path = output_dir / "Disease_data_with_censoring.csv"
    df_disease.to_csv(disease_path, index=False)
    print(f"  [+] Wrote {disease_path.name}: {df_disease.shape}")

    # --------------------------------------------------------------------------
    # 4. Small fake Parquet matrices for clock estimation smoke-test
    # --------------------------------------------------------------------------
    all_probe_cols = sorted(list(set(CPG_10 + EXTRA_CLOCK_CPGS)))

    # FHS Parquet
    fhs_pq_data = {
        "Share_ID": share_ids,
        "sex": fhs_sex,
        "age": fhs_age,
        "event": fhs_event,
        "time": fhs_time,
    }
    for p in all_probe_cols:
        fhs_pq_data[p] = np.random.uniform(0.02, 0.98, size=n_fhs).astype(np.float32)

    df_fhs_pq = pd.DataFrame(fhs_pq_data)
    fhs_pq_path = output_dir / "FHS_methylation_merged.parquet"
    df_fhs_pq.to_parquet(fhs_pq_path, engine="pyarrow")
    print(f"  [+] Wrote {fhs_pq_path.name}: {df_fhs_pq.shape}")

    # WHI Parquet (with trailing spaces in schema per dbGaP raw format)
    whi_pq_data = {
        "Subject_ID": whi_subj_ids,
        "sex ": whi_sex,
        "age": whi_age,
        "event ": whi_event,
        "time": whi_time,
    }
    for p in all_probe_cols:
        whi_pq_data[p] = np.random.uniform(0.02, 0.98, size=n_whi).astype(np.float32)

    df_whi_pq = pd.DataFrame(whi_pq_data)
    whi_pq_path = output_dir / "WHI_full_imputed.parquet"
    df_whi_pq.to_parquet(whi_pq_path, engine="pyarrow")
    print(f"  [+] Wrote {whi_pq_path.name}: {df_whi_pq.shape}")

    print("\nSynthetic data generation completed successfully!")
    print("Reminder: These files are for code verification only. Do not use for research.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for smoke-testing.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "synthetic_data"),
        help="Directory to save synthetic files."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_synthetic_data(Path(args.output_dir), seed=args.seed)
