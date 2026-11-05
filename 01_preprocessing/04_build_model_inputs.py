"""Assemble model input CSVs from preprocessed methylation and dbGaP phenotypes.

================================================================================
STEP 04: BUILD MODEL INPUTS (SKELETON / PROVENANCE DOCUMENTATION)
================================================================================

PURPOSE:
This script documents the step that merges the preprocessed Illumina 450K DNA
methylation beta values (from Steps 01–03) with clinical phenotypes and survival
endpoints from dbGaP to construct the final modeling datasets in data/:
  1. data/FHS_non_normalized.csv (n = 3,493)
  2. data/WHI_non_normalized.csv (n = 196)
  3. data/Disease_data_with_censoring.csv (n = 3,493)

REQUIRED INPUT FILES:
1. Methylation beta values (Steps 01-03):
   - FHS: 10 target CpGs extracted from FHS_methylation_merged.parquet (or raw
     batches: Gen 3, JHU, UMN)
   - WHI: 10 target CpGs extracted from WHI_full_imputed.parquet
   - 10 CpGs: cg04604946, cg06639320, cg07955995, cg10246448, cg10501210,
              cg11617964, cg16867657, cg20591472, cg24724428, cg25590826
2. FHS dbGaP Phenotypes (phs000007):
   - Exam clinic/demographics: age at baseline exam, sex, exam dates, batch
   - Fasting blood biomarkers: Total_Cholesterol, Triglycerides, Glucose, HDL
   - Mortality follow-up: time (years to death or censoring), event (1=death, 0=censored)
   - ID Cross-walk: Share_ID <-> dbGaP_Subject_ID
   - Adjudicated CVD outcomes: event and follow-up time for MI, AP, CVA, IC, CHF
3. WHI dbGaP Phenotypes (phs000200):
   - Baseline demographics: age, sex (all female)
   - Fasting blood biomarkers: Total_Cholesterol, Triglycerides, Glucose, HDL
   - Mortality follow-up: time (years to death or censoring), event (1=death, 0=censored)
   - ID columns: SUBJECT_ID, ID

MERGE KEYS:
- FHS: Share_ID (participant identifier in FHS methylation arrays)
- WHI: SUBJECT_ID / ID (participant identifier in WHI BA23 methylation data)
- FHS Disease Data: dbGaP_Subject_ID

OUTPUT SPECIFICATIONS:
1. data/FHS_non_normalized.csv
   Columns: Share_ID, dbGaP_Subject_ID, sex, age, event, batch, time,
            Total_Cholesterol, Triglycerides, Glucose, HDL,
            cg04604946, cg06639320, cg07955995, cg10246448, cg10501210,
            cg11617964, cg16867657, cg20591472, cg24724428, cg25590826
2. data/WHI_non_normalized.csv
   Columns: SUBJECT_ID, ID, age, sex, event, time,
            Total_Cholesterol, Triglycerides, Glucose, HDL,
            cg04604946, cg06639320, cg07955995, cg10246448, cg10501210,
            cg11617964, cg16867657, cg20591472, cg24724428, cg25590826
3. data/Disease_data_with_censoring.csv
   Columns: dbGaP_Subject_ID, sex, age, batch,
            MI, MI_time, AP, AP_time, CVA, CVA_time, IC, IC_time, CHF, CHF_time
"""

import sys
from pathlib import Path

# Resolve repository root and import config
_repo_root = Path(__file__).resolve().parent
while _repo_root != _repo_root.parent:
    if (_repo_root / "config.py").exists():
        if str(_repo_root) not in sys.path:
            sys.path.insert(0, str(_repo_root))
        break
    _repo_root = _repo_root.parent

import config
import pandas as pd


# ==============================================================================
# Pipeline Interface
# ------------------------------------------------------------------------------
# The assembly of data/FHS_non_normalized.csv, data/WHI_non_normalized.csv,
# and data/Disease_data_with_censoring.csv was performed by merging preprocessed
# methylation matrices with dbGaP phenotype tables during cohort preparation.
#
# This script serves as reference documentation for the required inputs, merge keys,
# and schemas. Direct pipeline execution will be enabled upon public data release.
# ==============================================================================

def build_fhs_model_inputs():
    """Build data/FHS_non_normalized.csv and data/Disease_data_with_censoring.csv."""
    raise NotImplementedError(
        "Direct phenotype-methylation assembly script will be released with final public data access. "
        "See docstring for expected input files, merge keys, and column schema."
    )


def build_whi_model_inputs():
    """Build data/WHI_non_normalized.csv."""
    raise NotImplementedError(
        "Direct phenotype-methylation assembly script will be released with final public data access. "
        "See docstring for expected input files, merge keys, and column schema."
    )


def main():
    print("01_preprocessing/04_build_model_inputs.py")
    print("------------------------------------------")
    print("This script is a documented skeleton for the phenotype-methylation merge step.")
    print("Expected outputs:")
    print(f"  - {config.DATA_DIR / 'FHS_non_normalized.csv'}")
    print(f"  - {config.DATA_DIR / 'WHI_non_normalized.csv'}")
    print(f"  - {config.DATA_DIR / 'Disease_data_with_censoring.csv'}")
    print("\nTo generate synthetic mock data for code testing without dbGaP data, run:")
    print("  python tools/make_synthetic_data.py")


if __name__ == "__main__":
    main()

