#!/usr/bin/env python
"""
Stage 2 Feature-set Construction

This script constructs the feature sets analyzed in this study:
1. CpG-only features: The ten canonical epigenetic clock CpGs.
2. CpG + Clinical features: The ten CpGs combined with four readily-available
   clinical biomarkers (Total Cholesterol, HDL, Glucose, and Triglycerides).
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# The ten canonical age-associated CpGs from the parent epigenetic clock paper
EPICLOCK_CPGS = [
    "cg04604946",
    "cg06639320",
    "cg07955995",
    "cg10246448",
    "cg10501210",
    "cg11617964",
    "cg16867657",
    "cg20591472",
    "cg24724428",
    "cg25590826"
]

# The four routine clinical biomarkers
CLINICAL_BIOMARKERS = [
    "Total_Cholesterol",
    "HDL",
    "Glucose",
    "Triglycerides"
]

def construct_feature_sets(df):
    """
    Constructs the feature matrix for both CpG-only and CpG + clinical feature sets.
    """
    logger.info("Constructing feature sets from input data...")
    
    # 1. CpG-only feature set
    cpg_only_features = [col for col in EPICLOCK_CPGS if col in df.columns]
    logger.info(f"CpG-only feature set size: {len(cpg_only_features)} probes.")
    
    # 2. Combined CpG + Clinical feature set
    clinical_features = [col for col in CLINICAL_BIOMARKERS if col in df.columns]
    combined_features = cpg_only_features + clinical_features
    logger.info(f"Combined feature set size: {len(combined_features)} features ({len(cpg_only_features)} CpGs + {len(clinical_features)} biomarkers).")
    
    return {
        "cpg_only": df[cpg_only_features].copy(),
        "combined": df[combined_features].copy(),
        "metadata": df[["patient_id", "time", "event", "age", "sex"]].copy()
    }

def main():
    logger.info("Feature-set construction stage initialized.")
    logger.info(f"Ten EpiClock CpGs: {EPICLOCK_CPGS}")
    logger.info(f"Four Clinical Biomarkers: {CLINICAL_BIOMARKERS}")

if __name__ == "__main__":
    main()
