#!/usr/bin/env python
"""
Stage 4 Clock Comparison: Biological Clocks Estimation

This script retrieves weights for established biological clocks (Horvath, Hannum,
PhenoAge, GrimAge) and calculates their age predictions on the FHS and WHI cohorts
using the pyaging package.
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path
import torch
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import anndata as ad
import pyaging as pya

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

# Default raw parquet file paths (access-restricted dbGaP files)
FHS_PATH = r"F:\dbgap\FHS_methylation_merged.parquet"
WHI_PATH = r"F:\dbgap\WHI_full_imputed.parquet"

# Clock specifications
CLOCKS = ["horvath2013", "hannum", "dnamphenoage", "grimage"]

def main():
    logger.info("Initializing Epigenetic Biological Age Estimation Pipeline...")
    
    # Resolve directories relative to repo root
    REPO_ROOT = Path(__file__).resolve().parents[2]
    PRED_DIR = REPO_ROOT / "predictions"
    DATA_DIR = REPO_ROOT / "pyaging_data"
    
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Retrieve all required CpG features across the 4 clocks
    logger.info("Loading clock models to retrieve required CpGs...")
    all_cpgs = set()
    clock_models = {}
    for name in CLOCKS:
        url = f"https://pyaging.s3.amazonaws.com/clocks/weights0.1.0/{name}.pt"
        path = DATA_DIR / f"{name}.pt"
        if not path.exists():
            logger.info(f"Downloading {name} model weights from {url}...")
            urllib.request.urlretrieve(url, path)
        
        # Load clock weights and extract required features
        model = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(f"  Clock '{name}' requires {len(model.features)} CpG probes.")
        clock_models[name] = model
        all_cpgs.update(model.features)
        
    logger.info(f"Total unique CpG probes across all clocks: {len(all_cpgs)}")
    
    # 2. Process FHS Cohort
    logger.info("Processing FHS Cohort...")
    if os.path.exists(FHS_PATH):
        fhs_file = pq.ParquetFile(FHS_PATH)
        fhs_all_cols = fhs_file.schema.names
        fhs_cpg_cols = [col for col in all_cpgs if col in fhs_all_cols]
        logger.info(f"  Found {len(fhs_cpg_cols)} out of {len(all_cpgs)} required CpG columns in FHS.")
        
        fhs_meta_cols = ["Share_ID", "sex", "age", "event", "time"]
        load_cols = list(set(fhs_meta_cols + fhs_cpg_cols))
        
        logger.info("  Loading FHS data (only required columns)...")
        df_fhs = pd.read_parquet(FHS_PATH, columns=load_cols, engine="pyarrow")
        df_fhs = df_fhs.rename(columns={"Share_ID": "patient_id"})
        
        logger.info("  Converting FHS data to AnnData...")
        X_fhs = df_fhs[fhs_cpg_cols].values
        obs_fhs = df_fhs[["patient_id", "sex", "age", "event", "time"]].copy()
        
        adata_fhs = ad.AnnData(X=X_fhs, obs=obs_fhs)
        adata_fhs.var_names = fhs_cpg_cols
        
        logger.info("  Predicting ages for FHS...")
        pya.pred.predict_age(adata_fhs, clock_names=CLOCKS, dir=str(DATA_DIR), verbose=False)
        
        fhs_preds = adata_fhs.obs.copy()
        fhs_preds_path = PRED_DIR / "fhs_biological_clocks.csv"
        fhs_preds.to_csv(fhs_preds_path, index=False)
        logger.info(f"  FHS clock predictions saved -> {fhs_preds_path}")
    else:
        logger.warning(f"FHS parquet file not found at {FHS_PATH}. Skipping FHS clock estimation.")
        
    # 3. Process WHI Cohort
    logger.info("Processing WHI Cohort...")
    if os.path.exists(WHI_PATH):
        limit = 2**31 - 1
        whi_file = pq.ParquetFile(WHI_PATH, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
        whi_all_cols = whi_file.schema.names
        whi_cpg_cols = [col for col in all_cpgs if col in whi_all_cols]
        logger.info(f"  Found {len(whi_cpg_cols)} out of {len(all_cpgs)} required CpG columns in WHI.")
        
        whi_meta_cols = ["Subject_ID", "sex ", "age", "event ", "time"]
        load_cols = list(set(whi_meta_cols + list(whi_cpg_cols)))
        
        logger.info("  Loading WHI data (only required columns)...")
        df_whi = pd.read_parquet(
            WHI_PATH, 
            columns=load_cols, 
            engine="pyarrow",
            thrift_string_size_limit=limit,
            thrift_container_size_limit=limit
        )
        df_whi = df_whi.rename(columns={
            "Subject_ID": "patient_id",
            "sex ": "sex",
            "event ": "event"
        })
        
        logger.info("  Converting WHI data to AnnData...")
        X_whi = df_whi[whi_cpg_cols].values
        obs_whi = df_whi[["patient_id", "sex", "age", "event", "time"]].copy()
        
        adata_whi = ad.AnnData(X=X_whi, obs=obs_whi)
        adata_whi.var_names = whi_cpg_cols
        
        logger.info("  Predicting ages for WHI...")
        pya.pred.predict_age(adata_whi, clock_names=CLOCKS, dir=str(DATA_DIR), verbose=False)
        
        whi_preds = adata_whi.obs.copy()
        whi_preds_path = PRED_DIR / "whi_biological_clocks.csv"
        whi_preds.to_csv(whi_preds_path, index=False)
        logger.info(f"  WHI clock predictions saved -> {whi_preds_path}")
    else:
        logger.warning(f"WHI parquet file not found at {WHI_PATH}. Skipping WHI clock estimation.")
        
    logger.info("Biological age estimation completed.")

if __name__ == "__main__":
    main()
