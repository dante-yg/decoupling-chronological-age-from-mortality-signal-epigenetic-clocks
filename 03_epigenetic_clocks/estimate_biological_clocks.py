"""Estimate biological epigenetic clocks using pyaging on 450K DNA methylation data.

================================================================================
ESTIMATE BIOLOGICAL CLOCKS (Horvath, Hannum, PhenoAge, GrimAge)
================================================================================

PURPOSE:
Calculates four established biological epigenetic clocks using the pyaging library:
  1. Horvath (2013) pan-tissue clock (353 CpGs)
  2. Hannum et al. (2013) whole-blood clock (71 features: 69 CpGs + 2 non-CpG probes)
  3. DNAm PhenoAge (Levine et al. 2018; 513 CpGs)
  4. DNAm GrimAge (Lu et al. 2019; 1,032 features: 1,030 CpGs + chronological age + female sex)

INPUTS:
- FHS methylation matrix: config.DATA_DIR / "FHS_methylation_merged.parquet"
  (columns: 'Share_ID', 'sex', 'age', 'event', 'time', and 450K CpG beta values)
- WHI methylation matrix: config.DATA_DIR / "WHI_full_imputed.parquet"
  (columns: 'Subject_ID', 'sex ', 'age', 'event ', 'time', and 450K CpG beta values)
- Model weight files: downloaded automatically to pyaging_data/ from
  https://pyaging.s3.amazonaws.com/clocks/weights0.1.0/

OUTPUTS:
- config.PREDICTIONS_DIR / "fhs_biological_clocks.csv"
  (FHS cohort metadata + predicted epigenetic clock ages)
- config.PREDICTIONS_DIR / "whi_biological_clocks.csv"
  (WHI cohort metadata + predicted epigenetic clock ages)
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
import sys
import urllib.request
import torch
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import anndata as ad
import pyaging as pya

# Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(BASE_DIR, "predictions")
DATA_DIR = "pyaging_data"
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FHS_PATH = config.DATA_DIR / "FHS_methylation_merged.parquet" if (config.DATA_DIR / "FHS_methylation_merged.parquet").exists() else config.RAW_DIR / "FHS_methylation_merged.parquet"
WHI_PATH = config.DATA_DIR / "WHI_full_imputed.parquet" if (config.DATA_DIR / "WHI_full_imputed.parquet").exists() else config.RAW_DIR / "WHI_full_imputed.parquet"

CLOCKS = ['horvath2013', 'hannum', 'dnamphenoage', 'grimage']

def main():
    print("=== Epigenetic Biological Age Estimation Pipeline ===")
    
    # 1. Retrieve all required CpG features across the 4 clocks
    print("\n[Step 1] Loading clock models to retrieve required CpGs...")
    all_cpgs = set()
    clock_models = {}
    for name in CLOCKS:
        url = f"https://pyaging.s3.amazonaws.com/clocks/weights0.1.0/{name}.pt"
        path = os.path.join(DATA_DIR, f"{name}.pt")
        if not os.path.exists(path):
            print(f"Downloading {name} model weights from {url}...")
            urllib.request.urlretrieve(url, path)
        
        # Load clock and extract required features
        model = torch.load(path, map_location='cpu', weights_only=False)
        print(f"  Clock '{name}' requires {len(model.features)} CpG probes.")
        clock_models[name] = model
        all_cpgs.update(model.features)
        
    print(f"Total unique CpG probes across all clocks: {len(all_cpgs)}")
    
    # 2. Process FHS Cohort
    print("\n[Step 2] Processing FHS Cohort...")
    if os.path.exists(FHS_PATH):
        # Read FHS schema to find matching columns
        fhs_file = pq.ParquetFile(FHS_PATH)
        fhs_all_cols = fhs_file.schema.names
        fhs_cpg_cols = [col for col in all_cpgs if col in fhs_all_cols]
        print(f"  Found {len(fhs_cpg_cols)} out of {len(all_cpgs)} required CpG columns in FHS parquet.")
        
        # Metadata columns
        fhs_meta_cols = ['Share_ID', 'sex', 'age', 'event', 'time']
        load_cols = list(set(fhs_meta_cols + fhs_cpg_cols))
        
        print(f"  Loading FHS data (only required columns)...")
        df_fhs = pd.read_parquet(FHS_PATH, columns=load_cols, engine='pyarrow')
        print(f"  FHS loaded successfully. Shape: {df_fhs.shape}")
        
        # Standardize metadata naming
        df_fhs = df_fhs.rename(columns={'Share_ID': 'patient_id'})
        
        # Convert to AnnData
        print("  Converting FHS data to AnnData...")
        # We separate the CpG features (X) and metadata (obs)
        X_fhs = df_fhs[fhs_cpg_cols].values
        obs_fhs = df_fhs[['patient_id', 'sex', 'age', 'event', 'time']].copy()
        
        # Create AnnData object
        adata_fhs = ad.AnnData(X=X_fhs, obs=obs_fhs)
        adata_fhs.var_names = fhs_cpg_cols
        
        # Run pyaging predictions
        print("  Predicting ages for FHS...")
        pya.pred.predict_age(adata_fhs, clock_names=CLOCKS, dir=DATA_DIR, verbose=False)
        
        # Save predictions
        fhs_preds = adata_fhs.obs.copy()
        fhs_preds_path = os.path.join(PRED_DIR, "fhs_biological_clocks.csv")
        fhs_preds.to_csv(fhs_preds_path, index=False)
        print(f"  FHS clock predictions saved to: {fhs_preds_path}")
    else:
        print(f"  Error: FHS parquet file not found at {FHS_PATH}")
        
    # 3. Process WHI Cohort
    print("\n[Step 3] Processing WHI Cohort...")
    if os.path.exists(WHI_PATH):
        # Read WHI schema with thrift limits to avoid deserialization exception
        limit = 2**31 - 1
        whi_file = pq.ParquetFile(WHI_PATH, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
        whi_all_cols = whi_file.schema.names
        whi_cpg_cols = [col for col in all_cpgs if col in whi_all_cols]
        print(f"  Found {len(whi_cpg_cols)} out of {len(all_cpgs)} required CpG columns in WHI parquet.")
        
        # Metadata columns (note trailing spaces in 'sex ' and 'event ')
        whi_meta_cols = ['Subject_ID', 'sex ', 'age', 'event ', 'time']
        load_cols = list(set(whi_meta_cols + list(whi_cpg_cols)))
        
        print(f"  Loading WHI data (only required columns with thrift limits)...")
        df_whi = pd.read_parquet(
            WHI_PATH, 
            columns=load_cols, 
            engine='pyarrow',
            thrift_string_size_limit=limit,
            thrift_container_size_limit=limit
        )
        print(f"  WHI loaded successfully. Shape: {df_whi.shape}")
        
        # Standardize metadata naming and strip trailing spaces
        df_whi = df_whi.rename(columns={
            'Subject_ID': 'patient_id',
            'sex ': 'sex',
            'event ': 'event'
        })
        
        # Convert to AnnData
        print("  Converting WHI data to AnnData...")
        X_whi = df_whi[whi_cpg_cols].values
        obs_whi = df_whi[['patient_id', 'sex', 'age', 'event', 'time']].copy()
        
        # Create AnnData object
        adata_whi = ad.AnnData(X=X_whi, obs=obs_whi)
        adata_whi.var_names = whi_cpg_cols
        
        # Run pyaging predictions
        print("  Predicting ages for WHI...")
        pya.pred.predict_age(adata_whi, clock_names=CLOCKS, dir=DATA_DIR, verbose=False)
        
        # Save predictions
        whi_preds = adata_whi.obs.copy()
        whi_preds_path = os.path.join(PRED_DIR, "whi_biological_clocks.csv")
        whi_preds.to_csv(whi_preds_path, index=False)
        print(f"  WHI clock predictions saved to: {whi_preds_path}")
    else:
        print(f"  Error: WHI parquet file not found at {WHI_PATH}")
        
    print("\n=== EPIGENETIC CLOCK PREDICTION PIPELINE COMPLETED ===")

if __name__ == '__main__':
    main()
