import os
import sys
import gzip
import torch
import pyarrow.parquet as pq
import pandas as pd

# Compatibility shim
import numpy as np
np.unicode_ = np.str_
import importlib.metadata, importlib_metadata
_orig = importlib.metadata.version
importlib.metadata.version = lambda p: np.__version__ if p == 'numpy' else _orig(p)
importlib_metadata.version = importlib.metadata.version

# Paths
PYAGING_DIR = r"C:\sharing\base_3\pyaging_data"
FHS_PARQUET = r"C:\Yugan_Gogul\datasets\combined\data\FHS_methylation_merged.parquet"
WHI_PARQUET = r"C:\Yugan_Gogul\datasets\combined\data\WHI_full_imputed.parquet"
MANIFEST_PATH = r"C:\sharing\base_3\scratch\annotations\HM450.hg19.manifest.tsv.gz"
FHS_RAW_DIR = r"D:\FHS_processed"

# 1. Load GrimAge model
model = torch.load(os.path.join(PYAGING_DIR, "grimage.pt"), map_location="cpu", weights_only=False)
grimage_features = list(model.features)
cpg_features = [f for f in grimage_features if f.startswith("cg")]
print(f"GrimAge total features: {len(grimage_features)}, CpG features: {len(cpg_features)}")

# 2. Get FHS and WHI column names
limit = 2**31 - 1
fhs_pf = pq.ParquetFile(FHS_PARQUET, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
fhs_cols = set(fhs_pf.schema.names)

whi_pf = pq.ParquetFile(WHI_PARQUET, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
whi_cols = set(whi_pf.schema.names)

present_fhs = [f for f in cpg_features if f in fhs_cols]
absent_fhs = [f for f in cpg_features if f not in fhs_cols]

present_whi = [f for f in cpg_features if f in whi_cols]
absent_whi = [f for f in cpg_features if f not in whi_cols]

print(f"\nFHS: present={len(present_fhs)}, absent={len(absent_fhs)}")
print(f"WHI: present={len(present_whi)}, absent={len(absent_whi)}")

# 3. Read 450K Manifest properly using Probe_ID column (index 8)
print("\nReading 450K manifest...")
manifest_cpgs = set()
with gzip.open(MANIFEST_PATH, 'rt', encoding='utf-8', errors='ignore') as f:
    header = f.readline().strip().split('\t')
    probe_idx = header.index("Probe_ID")
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > probe_idx:
            manifest_cpgs.add(parts[probe_idx])

print(f"Total probes in 450K manifest: {len(manifest_cpgs)}")
cpgs_on_450k = [c for c in cpg_features if c in manifest_cpgs]
cpgs_not_on_450k = [c for c in cpg_features if c not in manifest_cpgs]
print(f"GrimAge CpGs on 450K array: {len(cpgs_on_450k)} / {len(cpg_features)}")
print(f"GrimAge CpGs NOT on 450K array: {len(cpgs_not_on_450k)}")
if cpgs_not_on_450k:
    print(f"  CpGs not on 450K: {cpgs_not_on_450k}")

absent_fhs_on_450k = [c for c in absent_fhs if c in manifest_cpgs]
absent_fhs_not_on_450k = [c for c in absent_fhs if c not in manifest_cpgs]
print(f"\nFHS-absent GrimAge CpGs on 450K array: {len(absent_fhs_on_450k)}")
print(f"FHS-absent GrimAge CpGs NOT on 450K array: {len(absent_fhs_not_on_450k)}")

absent_whi_on_450k = [c for c in absent_whi if c in manifest_cpgs]
absent_whi_not_on_450k = [c for c in absent_whi if c not in manifest_cpgs]
print(f"\nWHI-absent GrimAge CpGs on 450K array: {len(absent_whi_on_450k)}")
print(f"WHI-absent GrimAge CpGs NOT on 450K array: {len(absent_whi_not_on_450k)}")
print(f"  WHI-absent details: {absent_whi}")

# 4. Check the common_probes files in D:\FHS_processed
# In these files, first column has probe names.
print("\nChecking probe IDs in D:\\FHS_processed common_probes files...")
batch_common = {}
for b in ["Gen3", "JHU", "UMN"]:
    p = os.path.join(FHS_RAW_DIR, f"{b}_common_probes.csv")
    if os.path.exists(p):
        print(f"Reading first column of {b}_common_probes.csv...")
        # read only first column
        probes = set()
        with open(p, 'r') as f:
            header = f.readline()
            for line in f:
                cpg = line.split(',', 1)[0].strip().strip('"')
                if cpg.startswith("cg"):
                    probes.add(cpg)
        batch_common[b] = probes
        print(f"  {b} has {len(probes)} CpGs starting with 'cg'.")

# Intersections across batches
if len(batch_common) == 3:
    gen3 = batch_common["Gen3"]
    jhu = batch_common["JHU"]
    umn = batch_common["UMN"]
    common_all = gen3 & jhu & umn
    print(f"\nIntersection across Gen3, JHU, UMN common probes: {len(common_all)}")
    
    # Check where absent_fhs_on_450k exist!
    in_gen3 = [c for c in absent_fhs_on_450k if c in gen3]
    in_jhu = [c for c in absent_fhs_on_450k if c in jhu]
    in_umn = [c for c in absent_fhs_on_450k if c in umn]
    in_all_three = [c for c in absent_fhs_on_450k if c in common_all]
    
    print(f"\nOf the {len(absent_fhs_on_450k)} FHS-absent 450K GrimAge probes:")
    print(f"  Present in Gen3 common: {len(in_gen3)} / {len(absent_fhs_on_450k)}")
    print(f"  Present in JHU common: {len(in_jhu)} / {len(absent_fhs_on_450k)}")
    print(f"  Present in UMN common: {len(in_umn)} / {len(absent_fhs_on_450k)}")
    print(f"  Present in ALL THREE common: {len(in_all_three)} / {len(absent_fhs_on_450k)}")

# Check Filtered_SNP files
print("\nChecking Filtered_SNP columns...")
for b in ["Gen3", "JHU", "UMN"]:
    snp_p = os.path.join(FHS_RAW_DIR, f"Filtered_SNP_{b}.csv")
    if os.path.exists(snp_p):
        with open(snp_p, 'r') as f:
            line = f.readline()
        cols = set([x.strip().strip('"') for x in line.split(',') if x.strip().strip('"').startswith('cg')])
        print(f"  Filtered_SNP_{b} has {len(cols)} CpGs.")
        in_snp = [c for c in absent_fhs_on_450k if c in cols]
        print(f"    Of absent_fhs_on_450k, present in Filtered_SNP_{b}: {len(in_snp)}")
