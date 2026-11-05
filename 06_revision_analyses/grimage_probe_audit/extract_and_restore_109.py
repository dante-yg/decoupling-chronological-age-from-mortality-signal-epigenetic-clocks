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
import time
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Compatibility shim
np.unicode_ = np.str_
import importlib.metadata, importlib_metadata
_orig = importlib.metadata.version
importlib.metadata.version = lambda p: np.__version__ if p == 'numpy' else _orig(p)
importlib_metadata.version = importlib.metadata.version

import anndata as ad
import pyaging as pya
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from neuroCombat import neuroCombat
from lifelines import CoxPHFitter

print("================================================================================")
print("           FHS GRIMAGE 109-CPG RESTORATION & SENSITIVITY PIPELINE               ")
print("================================================================================")

# 1. Load FHS metadata and cohort splits
FHS_META_CSV = str(config.DATA_DIR / "FHS_non_normalized.csv")
fhs_df = pd.read_csv(FHS_META_CSV).rename(columns={"Share_ID": "patient_id"})
print(f"Loaded FHS cohort: {fhs_df.shape} subjects, {fhs_df['event'].sum()} deaths.")

# Exact stratified splits from reviewer_table3_rebuild.py
SEED = 42
fhs_df["event_stratify"] = fhs_df["event"].astype(int)
tr_va_df, te_df = train_test_split(fhs_df, test_size=0.20, random_state=SEED, stratify=fhs_df["event_stratify"])
tr_df, va_df = train_test_split(tr_va_df, test_size=0.25, random_state=SEED, stratify=tr_va_df["event_stratify"])

split_ids = {
    "Train": set(tr_df["patient_id"]),
    "Val":   set(va_df["patient_id"]),
    "Test":  set(te_df["patient_id"])
}
print(f"Split sizes: Train = {len(split_ids['Train'])}, Val = {len(split_ids['Val'])}, Test = {len(split_ids['Test'])}")

# 2. Identify the 109 CpGs to restore
causes_csv = str(config.REVIEWS_DIR / "grimage_probe_audit" / "grimage_missing_probe_causes.csv")
df_causes = pd.read_csv(causes_csv)
cpgs_109 = sorted(list(set(df_causes[df_causes["in_dbgap_missing_table"] == True]["CpG"])))
print(f"Restoring {len(cpgs_109)} GrimAge CpGs from raw dbGaP files...")

# 3. Extract 109 CpGs from raw dbGaP files
dbgap_dir = str(Path(os.environ.get("EPICLOCK_RAW_MATRIX_DIR", config.RAW_DIR / "FHS_DNAMethylation")))
batches = [
    ("Gen 3", os.path.join(dbgap_dir, "gen3_methylation_c1.csv")),
    ("JHU",   os.path.join(dbgap_dir, "JHU_methylation_c1.csv")),
    ("UMN",   os.path.join(dbgap_dir, "UMN_methylation_c1.csv"))
]

cpg_set = set(cpgs_109)
batch_dfs = []

for b_name, b_path in batches:
    t0 = time.time()
    print(f"\nScanning {b_name} from {b_path}...")
    fhs_batch_subjects = fhs_df[fhs_df["batch"] == b_name]["patient_id"].tolist()
    fhs_sub_set = set(fhs_batch_subjects)
    
    extracted_data = {}
    with open(b_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        sample_ids = [int(x) for x in header[1:]]
        keep_indices = [i for i, sid in enumerate(sample_ids) if sid in fhs_sub_set]
        kept_sample_ids = [sample_ids[i] for i in keep_indices]
        print(f"  {b_name}: {len(sample_ids)} total samples in file, {len(kept_sample_ids)} match FHS cohort.")
        
        for line in f:
            idx = line.find(",")
            if idx != -1:
                pid = line[:idx].strip('"').strip()
                if pid in cpg_set:
                    tokens = line.strip().split(",")
                    vals = []
                    for i in keep_indices:
                        val_str = tokens[i+1].strip()
                        vals.append(float(val_str) if val_str != "" and val_str != "NA" else np.nan)
                    extracted_data[pid] = vals
                    
    df_b = pd.DataFrame(extracted_data, index=kept_sample_ids)
    df_b["batch"] = b_name
    df_b["patient_id"] = kept_sample_ids
    print(f"  Extracted {len(extracted_data)} CpGs for {b_name} in {time.time() - t0:.2f}s.")
    batch_dfs.append(df_b)

df_all_109 = pd.concat(batch_dfs, axis=0)
# Reorder to match exactly fhs_df['patient_id']
df_all_109 = df_all_109.set_index("patient_id").loc[fhs_df["patient_id"]].reset_index()

# 4. Check and apply KNN Imputation (k=5)
X_109 = df_all_109[cpgs_109].values
nan_count = np.isnan(X_109).sum()
print(f"\nTotal NaNs across 109 CpGs in 3,493 participants: {nan_count}")
if nan_count > 0:
    print("Applying KNN imputation (k=5)...")
    imputer = KNNImputer(n_neighbors=5)
    X_109 = imputer.fit_transform(X_109)
    print("KNN imputation complete. NaNs remaining:", np.isnan(X_109).sum())

# 5. Apply ComBat batch correction using neuroCombat
print("\nApplying ComBat batch correction across Gen 3, JHU, UMN via neuroCombat...")
covars = fhs_df[["batch", "age", "sex"]].copy()
res_combat = neuroCombat(
    dat=X_109.T, # neuroCombat expects features x samples
    covars=covars,
    batch_col="batch"
)
X_109_combat = res_combat["data"].T
print("ComBat complete. Output shape:", X_109_combat.shape)

df_restored_combat = pd.DataFrame(X_109_combat, columns=cpgs_109)
df_restored_combat["patient_id"] = fhs_df["patient_id"].values

# 6. Load existing 829 GrimAge CpGs from FHS_methylation_merged.parquet
print("\nLoading existing 829 GrimAge CpGs from FHS parquet...")
FHS_PARQUET = str(config.DATA_DIR / "FHS_methylation_merged.parquet" if (config.DATA_DIR / "FHS_methylation_merged.parquet").exists() else config.RAW_DIR / "FHS_methylation_merged.parquet")
PYAGING_DATA = str(config.PYAGING_DIR)
model = torch.load(os.path.join(PYAGING_DATA, "grimage.pt"), map_location="cpu", weights_only=False)
grimage_cpgs = [f for f in model.features if f.startswith("cg")]

limit = 2**31 - 1
fhs_pf = pq.ParquetFile(FHS_PARQUET, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
fhs_cols = set(fhs_pf.schema.names)
present_fhs_829 = [c for c in grimage_cpgs if c in fhs_cols]

fhs_829_df = pd.read_parquet(
    FHS_PARQUET,
    columns=["Share_ID", "age", "sex"] + present_fhs_829,
    engine="pyarrow",
    thrift_string_size_limit=limit,
    thrift_container_size_limit=limit
).rename(columns={"Share_ID": "patient_id"})
fhs_829_df = fhs_829_df.set_index("patient_id").loc[fhs_df["patient_id"]].reset_index()

# 7. Predict Original GrimAge vs Restored GrimAge with pyaging
print("\nScoring Original GrimAge (829 CpGs present, 201 imputed with ref values)...")
feats_orig = present_fhs_829 + ["age"]
adata_orig = ad.AnnData(X=fhs_829_df[feats_orig].values)
adata_orig.var_names = feats_orig
pya.pred.predict_age(adata_orig, clock_names=["grimage"], dir=PYAGING_DATA, verbose=False)
grimage_orig = adata_orig.obs["grimage"].values

print("\nScoring Restored GrimAge (938 CpGs present, only 92 imputed with ref values)...")
merged_fhs = pd.merge(fhs_829_df, df_restored_combat, on="patient_id")
all_938_cpgs = present_fhs_829 + cpgs_109
feats_restored = all_938_cpgs + ["age"]
adata_rest = ad.AnnData(X=merged_fhs[feats_restored].values)
adata_rest.var_names = feats_restored
pya.pred.predict_age(adata_rest, clock_names=["grimage"], dir=PYAGING_DATA, verbose=False)
grimage_rest = adata_rest.obs["grimage"].values

# Compare scores
r_fhs = np.corrcoef(grimage_orig, grimage_rest)[0, 1]
diff_fhs = grimage_rest - grimage_orig
print(f"\n================== FHS Score Comparison ==================")
print(f"New number of available CpGs: {len(all_938_cpgs)} / {len(grimage_cpgs)} ({len(all_938_cpgs)/len(grimage_cpgs)*100:.2f}%)")
print(f"Pearson correlation r: {r_fhs:.5f}")
print(f"Mean difference (restored - original): {np.mean(diff_fhs):.4f} +/- {np.std(diff_fhs, ddof=1):.4f} years")

# 8. Cox Proportional Hazards Models (Age- and Sex-Adjusted)
master_df = fhs_df[["patient_id", "age", "sex", "event", "time"]].copy()
master_df["grimage_orig"] = grimage_orig
master_df["grimage_rest"] = grimage_rest

# Standard deviation on Train split
train_mask = master_df["patient_id"].isin(split_ids["Train"])
sd_train_orig = master_df.loc[train_mask, "grimage_orig"].std()
sd_train_rest = master_df.loc[train_mask, "grimage_rest"].std()
print(f"\nFHS Train SD (Original 829 CpGs): {sd_train_orig:.4f} years")
print(f"FHS Train SD (Restored 938 CpGs): {sd_train_rest:.4f} years")

# Express standardized predictors per Train SD
master_df["grimage_orig_z"] = master_df["grimage_orig"] / sd_train_orig
master_df["grimage_rest_z"] = master_df["grimage_rest"] / sd_train_rest

results = []

def run_cox_split(split_name, pred_col, clock_label, scale_label):
    mask = master_df["patient_id"].isin(split_ids[split_name])
    df_s = master_df.loc[mask, [pred_col, "age", "sex", "time", "event"]].copy()
    cph = CoxPHFitter()
    cph.fit(df_s, duration_col="time", event_col="event")
    row = cph.summary.loc[pred_col]
    coef = row["coef"]
    se = row["se(coef)"]
    hr = row["exp(coef)"]
    ci_lower = row["exp(coef) lower 95%"]
    ci_upper = row["exp(coef) upper 95%"]
    p = row["p"]
    return {
        "Split": split_name,
        "Clock_Version": clock_label,
        "Scale": scale_label,
        "n_samples": len(df_s),
        "n_deaths": int(df_s["event"].sum()),
        "coef": coef,
        "se": se,
        "HR": hr,
        "HR_lower95": ci_lower,
        "HR_upper95": ci_upper,
        "p_value": p,
        "HR_string": f"{hr:.2f} ({ci_lower:.2f} - {ci_upper:.2f})"
    }

for s_name in ["Train", "Val", "Test"]:
    # Per 1-Year
    results.append(run_cox_split(s_name, "grimage_orig", "Original (829 CpGs)", "per 1-Year"))
    results.append(run_cox_split(s_name, "grimage_rest", "Restored (938 CpGs)", "per 1-Year"))
    # Per 1-SD (Train SD)
    results.append(run_cox_split(s_name, "grimage_orig_z", "Original (829 CpGs)", "per 1-SD"))
    results.append(run_cox_split(s_name, "grimage_rest_z", "Restored (938 CpGs)", "per 1-SD"))

res_df = pd.DataFrame(results)
out_csv = str(config.REVIEWS_DIR / "grimage_probe_audit" / "restored_grimage_cox_results.csv")
res_df.to_csv(out_csv, index=False)
print(f"\nSaved restoration Cox results to {out_csv}")

print("\n================== AGE- AND SEX-ADJUSTED COX RESULTS ==================")
print(res_df[["Split", "Clock_Version", "Scale", "HR_string", "p_value"]].to_string(index=False))
