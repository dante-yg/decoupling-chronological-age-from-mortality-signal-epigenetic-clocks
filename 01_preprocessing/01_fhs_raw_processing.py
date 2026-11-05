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

# CELL 0 (code)
import pandas as pd

# CELL 1 (code)
G3 = pd.read_csv(config.RAW_DIR / "FHS" / "raw" / "Gen3_common_probes.csv", low_memory = False)

# CELL 2 (code)
JHU = pd.read_csv(config.RAW_DIR / "FHS" / "raw" / "JHU_common_probes.csv", low_memory = False)

# CELL 3 (code)
UMN = pd.read_csv(config.RAW_DIR / "FHS" / "raw" / "UMN_common_probes.csv", low_memory = False)

# CELL 4 (code)
G3

# CELL 5 (code)
JHU

# CELL 6 (code)
UMN

# CELL 7 (code)
G3 = G3.drop(index="Index")


# CELL 8 (code)
G3

# CELL 9 (code)
# Drop the first row by position
UMN = UMN.iloc[1:, :]


# CELL 10 (code)
UMN = UMN.rename(index={"ProbeID": "Probe"})


# CELL 11 (code)
UMN

# CELL 12 (code)
JHU = JHU.reset_index().rename(columns={"index": "Probe"})


# CELL 13 (code)
JHU

# CELL 14 (code)
JHU.to_csv(config.RAW_DIR / "FHS" / "raw" / "JHU_clean.csv", index=False)

# CELL 15 (code)
JHU = pd.read_csv(config.RAW_DIR / "FHS" / "raw" / "JHU_clean.csv", low_memory = False)

# CELL 16 (code)
G3

# CELL 17 (code)
JHU

# CELL 18 (code)
import pandas as pd
G3 = pd.read_csv(config.RAW_DIR / "FHS" / "raw" / "G3_clean.csv", low_memory = False)

# CELL 19 (code)
JHU = pd.read_csv(config.RAW_DIR / "FHS" / "raw" / "JHU_clean.csv", low_memory = False)

# CELL 20 (code)
UMN = pd.read_csv(config.RAW_DIR / "FHS" / "raw" / "UMN_clean.csv", low_memory = False)

# CELL 21 (code)
G3

# CELL 22 (code)
JHU

# CELL 23 (code)
UMN

# CELL 24 (code)
meta_data = pd.read_csv(config.RAW_DIR / "FHS" / "FHS_dependencies" / "meta_data.csv", low_memory = False)

# CELL 25 (code)
meta_data

# CELL 26 (code)
import numpy as np
import pandas as pd
import gc

# ============================================================
# 1) Get CpG sets from each raw batch (only header row)
# ============================================================

def get_cpg_list(df_raw):
    """
    From a raw FHS-format matrix:
    - Row 0 = header: first cell 'Probe', then CpG names
    Returns: list of CpG IDs (strings) from that header.
    """
    return df_raw.iloc[0, 1:].tolist()  # skip first column ("Probe")

cpgs_g3  = set(get_cpg_list(G3))
cpgs_umn = set(get_cpg_list(UMN))
cpgs_jhu = set(get_cpg_list(JHU))

# Intersection: CpGs present in ALL three batches
common_cpgs = sorted(cpgs_g3 & cpgs_umn & cpgs_jhu)

print(f"Common CpGs across G3/UMN/JHU: {len(common_cpgs)}")

if len(common_cpgs) == 0:
    raise ValueError("No common CpGs found across G3, UMN, and JHU.")

# ============================================================
# 2) Helper: clean AND align one batch to common CpG order
# ============================================================

def clean_and_align_to_arrays(df_raw, common_cpgs, to_float32=True):
    """
    Takes a raw FHS-format methylation DataFrame and returns:
      - ids:    (n_samples,) int32 Share_ID array
      - values: (n_samples, n_common_cpgs) float32/float64 array
      - cpgs:   list of CpG names in the order used for columns
    Steps:
      - Use row 0 as header
      - Rename first column to 'Share_ID'
      - Restrict to common_cpgs in the same order for all batches
    """
    # Row 0: header row -> all column names
    header = df_raw.iloc[0, :].tolist()
    df = df_raw.iloc[1:, :].copy()
    df.columns = header
    
    # First column (originally 'Probe') actually holds Share_ID in data rows
    first_col_name = df.columns[0]
    df = df.rename(columns={first_col_name: "Share_ID"})
    
    # Ensure Share_ID is int32
    df["Share_ID"] = df["Share_ID"].astype(np.int32)
    
    # Restrict to common CpGs, in a fixed order
    # keep_cols = ["Share_ID"] + common_cpgs
    # Some CpGs might be missing in this batch due to quirks; double-check:
    available_cpgs = [c for c in common_cpgs if c in df.columns]
    if len(available_cpgs) != len(common_cpgs):
        missing = set(common_cpgs) - set(available_cpgs)
        raise ValueError(f"This batch is missing {len(missing)} common CpGs, e.g. {list(missing)[:5]}")
    
    keep_cols = ["Share_ID"] + available_cpgs
    df = df[keep_cols]
    
    # Extract arrays
    ids = df["Share_ID"].to_numpy(dtype=np.int32)
    
    if to_float32:
        values = df[available_cpgs].to_numpy(dtype=np.float32, copy=True)
    else:
        values = df[available_cpgs].to_numpy(dtype=np.float64, copy=True)
    
    # Clean up
    del df
    gc.collect()
    
    return ids, values, available_cpgs

# ============================================================
# 3) Process each batch: G3, UMN, JHU
# ============================================================

g3_ids,  g3_vals,  cpgs_g3_aligned  = clean_and_align_to_arrays(G3,  common_cpgs, to_float32=True)
del G3;  gc.collect()

umn_ids, umn_vals, cpgs_umn_aligned = clean_and_align_to_arrays(UMN, common_cpgs, to_float32=True)
del UMN; gc.collect()

jhu_ids, jhu_vals, cpgs_jhu_aligned = clean_and_align_to_arrays(JHU, common_cpgs, to_float32=True)
del JHU; gc.collect()

# Final sanity check: all aligned CpG lists must match (same order)
if not (cpgs_g3_aligned == cpgs_umn_aligned == cpgs_jhu_aligned):
    raise ValueError("Aligned CpG lists still differ between batches. Check headers and preprocessing.")

cpgs_final = cpgs_g3_aligned
print(f"Using {len(cpgs_final)} aligned CpGs.")

# ============================================================
# 4) Stack IDs and values using NumPy (fast)
# ============================================================

# Stack Share_IDs
all_ids = np.concatenate([g3_ids, umn_ids, jhu_ids], axis=0)
del g3_ids, umn_ids, jhu_ids
gc.collect()

# Stack CpG matrices (rows = samples)
all_vals = np.vstack([g3_vals, umn_vals, jhu_vals])
del g3_vals, umn_vals, jhu_vals
gc.collect()

print("Stacked methylation array shape:", all_vals.shape)

# ============================================================
# 5) Build the final methylation DataFrame ONCE
# ============================================================

methylation_all = pd.DataFrame(all_vals, columns=cpgs_final)
methylation_all.insert(0, "Share_ID", all_ids)

print("methylation_all shape:", methylation_all.shape)

# ============================================================
# 6) Merge with metadata on Share_ID
# ============================================================

# Ensure same dtype for key
meta_data["Share_ID"] = meta_data["Share_ID"].astype(np.int32)

final_df = meta_data.merge(methylation_all, on="Share_ID", how="inner")

# Optionally free methylation_all if you don't need it separately
del methylation_all
gc.collect()

print("final_df shape:", final_df.shape)


# CELL 27 (code)
final_df

# CELL 28 (code)
import os
import pyarrow as pa
import pyarrow.parquet as pq

output_dir = config.RAW_DIR / "FHS" / "processed"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "FHS_methylation_merged.parquet")

# Convert to Arrow table
table = pa.Table.from_pandas(final_df, preserve_index=False)

pq.write_table(
    table,
    output_path,
    compression="zstd",
    compression_level=3
)

print("Saved Parquet:", output_path)


# CELL 29 (code)
epi = pd.read_csv(config.DATA_DIR / "6k probes and coefficients.csv" if (config.DATA_DIR / "6k probes and coefficients.csv").exists() else config.RAW_DIR / "6k probes and coefficients.csv", low_memory=False)

# CELL 30 (code)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================================
# 0. Assume you already have:
#    - epi: coefficient table (same as in your WHI script)
#    - final_df: FHS metadata + CpGs (what we just built)
# ==========================================================

# Work on a copy for clarity
FHS = final_df.copy()

# ==========================================================
# 1. Extract intercept from epi table
#    (same logic you used before)
# ==========================================================
intercept = (
    epi.loc[epi["Unnamed: 3"] == "Intercept", "Unnamed: 4"]
    .astype(float)
    .iloc[0]
)
print("Intercept:", intercept)

# ==========================================================
# 2. Clean coefficient table (CpG + coef only)
# ==========================================================
coef_df = epi.loc[epi["Coefficient Estimate"].notna(), ["Columns", "Coefficient Estimate"]].copy()
coef_df.columns = ["CpG", "coef"]

# Drop NaN and duplicate CpGs
coef_df = coef_df.dropna(subset=["coef"]).drop_duplicates(subset=["CpG"])

print("Number of coefficient CpGs (raw):", len(coef_df))

# ==========================================================
# 3. Align CpGs with FHS CpG columns
# ==========================================================
available_cpgs = set(FHS.columns)
coef_df = coef_df[coef_df["CpG"].isin(available_cpgs)].copy()

print("CpGs found in FHS:", len(coef_df))

# Design matrix X: FHS rows × selected CpGs
X = FHS[coef_df["CpG"]]

# Beta vector in aligned order
beta = coef_df.set_index("CpG")["coef"]
X = X[beta.index]  # ensure same order

# ==========================================================
# 4. Compute epi-age and epi-age acceleration for FHS
# ==========================================================
epi_age = intercept + X.mul(beta, axis=1).sum(axis=1)

FHS["epi_age"] = epi_age
FHS["epi_age_accel"] = FHS["epi_age"] - FHS["age"]

print("Computed epi_age. Example rows:")
id_col = "Share_ID" if "Share_ID" in FHS.columns else (
    "Patient_ID" if "Patient_ID" in FHS.columns else FHS.columns[0]
)
print(FHS[[id_col, "age", "epi_age", "epi_age_accel"]].head())

# ==========================================================
# 5. Metrics function
# ==========================================================
def print_metrics(name, age_true, age_pred):
    age_true = np.asarray(age_true, float)
    age_pred = np.asarray(age_pred, float)

    mse = mean_squared_error(age_true, age_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(age_true, age_pred)
    r = np.corrcoef(age_true, age_pred)[0, 1]
    r2 = r**2

    print(f"\n=== Performance on {name} ===")
    print(f"N: {len(age_true)}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"Pearson r: {r:.4f}")
    print(f"R²:       {r2:.4f}")

print_metrics("FHS", FHS["age"], FHS["epi_age"])

# ==========================================================
# 6. Scatter plot: Chronological age vs epi-age (FHS)
# ==========================================================
age_true = FHS["age"]
age_pred = FHS["epi_age"]

min_age = min(age_true.min(), age_pred.min())
max_age = max(age_true.max(), age_pred.max())

plt.figure(figsize=(6, 6))
plt.scatter(age_true, age_pred, s=10, alpha=0.5)
plt.plot([min_age, max_age], [min_age, max_age], linestyle="--")

plt.xlim(min_age, max_age)
plt.ylim(min_age, max_age)
plt.xlabel("Chronological age (FHS)")
plt.ylabel("Predicted epi-age")
plt.title("FHS – epi-age vs Chronological age")
plt.grid(False)
plt.tight_layout()
plt.show()


# CELL 31 (code)


