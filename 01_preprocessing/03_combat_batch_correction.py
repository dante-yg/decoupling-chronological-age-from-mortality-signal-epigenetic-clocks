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
import pyarrow.parquet as pq
import pyarrow as pa

path_whi = config.DATA_DIR / "WHI_full_imputed.parquet" if (config.DATA_DIR / "WHI_full_imputed.parquet").exists() else config.RAW_DIR / "WHI_full_imputed.parquet"
path_fhs = config.DATA_DIR / "FHS_methylation_merged.parquet" if (config.DATA_DIR / "FHS_methylation_merged.parquet").exists() else config.RAW_DIR / "FHS_methylation_merged.parquet"

# Big limits for schema/metadata
KW = {
    "thrift_string_size_limit": 1_000_000_000,
    "thrift_container_size_limit": 1_000_000_000,
}

# ---- WHI ----
whi_table = pq.read_table(path_whi, **KW)
WHI = whi_table.to_pandas(split_blocks=True, self_destruct=True)
print("WHI shape:", WHI.shape)

# ---- FHS ----
fhs_table = pq.read_table(path_fhs, **KW)
FHS = fhs_table.to_pandas(split_blocks=True, self_destruct=True)
print("FHS shape:", FHS.shape)


# CELL 2 (code)
WHI

# CELL 3 (code)
FHS

# CELL 4 (code)
import pandas as pd

# --- Helper: find a column in a case/space-insensitive way ---
def find_col(df, target):
    target = target.lower()
    matches = [c for c in df.columns if c.strip().lower() == target]
    if not matches:
        raise KeyError(f"Could not find column matching '{target}' in WHI. "
                       f"First few columns: {list(df.columns[:20])}")
    return matches[0]

# =====================================================
# 1. Identify CpG columns and common set
# =====================================================
fhs_cpgs = [c for c in FHS.columns if c.startswith("cg")]
whi_cpgs = [c for c in WHI.columns if c.startswith("cg")]

common_cpgs = sorted(set(fhs_cpgs).intersection(whi_cpgs))
print("FHS CpGs:", len(fhs_cpgs))
print("WHI CpGs:", len(whi_cpgs))
print("Common CpGs:", len(common_cpgs))

# =====================================================
# 2. Harmonize WHI metadata → FHS style
# =====================================================

# Find original WHI metadata column names robustly
whi_sample_id   = find_col(WHI, "sample_id")
whi_subject_id  = find_col(WHI, "subject_id")
whi_age         = find_col(WHI, "age")
whi_sex         = find_col(WHI, "sex")
whi_event       = find_col(WHI, "event")
whi_time        = find_col(WHI, "time")

# Rename to match FHS
WHI.rename(columns={
    whi_sample_id:  "Share_ID",
    whi_subject_id: "Patient_ID",
    whi_age:        "age",
    whi_sex:        "sex",
    whi_event:      "event",
    whi_time:       "time",
}, inplace=True)

# Add batch column
WHI["batch"] = "WHI"

# =====================================================
# 3. Define common metadata columns in desired order
# =====================================================
meta_cols = ["Share_ID", "Patient_ID", "sex", "age", "event", "batch", "time"]

# Sanity check for FHS
missing_in_fhs = [c for c in meta_cols if c not in FHS.columns]
if missing_in_fhs:
    raise KeyError(f"These metadata columns are missing in FHS: {missing_in_fhs}")

# Sanity check for WHI
missing_in_whi = [c for c in meta_cols if c not in WHI.columns]
if missing_in_whi:
    raise KeyError(f"These metadata columns are missing in WHI after renaming: {missing_in_whi}")

# =====================================================
# 4. Restrict BOTH WHI and FHS to meta_cols + common CpGs
# =====================================================
WHI = WHI[meta_cols + common_cpgs]
print("WHI shape after harmonization:", WHI.shape)

FHS = FHS[meta_cols + common_cpgs]
print("FHS shape after harmonization:", FHS.shape)

# =====================================================
# 5. Combine
# =====================================================
combined_df = pd.concat([FHS, WHI], axis=0, ignore_index=True)
print("Combined shape:", combined_df.shape)


# CELL 5 (code)
combined_df

# CELL 6 (code)
import os
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- 1. Fix only the problematic metadata dtypes ----------
# We won't touch CpGs at all.

# Work directly on combined_df (no copy) unless you really need the original untouched
combined_df["Share_ID"] = combined_df["Share_ID"].astype(str)
combined_df["batch"]    = combined_df["batch"].astype(str)

# ---------- 2. Save to Parquet with PyArrow ----------
out_dir = config.DATA_DIR
os.makedirs(out_dir, exist_ok=True)

parquet_path = os.path.join(out_dir, "FHS_WHI_combined_float64.parquet")

print("Converting DataFrame → Arrow table...")
table = pa.Table.from_pandas(combined_df, preserve_index=False, safe=False)

print("Writing Parquet file...")
pq.write_table(
    table,
    parquet_path,
    compression="zstd",
    compression_level=3,
)

print("Saved Parquet at:")
print(parquet_path)


# CELL 7 (code)
import pandas as pd 
import numpy as np

df = pd.read_parquet(config.DATA_DIR / "FHS_WHI_combined_float64.parquet")

# CELL 8 (code)
df

# CELL 9 (code)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# ============================================================
# 0. Let BLAS (NumPy / sklearn) use all CPU cores
#    (must be set before heavy linear algebra kicks in)
# ============================================================
n_cores = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
print(f"Using up to {n_cores} CPU cores (BLAS threads).")

# df is already loaded as:
# df = pd.read_parquet(config.DATA_DIR / "FHS_WHI_combined_float64.parquet")

# ============================================================
# 1. Identify CpG columns (all of them)
# ============================================================
cols_np = df.columns.to_numpy(dtype=str)
cpg_mask = np.char.startswith(cols_np, "cg")
cpg_cols = df.columns[cpg_mask]

print(f"Total columns      : {len(df.columns)}")
print(f"Number of CpG cols : {len(cpg_cols)}")

# ============================================================
# 2. Build CpG matrix X and batch labels
# ============================================================
# Use float32 to cut memory in half but DO NOT modify df
X = df[cpg_cols].to_numpy(dtype=np.float32)   # shape: (n_samples, n_cpgs)
batch_labels = df["batch"].astype(str).values

print("CpG matrix shape   :", X.shape)
print("Dtype              :", X.dtype)

# ============================================================
# 3. Handle missing values (NaNs) — mean imputation per CpG
#    (TruncatedSVD / PCA cannot handle NaNs)
# ============================================================
if np.isnan(X).any():
    print("NaNs detected in CpG matrix → imputing column means...")
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = col_means[inds[1]]
else:
    print("No NaNs detected in CpG matrix.")

# ============================================================
# 4. Run TruncatedSVD as PCA on all CpGs
#    - This is suitable for very wide matrices (features >> samples)
#    - n_components = 2 for 2D plot
# ============================================================
n_components = 2
print("Running TruncatedSVD (PCA) on all CpGs... this may take some time.")

svd = TruncatedSVD(
    n_components=n_components,
    n_iter=7,           # more iterations = better accuracy, more time
    random_state=42
)
pcs = svd.fit_transform(X)   # shape: (n_samples, 2)

explained = svd.explained_variance_ratio_
print("Explained variance ratio:", explained)

pc1_var = explained[0] * 100
pc2_var = explained[1] * 100

# ============================================================
# 5. Plot PC1 vs PC2 colored by batch
# ============================================================
plt.figure(figsize=(8, 6))

unique_batches = np.unique(batch_labels)
for b in unique_batches:
    mask = batch_labels == b
    plt.scatter(
        pcs[mask, 0],
        pcs[mask, 1],
        s=10,
        alpha=0.7,
        label=b
    )

plt.xlabel(f"PC1 ({pc1_var:.2f}% var)")
plt.ylabel(f"PC2 ({pc2_var:.2f}% var)")
plt.title("PCA (TruncatedSVD) on ALL CpGs, colored by batch")
plt.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.show()


# CELL 10 (code)
epi = pd.read_csv(config.DATA_DIR / "6k probes and coefficients.csv" if (config.DATA_DIR / "6k probes and coefficients.csv").exists() else config.RAW_DIR / "6k probes and coefficients.csv", low_memory=False)

# CELL 11 (code)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================================
# 0. Starting point:
#    - df: combined FHS + WHI methylation data (already loaded)
#      columns: Share_ID, Patient_ID, sex, age, event, batch, time, cg...
# ==========================================================

print("Combined df shape:", df.shape)
print("Unique batches:", df["batch"].unique())

# ==========================================================
# 1. Load Epiclock coefficient table
# ==========================================================
epi_path = config.DATA_DIR / "6k probes and coefficients.csv" if (config.DATA_DIR / "6k probes and coefficients.csv").exists() else config.RAW_DIR / "6k probes and coefficients.csv"
epi = pd.read_csv(epi_path, low_memory=False)

# ==========================================================
# 2. Extract intercept from epi table
#    (same logic you used before)
# ==========================================================
intercept = (
    epi.loc[epi["Unnamed: 3"] == "Intercept", "Unnamed: 4"]
    .astype(float)
    .iloc[0]
)
print("Intercept:", intercept)

# ==========================================================
# 3. Clean coefficient table (CpG + coef only)
# ==========================================================
coef_df = epi.loc[epi["Coefficient Estimate"].notna(), ["Columns", "Coefficient Estimate"]].copy()
coef_df.columns = ["CpG", "coef"]

# Drop NaN and duplicate CpGs
coef_df = coef_df.dropna(subset=["coef"]).drop_duplicates(subset=["CpG"])

print("Number of coefficient CpGs (raw):", len(coef_df))

# Align CpGs with available CpGs in df
available_cpgs = set(df.columns)
coef_df = coef_df[coef_df["CpG"].isin(available_cpgs)].copy()

print("CpGs found in combined df:", len(coef_df))

# ==========================================================
# 4. Build design matrix X and beta vector
# ==========================================================
# Design matrix X: all rows × selected CpGs
X = df[coef_df["CpG"]]

# Beta vector in aligned order
beta = coef_df.set_index("CpG")["coef"]
X = X[beta.index]  # ensure same CpG order as beta

# ==========================================================
# 5. Compute epi-age and epi-age acceleration for ALL samples
# ==========================================================
epi_age = intercept + X.mul(beta, axis=1).sum(axis=1)

df["epi_age"] = epi_age
df["epi_age_accel"] = df["epi_age"] - df["age"]

print("Computed epi_age. Example rows:")
id_col = "Share_ID" if "Share_ID" in df.columns else (
    "Patient_ID" if "Patient_ID" in df.columns else df.columns[0]
)
print(df[[id_col, "batch", "age", "epi_age", "epi_age_accel"]].head())

# ==========================================================
# 6. Metrics helper
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

# ==========================================================
# 7. Define FHS-like vs WHI subsets
#    FHS side = Gen 3 + JHU + UMN
# ==========================================================
fhs_batches = ["Gen 3", "JHU", "UMN"]

mask_fhs = df["batch"].isin(fhs_batches)
mask_whi = df["batch"] == "WHI"

FHS = df[mask_fhs].copy()
WHI = df[mask_whi].copy()

print("\nSubset sizes:")
print("FHS-like (Gen3/JHU/UMN):", FHS.shape)
print("WHI                   :", WHI.shape)

# ==========================================================
# 8. Metrics for FHS and WHI
# ==========================================================
print_metrics("FHS-like (Gen3/JHU/UMN)", FHS["age"], FHS["epi_age"])
print_metrics("WHI", WHI["age"], WHI["epi_age"])

# ==========================================================
# 9. Scatter plot: Chronological age vs epi-age (FHS-like)
# ==========================================================
age_true_fhs = FHS["age"]
age_pred_fhs = FHS["epi_age"]

min_age_fhs = min(age_true_fhs.min(), age_pred_fhs.min())
max_age_fhs = max(age_true_fhs.max(), age_pred_fhs.max())

plt.figure(figsize=(6, 6))
plt.scatter(age_true_fhs, age_pred_fhs, s=10, alpha=0.5)
plt.plot([min_age_fhs, max_age_fhs], [min_age_fhs, max_age_fhs], linestyle="--")

plt.xlim(min_age_fhs, max_age_fhs)
plt.ylim(min_age_fhs, max_age_fhs)
plt.xlabel("Chronological age (FHS: Gen3/JHU/UMN)")
plt.ylabel("Predicted epi-age")
plt.title("FHS-like – epi-age vs Chronological age")
plt.grid(False)
plt.tight_layout()
plt.show()

# ==========================================================
# 10. Scatter plot: Chronological age vs epi-age (WHI)
# ==========================================================
age_true_whi = WHI["age"]
age_pred_whi = WHI["epi_age"]

min_age_whi = min(age_true_whi.min(), age_pred_whi.min())
max_age_whi = max(age_true_whi.max(), age_pred_whi.max())

plt.figure(figsize=(6, 6))
plt.scatter(age_true_whi, age_pred_whi, s=10, alpha=0.5)
plt.plot([min_age_whi, max_age_whi], [min_age_whi, max_age_whi], linestyle="--")

plt.xlim(min_age_whi, max_age_whi)
plt.ylim(min_age_whi, max_age_whi)
plt.xlabel("Chronological age (WHI)")
plt.ylabel("Predicted epi-age")
plt.title("WHI – epi-age vs Chronological age")
plt.grid(False)
plt.tight_layout()
plt.show()


# CELL 12 (code)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from neuroCombat import neuroCombat

# =========================================================================
# ASSUMPTION: The full DataFrame 'df' is ALREADY LOADED in your environment.
# It MUST contain the columns 'batch', 'age', 'sex', and all 'cg...' sites.
# =========================================================================

def check_for_confounding(df_data, batch_col, potential_covars):
    """
    Checks for covariates that are perfectly confounded (constant within every batch),
    which is the root cause of the persistent LinAlgError.
    Also handles smart typing of covariates (e.g., treating 0/1 'sex' as categorical).
    """
    covars = df_data[[batch_col] + potential_covars].copy()
    categorical_cols = []
    continuous_cols = []
    
    # 1. Type Assignment and Initial Variance Check
    for col in potential_covars:
        if covars[col].nunique() <= 2 and covars[col].dtype != 'object':
            categorical_cols.append(col)
        elif covars[col].var() > 1e-8:
            continuous_cols.append(col)
        else:
            print(f"[Warning] Skipping '{col}': Variance is too low (effectively constant).")
    
    # 2. Confounding Check
    final_covars = [c for c in categorical_cols + continuous_cols]
    confounded_to_drop = []

    for col in final_covars:
        is_confounded = True
        for batch in covars[batch_col].unique():
            sub_covariate = covars.loc[covars[batch_col] == batch, col].dropna()
            # If the unique count is 1 for ALL batches, it's confounded.
            if sub_covariate.nunique() > 1:
                is_confounded = False
                break
                
        if is_confounded:
            print(f"[Critical Warning] Covariate '{col}' is perfectly confounded with '{batch_col}'. Removing it.")
            confounded_to_drop.append(col)
            
    # 3. Apply drops
    for col in confounded_to_drop:
        if col in continuous_cols: continuous_cols.remove(col)
        if col in categorical_cols: categorical_cols.remove(col)

    all_final_cols = [batch_col] + continuous_cols + categorical_cols
    covars_clean = covars[all_final_cols].copy()

    return covars_clean, continuous_cols, categorical_cols

def run_neurocombat_robustly(df, batch_col, potential_covars):
    """Full pipeline with filtering, M-value conversion, and robust ComBat execution."""
    
    print("--- 1. Data Cleaning and M-Value Conversion ---")
    
    # 1.1. Identify and filter CpG columns
    cpg_cols_all = df.columns[df.columns.str.startswith("cg")]
    # Keep only CpGs with NO missing values across samples
    cpg_cols = cpg_cols_all[~df[cpg_cols_all].isna().any(axis=0)]
    
    # 1.2. Filter samples for NaN covariates
    covar_cols_initial = [batch_col] + potential_covars
    row_mask = df[covar_cols_initial].notna().all(axis=1)
    df_clean = df.loc[row_mask].reset_index(drop=True).copy()
    
    beta = df_clean[cpg_cols].to_numpy(dtype=np.float64) 
    
    # 1.3. Beta -> M values
    beta_clipped = np.clip(beta, 1e-6, 1 - 1e-6)
    M = np.log2(beta_clipped / (1.0 - beta_clipped)) 

    # 1.4. Filter M-values for zero variance (crucial for singularity)
    M_var = np.var(M, axis=0)
    non_zero_var_cpg_mask = M_var > 1e-8 
    M_filtered = M[:, non_zero_var_cpg_mask]
    cpg_cols_filtered = cpg_cols[non_zero_var_cpg_mask]
    M_t = M_filtered.T # (CpGs x samples)

    print(f"Total CpGs processed: {len(cpg_cols)}")
    print(f"CpGs dropped due to near-zero variance: {M.shape[1] - M_filtered.shape[1]}")
    print(f"Final M-value matrix shape (features x samples): {M_t.shape}")
    
    # 2. Covariate Configuration and Singularity Check
    covars_to_use, continuous_cols, categorical_cols = check_for_confounding(
        df_clean, batch_col, potential_covars
    )
    
    print("\n--- 2. Final ComBat Model Configuration ---")
    print(f"Covariates included: {covars_to_use.columns.tolist()}")
    print(f"Categorical columns (excluding batch): {categorical_cols}")
    print(f"Continuous columns: {continuous_cols}")
    print("------------------------------------------")
    
    # 3. Run ComBat with Iterative Singularity Fix
    
    # We use a loop to retry by removing covariates if needed
    current_covariates = [c for c in potential_covars if c in covars_to_use.columns]
    
    while True:
        try:
            # Attempt 1: Parametric mode
            print(f"\nRunning ComBat... (Model: ~ {batch_col} + {' + '.join(current_covariates)})")
            
            combat_res = neuroCombat(
                dat=M_t,
                covars=covars_to_use,
                batch_col=batch_col,
                categorical_cols=categorical_cols,
                continuous_cols=continuous_cols,
                parametric=True,
                mean_only=False,
            )
            print("\nComBat successfully completed with Parametric=True.")
            break # Success, exit loop

        except np.linalg.LinAlgError:
            print("\nLinAlgError in parametric mode. Falling back to non-parametric mode...")
            
            try:
                # Attempt 2: Non-parametric mode
                combat_res = neuroCombat(
                    dat=M_t,
                    covars=covars_to_use,
                    batch_col=batch_col,
                    categorical_cols=categorical_cols,
                    continuous_cols=continuous_cols,
                    parametric=False, # Fallback
                    mean_only=False,
                )
                print("\nComBat successfully completed with Parametric=False.")
                break # Success, exit loop
            
            except np.linalg.LinAlgError:
                # Attempt 3: If non-parametric also fails, remove a covariate and try again.
                if not current_covariates:
                    print("\nFATAL ERROR: Cannot resolve singularity. All non-batch covariates removed.")
                    return None, None, None, None # Complete Failure
                
                removed_covar = current_covariates.pop()
                print(f"\nFATAL LinAlgError persists. Removing '{removed_covar}' and retrying...")
                
                # Update covariate lists for the retry
                if removed_covar in continuous_cols: continuous_cols.remove(removed_covar)
                if removed_covar in categorical_cols: categorical_cols.remove(removed_covar)
                covars_to_use = covars_to_use.drop(columns=[removed_covar])

    # 4. M-values -> Corrected Beta values
    M_corrected_t = combat_res["data"]
    M_corrected = M_corrected_t.T 

    pow2M = np.exp2(M_corrected)
    beta_corrected = np.clip(pow2M / (1.0 + pow2M), 0.0, 1.0)
    
    return beta_corrected, df_clean.index, cpg_cols_filtered, row_mask

# =========================================================================
# 4. EXECUTION AND PCA VISUALIZATION
# =========================================================================

batch_col = "batch"
potential_covars = ["age", "sex"]

# Run the full robust pipeline
beta_corrected, sample_indices, cpg_cols_filtered, row_mask = run_neurocombat_robustly(
    df, batch_col, potential_covars
)

if beta_corrected is None:
    print("\nBatch correction failed and could not be completed. Check your original data structure.")
else:
    # 5. Build final corrected DataFrame (df_combat)
    df_combat = df.copy()
    rows_idx = df.index[row_mask].tolist()

    # Fill only the filtered CpGs with the corrected beta values
    for i, cpg in enumerate(cpg_cols_filtered):
        df_combat.loc[rows_idx, cpg] = beta_corrected[:, i]

    print(f"\nFinal corrected DataFrame shape (df_combat): {df_combat.shape}")

    # 6. PCA on Corrected Data for visualization
    print("\n--- Starting PCA Visualization ---")
    X = df_combat.loc[rows_idx, cpg_cols_filtered].to_numpy(dtype=np.float32) 
    batch_labels = df_combat.loc[rows_idx, "batch"].astype(str).values 

    # Clean up residual NaNs just in case
    if np.isnan(X).any():
        print("NaNs detected post-ComBat – imputing column means for safety.")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = col_means[inds[1]]

    # TruncatedSVD is faster and more memory-efficient for large, sparse data
    svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
    pcs = svd.fit_transform(X)

    explained = svd.explained_variance_ratio_
    pc1_var = explained[0] * 100
    pc2_var = explained[1] * 100
    print(f"Explained variance: PC1 ({pc1_var:.2f}%), PC2 ({pc2_var:.2f}%)")

    # 7. Plot PC1 vs PC2
    plt.figure(figsize=(8, 6))

    for b in np.unique(batch_labels):
        mask = (batch_labels == b)
        plt.scatter(
            pcs[mask, 0],
            pcs[mask, 1],
            s=10,
            alpha=0.7,
            label=b
        )

    plt.xlabel(f"PC1 ({pc1_var:.2f}% var)")
    plt.ylabel(f"PC2 ({pc2_var:.2f}% var)")
    plt.title("PCA on CpGs AFTER ComBat Correction")
    plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout()
    plt.show()
    print("PCA plot generated successfully.")

# CELL 13 (code)
import os
import pyarrow as pa
import pyarrow.parquet as pq

out_dir = config.DATA_DIR
os.makedirs(out_dir, exist_ok=True)

parquet_path = os.path.join(out_dir, "FHS_WHI_ComBat_batch_age_sex.parquet")

table = pa.Table.from_pandas(df_combat, preserve_index=False)
pq.write_table(table, parquet_path, compression="zstd", compression_level=3)

print("Saved:", parquet_path)


# CELL 14 (code)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================================
# 0. Starting point:
#    - df_combat: combined FHS + WHI methylation data AFTER ComBat
#      columns: Share_ID, Patient_ID, sex, age, event, batch, time, cg...
#    - Optional: row_mask from ComBat (boolean mask of rows that were corrected)
# ==========================================================

# If you want to use ONLY the rows that actually went into ComBat:
if 'row_mask' in globals():
    df_epi = df_combat.loc[row_mask].copy()
    print("Using only ComBat-corrected rows. Shape:", df_epi.shape)
else:
    df_epi = df_combat.copy()
    print("Using all rows in df_combat. Shape:", df_epi.shape)

print("Unique batches:", df_epi["batch"].unique())

# ==========================================================
# 1. Load Epiclock coefficient table
# ==========================================================
epi_path = config.DATA_DIR / "6k probes and coefficients.csv" if (config.DATA_DIR / "6k probes and coefficients.csv").exists() else config.RAW_DIR / "6k probes and coefficients.csv"
epi = pd.read_csv(epi_path, low_memory=False)

# ==========================================================
# 2. Extract intercept from epi table
# ==========================================================
intercept = (
    epi.loc[epi["Unnamed: 3"] == "Intercept", "Unnamed: 4"]
    .astype(float)
    .iloc[0]
)
print("Intercept:", intercept)

# ==========================================================
# 3. Clean coefficient table (CpG + coef only)
# ==========================================================
coef_df = epi.loc[epi["Coefficient Estimate"].notna(), ["Columns", "Coefficient Estimate"]].copy()
coef_df.columns = ["CpG", "coef"]

# Drop NaN and duplicate CpGs
coef_df = coef_df.dropna(subset=["coef"]).drop_duplicates(subset=["CpG"])

print("Number of coefficient CpGs (raw):", len(coef_df))

# Align CpGs with available CpGs in df_combat
available_cpgs = set(df_epi.columns)
coef_df = coef_df[coef_df["CpG"].isin(available_cpgs)].copy()

print("CpGs found in df_combat:", len(coef_df))

# ==========================================================
# 4. Build design matrix X and beta vector
# ==========================================================
X = df_epi[coef_df["CpG"]]

beta = coef_df.set_index("CpG")["coef"]
X = X[beta.index]  # ensure same CpG order as beta

# ==========================================================
# 5. Compute epi-age and epi-age acceleration for ALL samples
# ==========================================================
epi_age = intercept + X.mul(beta, axis=1).sum(axis=1)

df_epi["epi_age"] = epi_age
df_epi["epi_age_accel"] = df_epi["epi_age"] - df_epi["age"]

print("Computed epi_age. Example rows:")
id_col = "Share_ID" if "Share_ID" in df_epi.columns else (
    "Patient_ID" if "Patient_ID" in df_epi.columns else df_epi.columns[0]
)
print(df_epi[[id_col, "batch", "age", "epi_age", "epi_age_accel"]].head())

# ==========================================================
# 6. Metrics helper
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

# ==========================================================
# 7. Define FHS-like vs WHI subsets
#    FHS side = Gen 3 + JHU + UMN
# ==========================================================
fhs_batches = ["Gen 3", "JHU", "UMN"]

mask_fhs = df_epi["batch"].isin(fhs_batches)
mask_whi = df_epi["batch"] == "WHI"

FHS = df_epi[mask_fhs].copy()
WHI = df_epi[mask_whi].copy()

print("\nSubset sizes (ComBat-corrected):")
print("FHS-like (Gen3/JHU/UMN):", FHS.shape)
print("WHI                   :", WHI.shape)

# ==========================================================
# 8. Metrics for FHS and WHI
# ==========================================================
print_metrics("FHS-like (Gen3/JHU/UMN) – AFTER ComBat", FHS["age"], FHS["epi_age"])
print_metrics("WHI – AFTER ComBat", WHI["age"], WHI["epi_age"])

# ==========================================================
# 9. Scatter plot: Chronological age vs epi-age (FHS-like)
# ==========================================================
age_true_fhs = FHS["age"]
age_pred_fhs = FHS["epi_age"]

min_age_fhs = min(age_true_fhs.min(), age_pred_fhs.min())
max_age_fhs = max(age_true_fhs.max(), age_pred_fhs.max())

plt.figure(figsize=(6, 6))
plt.scatter(age_true_fhs, age_pred_fhs, s=10, alpha=0.5)
plt.plot([min_age_fhs, max_age_fhs], [min_age_fhs, max_age_fhs], linestyle="--")

plt.xlim(min_age_fhs, max_age_fhs)
plt.ylim(min_age_fhs, max_age_fhs)
plt.xlabel("Chronological age (FHS: Gen3/JHU/UMN)")
plt.ylabel("Predicted epi-age (ComBat)")
plt.title("FHS-like – epi-age vs Chronological age (AFTER ComBat)")
plt.grid(False)
plt.tight_layout()
plt.show()

# ==========================================================
# 10. Scatter plot: Chronological age vs epi-age (WHI)
# ==========================================================
age_true_whi = WHI["age"]
age_pred_whi = WHI["epi_age"]

min_age_whi = min(age_true_whi.min(), age_pred_whi.min())
max_age_whi = max(age_true_whi.max(), age_pred_whi.max())

plt.figure(figsize=(6, 6))
plt.scatter(age_true_whi, age_pred_whi, s=10, alpha=0.5)
plt.plot([min_age_whi, max_age_whi], [min_age_whi, max_age_whi], linestyle="--")

plt.xlim(min_age_whi, max_age_whi)
plt.ylim(min_age_whi, max_age_whi)
plt.xlabel("Chronological age (WHI)")
plt.ylabel("Predicted epi-age (ComBat)")
plt.title("WHI – epi-age vs Chronological age (AFTER ComBat)")
plt.grid(False)
plt.tight_layout()
plt.show()


