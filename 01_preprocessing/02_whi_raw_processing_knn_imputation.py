# CELL 0 (code)
import pandas as pd

# CELL 1 (code)
WHI_raw = pd.read_csv(r"C:/Yugan_Gogul/WHI/raw/BA23Methylation_dbGAP_0517-cg1.csv", low_memory=False)

# CELL 2 (code)
WHI_raw = WHI_raw.T

# CELL 3 (code)
WHI_raw = WHI_raw.drop(WHI_raw.index[1])

# CELL 4 (code)
WHI_raw

# CELL 5 (code)
ID = pd.read_csv(r"C:\Yugan_Gogul\WHI\WHI_metadata\processing\WHI_metadata.csv", low_memory = False)

# CELL 6 (code)
WHI_raw

# CELL 7 (code)
ID

# CELL 8 (code)
WHI_raw

# CELL 9 (code)
# Load your WHI_raw as you currently have it
df = WHI_raw.copy()



# CELL 10 (code)


# CELL 11 (code)


# CELL 12 (code)


# CELL 13 (code)
# Step 1: Extract true CpG column names from first row
true_columns = df.iloc[0].values.tolist()

# Step 2: Remove the first row (it contained CpG names)
df = df.iloc[1:].copy()

# Step 3: Set the row index as a new column called sample_ID
df = df.reset_index().rename(columns={"index": "sample_ID"})

# Step 4: Assign the correct CpG names as column headers
df.columns = ["sample_ID"] + true_columns

# Now df has:
#   sample_ID | cg00000029 | cg00000108 | ...
#   D279269-52    0.5923       0.9525      ...
#   D291159-49    0.6117       0.9541      ...

WHI_clean = df

# CELL 14 (code)
WHI_clean

# CELL 15 (code)


# CELL 16 (code)


# CELL 17 (code)
import pandas as pd

# Already loaded:
# WHI_clean
# ID

# Make sure sample_ID is string in both (usually already is, but just in case)
ID["sample_ID"] = ID["sample_ID"].astype(str)
WHI_clean["sample_ID"] = WHI_clean["sample_ID"].astype(str)

# Merge metadata to the left of CpGs
merged = ID.merge(
    WHI_clean,
    on="sample_ID",   # common key
    how="inner"       # keep only samples present in both
)

print("ID shape:", ID.shape)
print("WHI_clean shape:", WHI_clean.shape)
print("merged shape:", merged.shape)

# Optional: check first columns
print(merged.columns[:15])

# Save (CSV or Parquet)
merged.to_csv(r"C:\Yugan_Gogul\WHI\WHI_metadata\processing\WHI_with_metadata_and_CpGs.csv",
              index=False)

# Recommended for big matrix:
# merged.to_parquet(r"C:\Yugan_Gogul\WHI\WHI_metadata\processing\WHI_with_metadata_and_CpGs.parquet")


# CELL 18 (code)
merged

# CELL 19 (code)
# Get only CpG columns (exclude metadata)
cpg_cols = merged.columns[6:]   # because first 6 columns are metadata

# Count missing values per CpG
missing_per_cpg = merged[cpg_cols].isna().sum()

# How many CpGs have at least 1 missing?
num_cpg_with_missing = (missing_per_cpg > 0).sum()

print("Number of CpGs with missing values:", num_cpg_with_missing)


# CELL 20 (code)
import pandas as pd 
merged = pd.read_csv(r"C:\Yugan_Gogul\WHI\WHI_metadata\processing\WHI_with_metadata_and_CpGs.csv", low_memory=False)

# CELL 21 (code)


# CELL 22 (code)
merged

# CELL 23 (code)
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# -------- 1. Split metadata and CpG matrix by position --------
# First 6 columns = metadata, rest = CpGs
meta = merged.iloc[:, :6].copy()
cpg_df = merged.iloc[:, 6:].copy()

print("Metadata shape:", meta.shape)
print("CpG matrix shape (before filter):", cpg_df.shape)

# -------- 2. Drop CpGs with >10% missing --------
missing_frac = cpg_df.isna().mean(axis=0)          # fraction of NaNs per CpG
keep_mask = missing_frac <= 0.10                  # <= 10% missing
kept_cpg_cols = missing_frac.index[keep_mask]
dropped_cpg_cols = missing_frac.index[~keep_mask]

print("Total CpGs:", cpg_df.shape[1])
print("Kept CpGs (<=10% missing):", len(kept_cpg_cols))
print("Dropped CpGs (>10% missing):", len(dropped_cpg_cols))

# subset to kept CpGs and ensure float type
cpg_filtered = cpg_df[kept_cpg_cols].astype(float)


# CELL 24 (code)
# -------- 3. KNN imputation (k = 5) --------
imputer = KNNImputer(
    n_neighbors=5,
    weights="uniform",
    metric="nan_euclidean"
)

cpg_imputed_array = imputer.fit_transform(cpg_filtered)

# back to DataFrame with same index / columns
cpg_imputed = pd.DataFrame(
    cpg_imputed_array,
    index=merged.index,
    columns=kept_cpg_cols
)

# -------- 4. Reattach metadata --------
merged_imputed = pd.concat([meta, cpg_imputed], axis=1)

print("Final imputed shape:", merged_imputed.shape)

merged_imputed.to_parquet(r"C:\Yugan_Gogul\WHI\WHI_imputed_knn5.parquet")

# CELL 25 (code)
import pandas as pd

WHI = pd.read_parquet(
    r"C:\Yugan_Gogul\WHI\WHI_imputed_knn5.parquet",
    engine="fastparquet"
)


# CELL 26 (code)
WHI = WHI.drop(columns=["Unnamed: 0"], errors="ignore")



# CELL 27 (code)
WHI.to_csv(
    r"C:\Yugan_Gogul\WHI\WHI_imputed_knn5.csv",
    index=False
)

# CELL 28 (code)
import pandas as pd


# CELL 29 (code)
WHI = pd.read_csv(r"C:/Yugan_Gogul/datasets/WHI/processed_WHI/WHI_imputed_knn5.csv", low_memory = False)

# CELL 30 (code)
WHI

# CELL 31 (code)
import os
import pyarrow as pa
import pyarrow.parquet as pq

# Your loaded WHI dataframe
# WHI = pd.read_csv(...)

# -----------------------------------------------------------
# 1. Create output folder if it doesn't exist
# -----------------------------------------------------------
output_dir = r"C:\Yugan_Gogul\datasets\WHI\processed_parquet"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "WHI_full_imputed.parquet")

# -----------------------------------------------------------
# 2. Convert WHI -> Arrow Table (efficient)
# -----------------------------------------------------------
table = pa.Table.from_pandas(WHI, preserve_index=False)

# -----------------------------------------------------------
# 3. Write Parquet with compression
# -----------------------------------------------------------
pq.write_table(
    table,
    output_path,
    compression="zstd",        # best balance of speed + compression size
    compression_level=3        # default level, very fast
)

print("Saved Parquet file:")
print(output_path)


# CELL 32 (code)
meta_cols = ["sample_ID", "Subject_ID", "age", "sex", "event", "time"]

# Identify CpG columns
cpg_cols = WHI.columns.difference(meta_cols)

# Count how many CpG columns have any missing values
missing_cpg_count = WHI[cpg_cols].isna().any(axis=0).sum()

# List of CpG columns that have missing values
missing_cpg_list = WHI[cpg_cols].columns[WHI[cpg_cols].isna().any(axis=0)].tolist()

print("Number of CpG columns with missing values:", missing_cpg_count)
print("Example missing CpG column names:", missing_cpg_list[:10])   # print first 10


# CELL 33 (code)
epi = pd.read_csv(r"C:\Yugan_Gogul\Project_5\redundatn\Epiclock\6k probes and coefficients.csv", low_memory=False)

# CELL 34 (code)
epi

# CELL 35 (code)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================================
# 1. Extract intercept
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

# drop NaN and duplicates
coef_df = coef_df.dropna(subset=["coef"]).drop_duplicates(subset=["CpG"])

print("Number of coefficient CpGs:", len(coef_df))

# ==========================================================
# 3. Align CpGs with WHI
# ==========================================================
available_cpgs = set(WHI.columns)
coef_df = coef_df[coef_df["CpG"].isin(available_cpgs)].copy()

print("CpGs found in WHI:", len(coef_df))

# Create design matrix
X = WHI[coef_df["CpG"]]

# Create beta vector in aligned order
beta = coef_df.set_index("CpG")["coef"]
X = X[beta.index]

# ==========================================================
# 4. Compute epi_age
# ==========================================================
epi_age = intercept + X.mul(beta, axis=1).sum(axis=1)
WHI["epi_age"] = epi_age
WHI["epi_age_accel"] = WHI["epi_age"] - WHI["age"]

print("Computed epi_age. Example:")
print(WHI[["sample_ID", "age", "epi_age", "epi_age_accel"]].head())

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
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Pearson r: {r:.4f}")
    print(f"R²: {r2:.4f}")

print_metrics("WHI", WHI["age"], WHI["epi_age"])

# ==========================================================
# 6. Scatter plot (same style as your old project)
# ==========================================================

age_true = WHI["age"]
age_pred = WHI["epi_age"]

# identity line range
min_age = min(age_true.min(), age_pred.min())
max_age = max(age_true.max(), age_pred.max())

plt.figure(figsize=(6, 6))
plt.scatter(age_true, age_pred, s=10, alpha=0.5)
plt.plot([min_age, max_age], [min_age, max_age], linestyle="--")
plt.xlim(min_age, max_age)
plt.ylim(min_age, max_age)

plt.xlabel("Chronological age")
plt.ylabel("Predicted epi-age")
plt.title("WHI – epi-age vs Chronological age")
plt.grid(False)
plt.tight_layout()
plt.show()


# CELL 36 (code)


