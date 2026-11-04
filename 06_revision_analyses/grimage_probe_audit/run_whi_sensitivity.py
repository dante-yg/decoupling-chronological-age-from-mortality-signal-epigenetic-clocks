import os
import sys
import torch
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# Compatibility shim
np.unicode_ = np.str_
import importlib.metadata, importlib_metadata
_orig = importlib.metadata.version
importlib.metadata.version = lambda p: np.__version__ if p == 'numpy' else _orig(p)
importlib_metadata.version = importlib.metadata.version

import anndata as ad
import pyaging as pya
from scipy.stats import pearsonr
from lifelines import CoxPHFitter

# File paths
WHI_SUBSET_CSV = r"C:\sharing\base_3\data\WHI_non_normalized.csv"
WHI_PARQUET = r"C:\Yugan_Gogul\datasets\combined\data\WHI_full_imputed.parquet"
FHS_PARQUET = r"C:\Yugan_Gogul\datasets\combined\data\FHS_methylation_merged.parquet"
PYAGING_DATA = r"C:\sharing\base_3\pyaging_data"
PRED_CSV = r"C:\sharing\base_3\predictions\whi_biological_clocks.csv"
OUTPUT_DIR = r"C:\Yugan_Gogul\datasets\combined\data\grimage_audit"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load WHI 196 subset metadata
df_sub = pd.read_csv(WHI_SUBSET_CSV)
print(f"WHI subset loaded: {df_sub.shape} subjects, {df_sub['event'].sum()} deaths.")

# 2. Identify GrimAge CpGs and FHS-absent CpGs
model = torch.load(os.path.join(PYAGING_DATA, "grimage.pt"), map_location="cpu", weights_only=False)
grimage_cpgs = [f for f in model.features if f.startswith("cg")]
print(f"GrimAge requires {len(grimage_cpgs)} CpGs (plus age and female).")

limit = 2**31 - 1
fhs_pf = pq.ParquetFile(FHS_PARQUET, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
fhs_cols = set(fhs_pf.schema.names)

whi_pf = pq.ParquetFile(WHI_PARQUET, thrift_string_size_limit=limit, thrift_container_size_limit=limit)
whi_cols = set(whi_pf.schema.names)

present_fhs = set([c for c in grimage_cpgs if c in fhs_cols])
absent_fhs = set([c for c in grimage_cpgs if c not in fhs_cols])
print(f"FHS: present={len(present_fhs)}, absent={len(absent_fhs)}")

present_whi = [c for c in grimage_cpgs if c in whi_cols]
absent_whi = [c for c in grimage_cpgs if c not in whi_cols]
print(f"WHI: present={len(present_whi)}, absent={len(absent_whi)}")

# Masked WHI features: intersection of WHI present and FHS present
masked_whi_cpgs = [c for c in present_whi if c in present_fhs]
print(f"Common CpGs in both (Masked WHI): {len(masked_whi_cpgs)}")
print(f"Probes masked out in WHI: {len(present_whi) - len(masked_whi_cpgs)}")

# 3. Load WHI methylation data for 196 subjects
sub_ids = df_sub["SUBJECT_ID"].tolist()
meta_cols = ["Subject_ID", "sex ", "age", "event ", "time"]
load_cols = list(set(meta_cols + present_whi))

print("Loading WHI parquet for required columns...")
whi_df = pd.read_parquet(
    WHI_PARQUET,
    columns=load_cols,
    engine="pyarrow",
    thrift_string_size_limit=limit,
    thrift_container_size_limit=limit
)
whi_df = whi_df.rename(columns={"Subject_ID": "patient_id", "sex ": "sex", "event ": "event"})
whi_df_196 = whi_df.set_index("patient_id").loc[sub_ids].reset_index()
print(f"WHI 196 methylation matrix loaded: {whi_df_196.shape}")

# 4. Predict Unmasked GrimAge (with age feature)
print("Scoring unmasked GrimAge with pyaging...")
unmasked_features = present_whi + ["age"]
adata_unmasked = ad.AnnData(X=whi_df_196[unmasked_features].values)
adata_unmasked.var_names = unmasked_features
pya.pred.predict_age(adata_unmasked, clock_names=["grimage"], dir=PYAGING_DATA, verbose=False)
grimage_unmasked = adata_unmasked.obs["grimage"].values

# Check against saved predictions
df_saved_pred = pd.read_csv(PRED_CSV)
saved_196 = pd.merge(df_sub[["SUBJECT_ID"]], df_saved_pred, left_on="SUBJECT_ID", right_on="patient_id")
diff_unmasked = np.max(np.abs(grimage_unmasked - saved_196["grimage"].values))
print(f"Max absolute difference from published/saved WHI predictions: {diff_unmasked:.6e}")

# 5. Predict Masked GrimAge (with age feature, but only common CpGs)
print("Scoring masked GrimAge with pyaging (masking out FHS-absent probes)...")
masked_features = masked_whi_cpgs + ["age"]
adata_masked = ad.AnnData(X=whi_df_196[masked_features].values)
adata_masked.var_names = masked_features
pya.pred.predict_age(adata_masked, clock_names=["grimage"], dir=PYAGING_DATA, verbose=False)
grimage_masked = adata_masked.obs["grimage"].values

# 6. Compare Unmasked vs Masked GrimAge
r_val, p_val = pearsonr(grimage_unmasked, grimage_masked)
diff = grimage_masked - grimage_unmasked
mean_diff = np.mean(diff)
sd_diff = np.std(diff, ddof=1)
mae = np.mean(np.abs(diff))

sd_unmasked_whi = np.std(grimage_unmasked, ddof=1)
sd_masked_whi = np.std(grimage_masked, ddof=1)
fhs_train_sd = 11.353572 # from Table 3 rebuild summary

print("\n================== Comparison Metrics ==================")
print(f"Pearson r: {r_val:.5f} (p = {p_val:.2e})")
print(f"Mean difference (masked - unmasked): {mean_diff:.4f} +/- {sd_diff:.4f} years")
print(f"Mean absolute error (MAE): {mae:.4f} years")
print(f"WHI unmasked GrimAge SD: {sd_unmasked_whi:.4f} years")
print(f"WHI masked GrimAge SD:   {sd_masked_whi:.4f} years")
print(f"FHS Training split SD:   {fhs_train_sd:.4f} years")

# 7. Survival Analysis (Cox Proportional Hazards)
surv_df = df_sub[["SUBJECT_ID", "age", "sex", "event", "time"]].copy()
surv_df["grimage_unmasked"] = grimage_unmasked
surv_df["grimage_masked"] = grimage_masked

# Standardized predictors
surv_df["grimage_unmasked_z_whi"] = (grimage_unmasked - np.mean(grimage_unmasked)) / sd_unmasked_whi
surv_df["grimage_masked_z_whi"]   = (grimage_masked - np.mean(grimage_masked)) / sd_masked_whi

surv_df["grimage_unmasked_z_fhs"] = grimage_unmasked / fhs_train_sd
surv_df["grimage_masked_z_fhs"]   = grimage_masked / fhs_train_sd

results = []

def run_cox(formula_cols, predictor_name, model_type, scale_name):
    cph = CoxPHFitter()
    cph.fit(surv_df[formula_cols], duration_col="time", event_col="event")
    row = cph.summary.loc[predictor_name]
    coef = row["coef"]
    se = row["se(coef)"]
    hr = row["exp(coef)"]
    ci_lower = row["exp(coef) lower 95%"]
    ci_upper = row["exp(coef) upper 95%"]
    p = row["p"]
    return {
        "Clock_Version": "Unmasked (Original)" if "unmasked" in predictor_name else "Masked (Sensitivity)",
        "Adjustment": model_type,
        "Scale": scale_name,
        "coef": coef,
        "se": se,
        "HR": hr,
        "HR_lower95": ci_lower,
        "HR_upper95": ci_upper,
        "p_value": p,
        "ci_string": f"{hr:.2f} ({ci_lower:.2f} - {ci_upper:.2f})"
    }

# Unadjusted models
results.append(run_cox(["grimage_unmasked", "time", "event"], "grimage_unmasked", "Unadjusted", "per 1-Year"))
results.append(run_cox(["grimage_masked", "time", "event"], "grimage_masked", "Unadjusted", "per 1-Year"))

results.append(run_cox(["grimage_unmasked_z_whi", "time", "event"], "grimage_unmasked_z_whi", "Unadjusted", "per 1-SD (WHI SD)"))
results.append(run_cox(["grimage_masked_z_whi", "time", "event"], "grimage_masked_z_whi", "Unadjusted", "per 1-SD (WHI SD)"))

results.append(run_cox(["grimage_unmasked_z_fhs", "time", "event"], "grimage_unmasked_z_fhs", "Unadjusted", "per 1-SD (FHS-Train SD)"))
results.append(run_cox(["grimage_masked_z_fhs", "time", "event"], "grimage_masked_z_fhs", "Unadjusted", "per 1-SD (FHS-Train SD)"))

# Age-adjusted models
results.append(run_cox(["grimage_unmasked", "age", "time", "event"], "grimage_unmasked", "Age-Adjusted", "per 1-Year"))
results.append(run_cox(["grimage_masked", "age", "time", "event"], "grimage_masked", "Age-Adjusted", "per 1-Year"))

results.append(run_cox(["grimage_unmasked_z_whi", "age", "time", "event"], "grimage_unmasked_z_whi", "Age-Adjusted", "per 1-SD (WHI SD)"))
results.append(run_cox(["grimage_masked_z_whi", "age", "time", "event"], "grimage_masked_z_whi", "Age-Adjusted", "per 1-SD (WHI SD)"))

results.append(run_cox(["grimage_unmasked_z_fhs", "age", "time", "event"], "grimage_unmasked_z_fhs", "Age-Adjusted", "per 1-SD (FHS-Train SD)"))
results.append(run_cox(["grimage_masked_z_fhs", "age", "time", "event"], "grimage_masked_z_fhs", "Age-Adjusted", "per 1-SD (FHS-Train SD)"))

res_df = pd.DataFrame(results)
res_path = os.path.join(OUTPUT_DIR, "sensitivity_results.csv")
res_df.to_csv(res_path, index=False)
print(f"\nSaved sensitivity results to {res_path}")

# Print table
print("\n" + res_df[["Clock_Version", "Adjustment", "Scale", "HR", "HR_lower95", "HR_upper95", "p_value", "ci_string"]].to_string(index=False))
