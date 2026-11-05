# ==============================================================================
# Script: deconvolution.R
# Purpose: Estimate blood leukocyte composition from 450K DNA methylation for
#          the Framingham Heart Study (FHS) and Women's Health Initiative (WHI)
# Reference: EpiDISH (centDHSbloodDMC.m) & FlowSorted.Blood.450k (JaffeModelPars)
# Methods: RPC (Robust Partial Correlations; Teschendorff et al.) [Primary]
#          CP (Constrained Projection; Houseman et al.) [Secondary]
# Provenance: Full tracking of input matrices (pre- and post-ComBat)
# ==============================================================================

options(stringsAsFactors = FALSE)
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(EpiDISH)
  library(FlowSorted.Blood.450k)
  library(minfi)
  library(MASS)
  library(quadprog)
})

cat("==============================================================================\n")
cat("Starting Blood Leukocyte Deconvolution Analysis\n")
cat("Time:", as.character(Sys.time()), "\n")
cat("==============================================================================\n\n")

# ------------------------------------------------------------------------------
# Define directories and file paths
# ------------------------------------------------------------------------------
project_root <- Sys.getenv("EPICLOCK_ROOT", unset = ".")
data_dir     <- Sys.getenv("EPICLOCK_DATA_DIR", unset = file.path(project_root, "data"))
raw_dir      <- Sys.getenv("EPICLOCK_RAW_DIR", unset = file.path(data_dir, "raw"))
out_dir      <- Sys.getenv("EPICLOCK_CELL_DECONV_DIR", unset = file.path(project_root, "reviews", "cell_deconvolution"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

path_fhs_clean    <- file.path(out_dir, "FHS_reference_cpgs.parquet")
path_whi_clean    <- file.path(out_dir, "WHI_reference_cpgs.parquet")
path_combat_clean <- file.path(out_dir, "ComBat_reference_cpgs.parquet")

path_cbc <- Sys.getenv("EPICLOCK_CBC_PATH", unset = file.path(raw_dir, "FHS", "Phenotype", "phs000007.v33.pht002889.v5.p14.c1.l_fhslab_2011_m_0656s.HMB-IRB-MDS.txt", "phs000007.v33.pht002889.v5.p14.c1.l_fhslab_2011_m_0656s.HMB-IRB-MDS.txt"))
path_fhs_disease <- file.path(data_dir, "Disease_data_arranaged.xlsx")

# ------------------------------------------------------------------------------
# 1. Load Reference Matrices
# ------------------------------------------------------------------------------
cat("--- Step 1: Loading Deconvolution Reference Matrices ---\n")
data(centDHSbloodDMC.m)
ref_epidish <- centDHSbloodDMC.m
cat(sprintf("EpiDISH centDHSbloodDMC.m: %d probes x %d cell types (%s)\n",
            nrow(ref_epidish), ncol(ref_epidish), paste(colnames(ref_epidish), collapse = ", ")))

data(FlowSorted.Blood.450k.JaffeModelPars)
ref_flow <- FlowSorted.Blood.450k.JaffeModelPars
cat(sprintf("FlowSorted.Blood.450k.JaffeModelPars: %d probes x %d cell types (%s)\n\n",
            nrow(ref_flow), ncol(ref_flow), paste(colnames(ref_flow), collapse = ", ")))

all_ref_probes <- unique(c(rownames(ref_epidish), rownames(ref_flow)))

# ------------------------------------------------------------------------------
# 2. Reference Probe Coverage Assessment (Step 2)
# ------------------------------------------------------------------------------
cat("--- Step 2: Reference Probe Coverage Assessment ---\n")

coverage_df <- read.csv(file.path(out_dir, "reference_probe_coverage.csv"))
print(coverage_df[, 1:9])

# Check 10% threshold stop rule
if (any(coverage_df$Stop_Condition_Met)) {
  stop("FATAL: More than 10% of reference probes are missing entirely in one or more matrices.")
}

# ------------------------------------------------------------------------------
# 3. Load Matrices and Run Deconvolution (Step 3)
# ------------------------------------------------------------------------------

# Helper function to run deconvolution on a dataset
run_deconv_pipeline <- function(data_df, id_col, meta_cols, ref_epi, ref_flw) {
  # Separate metadata and CpG beta values
  sample_ids <- as.character(data_df[[id_col]])
  cpg_cols <- intersect(colnames(data_df), unique(c(rownames(ref_epi), rownames(ref_flw))))
  
  cat(sprintf("Extracting beta matrix: %d samples x %d available reference CpGs\n",
              nrow(data_df), length(cpg_cols)))
  
  beta_sub <- as.matrix(data_df[, cpg_cols, drop = FALSE])
  rownames(beta_sub) <- sample_ids
  # Transpose to probes x samples for EpiDISH
  beta_m <- t(beta_sub)
  
  # Check for NAs and impute with cohort median if any (<10% missing)
  na_per_probe <- rowSums(is.na(beta_m))
  if (any(na_per_probe > 0)) {
    cat(sprintf("Imputing NAs for %d probes with <10%% missingness...\n", sum(na_per_probe > 0)))
    for (i in which(na_per_probe > 0)) {
      med_val <- median(beta_m[i, ], na.rm = TRUE)
      beta_m[i, is.na(beta_m[i, ])] <- med_val
    }
  } else {
    cat("Zero missing values detected among reference probes.\n")
  }
  
  # Run Primary: EpiDISH RPC with centDHSbloodDMC.m
  cat("Running Primary Deconvolution: EpiDISH RPC (centDHSbloodDMC.m)...\n")
  t0 <- Sys.time()
  res_rpc <- epidish(beta.m = beta_m, ref.m = ref_epi, method = "RPC")
  t1 <- Sys.time()
  cat("RPC completed in", round(difftime(t1, t0, units = "secs"), 2), "seconds.\n")
  
  # Run Secondary: EpiDISH CP with centDHSbloodDMC.m (inequality constraint)
  cat("Running Secondary Deconvolution: EpiDISH CP (centDHSbloodDMC.m, inequality)...\n")
  t0 <- Sys.time()
  common_epi_probes <- intersect(rownames(ref_epi), rownames(beta_m))
  res_cp <- suppressMessages(epidish(beta.m = beta_m[common_epi_probes, , drop = FALSE],
                                     ref.m = ref_epi[common_epi_probes, , drop = FALSE],
                                     method = "CP", constraint = "inequality"))
  t1 <- Sys.time()
  cat("CP completed in", round(difftime(t1, t0, units = "secs"), 2), "seconds.\n")
  
  # Run Optional Third Check: Houseman CP with FlowSorted.Blood.450k.JaffeModelPars
  cat("Running Houseman Projection: FlowSorted.Blood.450k (JaffeModelPars, inequality)...\n")
  common_flow_probes <- intersect(rownames(ref_flw), rownames(beta_m))
  res_flow <- suppressMessages(epidish(beta.m = beta_m[common_flow_probes, , drop = FALSE],
                                       ref.m = ref_flw[common_flow_probes, , drop = FALSE],
                                       method = "CP", constraint = "inequality"))
  
  # Combine results into structured data frame
  rpc_df <- as.data.frame(res_rpc$estF)
  colnames(rpc_df) <- colnames(ref_epi)
  
  cp_df <- as.data.frame(res_cp$estF)
  colnames(cp_df) <- paste0(colnames(ref_epi), "_CP")
  
  flow_df <- as.data.frame(res_flow$estF)
  colnames(flow_df) <- paste0(colnames(ref_flw), "_Flow")
  
  meta_sub <- data_df[, meta_cols, drop = FALSE]
  
  out <- cbind(meta_sub, rpc_df, cp_df, flow_df)
  return(out)
}

# ------------------------------------------------------------------------------
# 3A. Process FHS (Pre-ComBat)
# ------------------------------------------------------------------------------
cat("\n=== Processing Cohort: FHS (Pre-ComBat) ===\n")
df_fhs_pre <- arrow::read_parquet(path_fhs_clean)
cat(sprintf("Loaded FHS pre-ComBat: %d rows x %d columns\n", nrow(df_fhs_pre), ncol(df_fhs_pre)))

fhs_pre_res <- run_deconv_pipeline(
  data_df = df_fhs_pre,
  id_col = "Share_ID",
  meta_cols = c("Share_ID", "Patient_ID", "batch", "age", "sex", "event", "time"),
  ref_epi = ref_epidish,
  ref_flw = ref_flow
)
fhs_pre_res$input_matrix <- "pre-ComBat"

# ------------------------------------------------------------------------------
# 3B. Process WHI (Pre-ComBat)
# ------------------------------------------------------------------------------
cat("\n=== Processing Cohort: WHI (Pre-ComBat) ===\n")
df_whi_pre <- arrow::read_parquet(path_whi_clean)
cat(sprintf("Loaded WHI pre-ComBat: %d rows x %d columns\n", nrow(df_whi_pre), ncol(df_whi_pre)))

# Clean up WHI column names (strip whitespace in 'sex ', 'event ')
colnames(df_whi_pre) <- trimws(colnames(df_whi_pre))
whi_pre_res <- run_deconv_pipeline(
  data_df = df_whi_pre,
  id_col = "Subject_ID",
  meta_cols = c("Subject_ID", "sample_ID", "age", "sex", "event", "time"),
  ref_epi = ref_epidish,
  ref_flw = ref_flow
)
whi_pre_res$input_matrix <- "pre-ComBat"
colnames(whi_pre_res)[colnames(whi_pre_res) == "Subject_ID"] <- "SUBJECT_ID"

# ------------------------------------------------------------------------------
# 3C. Process Post-ComBat Combined Matrix
# ------------------------------------------------------------------------------
cat("\n=== Processing Cohort: Combined (Post-ComBat) ===\n")
df_combat <- arrow::read_parquet(path_combat_clean)
cat(sprintf("Loaded Combined post-ComBat: %d rows x %d columns\n", nrow(df_combat), ncol(df_combat)))

# Separate FHS and WHI subsets from ComBat
df_combat_fhs <- df_combat[df_combat$batch %in% c("Gen 3", "JHU", "UMN"), ]
df_combat_whi <- df_combat[df_combat$batch == "WHI", ]

cat(sprintf("Running ComBat-FHS (n = %d)...\n", nrow(df_combat_fhs)))
fhs_post_res <- run_deconv_pipeline(
  data_df = df_combat_fhs,
  id_col = "Share_ID",
  meta_cols = c("Share_ID", "Patient_ID", "batch", "age", "sex", "event", "time"),
  ref_epi = ref_epidish,
  ref_flw = ref_flow
)
fhs_post_res$input_matrix <- "post-ComBat"

cat(sprintf("Running ComBat-WHI (n = %d)...\n", nrow(df_combat_whi)))
whi_post_res <- run_deconv_pipeline(
  data_df = df_combat_whi,
  id_col = "Patient_ID", # Patient_ID holds WHI Subject_ID in ComBat parquet
  meta_cols = c("Patient_ID", "Share_ID", "age", "sex", "event", "time"),
  ref_epi = ref_epidish,
  ref_flw = ref_flow
)
whi_post_res$input_matrix <- "post-ComBat"
colnames(whi_post_res)[colnames(whi_post_res) == "Patient_ID"] <- "SUBJECT_ID"
colnames(whi_post_res)[colnames(whi_post_res) == "Share_ID"]   <- "sample_ID"

# ------------------------------------------------------------------------------
# 4. Assemble Final Deliverable Proportion Tables (Deliverable 1)
# ------------------------------------------------------------------------------
cat("\n--- Assembling Deliverable 1: Cell Proportion Files ---\n")

cell_types_7 <- c("B", "NK", "CD4T", "CD8T", "Mono", "Neutro", "Eosino")
cell_types_cp <- paste0(cell_types_7, "_CP")
cell_types_flow <- c("CD8T_Flow", "CD4T_Flow", "NK_Flow", "Bcell_Flow", "Mono_Flow", "Gran_Flow")

fhs_cols_order <- c("Share_ID", "Patient_ID", "batch", "input_matrix",
                    cell_types_7, cell_types_cp, cell_types_flow,
                    "age", "sex", "event", "time")

whi_cols_order <- c("SUBJECT_ID", "sample_ID", "input_matrix",
                    cell_types_7, cell_types_cp, cell_types_flow,
                    "age", "sex", "event", "time")

fhs_combined_props <- rbind(fhs_pre_res[, fhs_cols_order], fhs_post_res[, fhs_cols_order])
whi_combined_props <- rbind(whi_pre_res[, whi_cols_order], whi_post_res[, whi_cols_order])

# Write Deliverable 1 files
file_fhs_out <- file.path(out_dir, "cell_proportions_FHS.csv")
file_whi_out <- file.path(out_dir, "cell_proportions_WHI.csv")

write.csv(fhs_combined_props, file_fhs_out, row.names = FALSE)
write.csv(whi_combined_props, file_whi_out, row.names = FALSE)
cat(sprintf("Saved %s (N = %d rows: %d pre-ComBat, %d post-ComBat)\n",
            file_fhs_out, nrow(fhs_combined_props), nrow(fhs_pre_res), nrow(fhs_post_res)))
cat(sprintf("Saved %s (N = %d rows: %d pre-ComBat, %d post-ComBat)\n\n",
            file_whi_out, nrow(whi_combined_props), nrow(whi_pre_res), nrow(whi_post_res)))

# ------------------------------------------------------------------------------
# 5. QC and Sanity Checks (Step 4 -> Deliverable 3)
# ------------------------------------------------------------------------------
cat("--- Step 4: Comprehensive QC and Sanity Checks ---\n")

qc_records <- list()

add_qc <- function(section, table_name, metric, cohort, batch, cell_type, n, value, details = "") {
  qc_records[[length(qc_records) + 1]] <<- data.frame(
    Section = section,
    Table = table_name,
    Metric = metric,
    Cohort = cohort,
    Batch = batch,
    Cell_Type = cell_type,
    N = n,
    Value = as.character(value),
    Details = details,
    stringsAsFactors = FALSE
  )
}

# 5.1 Distribution of fractions and sums
datasets_for_qc <- list(
  list(df = fhs_pre_res, name = "FHS", mat = "pre-ComBat"),
  list(df = fhs_post_res, name = "FHS", mat = "post-ComBat"),
  list(df = whi_pre_res, name = "WHI", mat = "pre-ComBat"),
  list(df = whi_post_res, name = "WHI", mat = "post-ComBat")
)

for (item in datasets_for_qc) {
  d <- item$df
  cname <- item$name
  imat <- item$mat
  
  batches <- if (cname == "FHS") c("ALL", unique(d$batch)) else "ALL"
  
  for (b in batches) {
    sub_d <- if (b == "ALL") d else d[d$batch == b, ]
    
    # RPC Cell Types
    for (ct in cell_types_7) {
      vals <- sub_d[[ct]]
      add_qc("Step4.1", "Fraction_Distributions_RPC", "Mean_SD", cname, paste0(b, "_", imat), ct,
             nrow(sub_d), sprintf("%.4f ± %.4f", mean(vals), sd(vals)))
      add_qc("Step4.1", "Fraction_Distributions_RPC", "Median", cname, paste0(b, "_", imat), ct,
             nrow(sub_d), sprintf("%.4f", median(vals)))
      add_qc("Step4.1", "Fraction_Distributions_RPC", "Min_Max", cname, paste0(b, "_", imat), ct,
             nrow(sub_d), sprintf("[%.4f, %.4f]", min(vals), max(vals)))
    }
    
    # RPC Sum
    rpc_sum <- rowSums(sub_d[, cell_types_7])
    add_qc("Step4.1", "Fraction_Distributions_RPC", "Sum_Mean_SD", cname, paste0(b, "_", imat), "SUM",
           nrow(sub_d), sprintf("%.4f ± %.4f", mean(rpc_sum), sd(rpc_sum)))
    add_qc("Step4.1", "Fraction_Distributions_RPC", "Sum_Min_Max", cname, paste0(b, "_", imat), "SUM",
           nrow(sub_d), sprintf("[%.4f, %.4f]", min(rpc_sum), max(rpc_sum)))
    
    # CP Cell Types
    for (ct in cell_types_7) {
      ct_cp <- paste0(ct, "_CP")
      vals_cp <- sub_d[[ct_cp]]
      add_qc("Step4.1", "Fraction_Distributions_CP", "Mean_SD", cname, paste0(b, "_", imat), ct,
             nrow(sub_d), sprintf("%.4f ± %.4f", mean(vals_cp), sd(vals_cp)))
      add_qc("Step4.1", "Fraction_Distributions_CP", "Median", cname, paste0(b, "_", imat), ct,
             nrow(sub_d), sprintf("%.4f", median(vals_cp)))
      add_qc("Step4.1", "Fraction_Distributions_CP", "Min_Max", cname, paste0(b, "_", imat), ct,
             nrow(sub_d), sprintf("[%.4f, %.4f]", min(vals_cp), max(vals_cp)))
    }
    
    # CP Sum
    cp_sum <- rowSums(sub_d[, cell_types_cp])
    add_qc("Step4.1", "Fraction_Distributions_CP", "Sum_Mean_SD", cname, paste0(b, "_", imat), "SUM",
           nrow(sub_d), sprintf("%.4f ± %.4f", mean(cp_sum), sd(cp_sum)))
    add_qc("Step4.1", "Fraction_Distributions_CP", "Sum_Min_Max", cname, paste0(b, "_", imat), "SUM",
           nrow(sub_d), sprintf("[%.4f, %.4f]", min(cp_sum), max(cp_sum)))
    
    # 5.2 Out-of-bounds checks
    # RPC
    n_neg_rpc <- sum(apply(sub_d[, cell_types_7], 1, function(r) any(r < 0)))
    n_gt1_rpc <- sum(apply(sub_d[, cell_types_7], 1, function(r) any(r > 1)))
    n_sum_bad_rpc <- sum(rpc_sum < 0.9 | rpc_sum > 1.1)
    
    add_qc("Step4.2", "Bound_Violations", "Any_Fraction_Negative_RPC", cname, paste0(b, "_", imat), "ALL_CELLS",
           nrow(sub_d), n_neg_rpc, sprintf("%.2f%%", n_neg_rpc/nrow(sub_d)*100))
    add_qc("Step4.2", "Bound_Violations", "Any_Fraction_Above_1_RPC", cname, paste0(b, "_", imat), "ALL_CELLS",
           nrow(sub_d), n_gt1_rpc, sprintf("%.2f%%", n_gt1_rpc/nrow(sub_d)*100))
    add_qc("Step4.2", "Bound_Violations", "Sum_Outside_0.9_1.1_RPC", cname, paste0(b, "_", imat), "SUM",
           nrow(sub_d), n_sum_bad_rpc, sprintf("%.2f%%", n_sum_bad_rpc/nrow(sub_d)*100))
    
    # CP
    n_neg_cp <- sum(apply(sub_d[, cell_types_cp], 1, function(r) any(r < 0)))
    n_gt1_cp <- sum(apply(sub_d[, cell_types_cp], 1, function(r) any(r > 1)))
    n_sum_bad_cp <- sum(cp_sum < 0.9 | cp_sum > 1.1)
    
    add_qc("Step4.2", "Bound_Violations", "Any_Fraction_Negative_CP", cname, paste0(b, "_", imat), "ALL_CELLS",
           nrow(sub_d), n_neg_cp, sprintf("%.2f%%", n_neg_cp/nrow(sub_d)*100))
    add_qc("Step4.2", "Bound_Violations", "Any_Fraction_Above_1_CP", cname, paste0(b, "_", imat), "ALL_CELLS",
           nrow(sub_d), n_gt1_cp, sprintf("%.2f%%", n_gt1_cp/nrow(sub_d)*100))
    add_qc("Step4.2", "Bound_Violations", "Sum_Outside_0.9_1.1_CP", cname, paste0(b, "_", imat), "SUM",
           nrow(sub_d), n_sum_bad_cp, sprintf("%.2f%%", n_sum_bad_cp/nrow(sub_d)*100))
    
    # 5.3 Correlation with Age and Sex Difference
    age_vals <- as.numeric(sub_d$age)
    sex_vals <- as.numeric(sub_d$sex) # 1 = male, 2 = female
    
    for (ct in cell_types_7) {
      vals <- sub_d[[ct]]
      # Age correlation
      valid_age <- which(!is.na(age_vals) & !is.na(vals))
      if (length(valid_age) > 10) {
        ctest <- cor.test(age_vals[valid_age], vals[valid_age])
        add_qc("Step4.3", "Age_Correlation_RPC", "Pearson_r", cname, paste0(b, "_", imat), ct,
               length(valid_age), sprintf("%.4f", ctest$estimate), sprintf("p = %.2e", ctest$p.value))
      }
      
      # Sex difference (FHS only, where both males and females exist)
      if (cname == "FHS") {
        m_vals <- vals[sex_vals == 1]
        f_vals <- vals[sex_vals == 2]
        if (length(m_vals) > 5 && length(f_vals) > 5) {
          diff_mean <- mean(m_vals) - mean(f_vals)
          tt <- t.test(m_vals, f_vals)
          add_qc("Step4.3", "Sex_Difference_RPC", "Mean_Diff_Male_Minus_Female", cname, paste0(b, "_", imat), ct,
                 nrow(sub_d), sprintf("%.4f", diff_mean), sprintf("p = %.2e (M: %.4f, F: %.4f)", tt$p.value, mean(m_vals), mean(f_vals)))
        }
      }
    }
  }
}

# 5.4 Agreement between RPC and CP (Pearson r and mean difference)
cat("Assessing RPC vs CP Agreement...\n")
for (item in datasets_for_qc) {
  d <- item$df
  cname <- item$name
  imat <- item$mat
  
  for (ct in cell_types_7) {
    v_rpc <- d[[ct]]
    v_cp  <- d[[paste0(ct, "_CP")]]
    
    ctest <- cor.test(v_rpc, v_cp)
    mean_diff <- mean(v_rpc - v_cp)
    mae <- mean(abs(v_rpc - v_cp))
    
    add_qc("Step4.4", "RPC_vs_CP_Agreement", "Pearson_r", cname, imat, ct,
           nrow(d), sprintf("%.4f", ctest$estimate), sprintf("p = %.2e", ctest$p.value))
    add_qc("Step4.4", "RPC_vs_CP_Agreement", "Mean_Difference_RPC_minus_CP", cname, imat, ct,
           nrow(d), sprintf("%.4f", mean_diff), sprintf("MAE = %.4f", mae))
  }
}

# 5.5 Agreement between Pre- and Post-ComBat Matrices
cat("Assessing Pre- vs Post-ComBat Agreement...\n")
# Match FHS on Share_ID
fhs_merged_combat_comp <- merge(fhs_pre_res, fhs_post_res, by = "Share_ID", suffixes = c("_pre", "_post"))
for (ct in cell_types_7) {
  v_pre <- fhs_merged_combat_comp[[paste0(ct, "_pre")]]
  v_post <- fhs_merged_combat_comp[[paste0(ct, "_post")]]
  
  ctest <- cor.test(v_pre, v_post)
  mean_diff <- mean(v_pre - v_post)
  mae <- mean(abs(v_pre - v_post))
  
  add_qc("Step4.5", "Pre_vs_Post_ComBat_Agreement", "Pearson_r", "FHS", "ALL", ct,
         nrow(fhs_merged_combat_comp), sprintf("%.4f", ctest$estimate), sprintf("p = %.2e", ctest$p.value))
  add_qc("Step4.5", "Pre_vs_Post_ComBat_Agreement", "Mean_Diff_Pre_minus_Post", "FHS", "ALL", ct,
         nrow(fhs_merged_combat_comp), sprintf("%.4f", mean_diff), sprintf("MAE = %.4f", mae))
}

# Match WHI on SUBJECT_ID
whi_merged_combat_comp <- merge(whi_pre_res, whi_post_res, by = "SUBJECT_ID", suffixes = c("_pre", "_post"))
for (ct in cell_types_7) {
  v_pre <- whi_merged_combat_comp[[paste0(ct, "_pre")]]
  v_post <- whi_merged_combat_comp[[paste0(ct, "_post")]]
  
  ctest <- cor.test(v_pre, v_post)
  mean_diff <- mean(v_pre - v_post)
  mae <- mean(abs(v_pre - v_post))
  
  add_qc("Step4.5", "Pre_vs_Post_ComBat_Agreement", "Pearson_r", "WHI", "ALL", ct,
         nrow(whi_merged_combat_comp), sprintf("%.4f", ctest$estimate), sprintf("p = %.2e", ctest$p.value))
  add_qc("Step4.5", "Pre_vs_Post_ComBat_Agreement", "Mean_Diff_Pre_minus_Post", "WHI", "ALL", ct,
         nrow(whi_merged_combat_comp), sprintf("%.4f", mean_diff), sprintf("MAE = %.4f", mae))
}

# 5.6 Correlation with Measured Complete Blood Count (CBC) in FHS Gen 3 Exam 2
cat("Correlating with Measured CBC (FHS Gen 3 Exam 2, table pht002889)...\n")
cbc_raw <- read.table(path_cbc, sep = "\t", header = TRUE, comment.char = "#", quote = "", stringsAsFactors = FALSE)
cat(sprintf("Loaded CBC table: %d rows x %d columns\n", nrow(cbc_raw), ncol(cbc_raw)))

cbc_df <- data.frame(
  Share_ID = as.character(cbc_raw$shareid),
  WBC = suppressWarnings(as.numeric(cbc_raw$WBC)),
  NE_PER = suppressWarnings(as.numeric(cbc_raw$NE_PER)),
  LY_PER = suppressWarnings(as.numeric(cbc_raw$LY_PER)),
  MO_PER = suppressWarnings(as.numeric(cbc_raw$MO_PER)),
  EO_PER = suppressWarnings(as.numeric(cbc_raw$EO_PER)),
  exdate = cbc_raw$exdate
)

# Merge with Gen 3 estimates (Pre-ComBat)
g3_pre <- fhs_pre_res[fhs_pre_res$batch == "Gen 3", ]
g3_pre$Share_ID <- as.character(g3_pre$Share_ID)
g3_cbc <- merge(g3_pre, cbc_df, by = "Share_ID")

# Estimated cell percentages (fraction * 100)
g3_cbc$Neutro_pct <- g3_cbc$Neutro * 100
g3_cbc$Lymph_pct  <- (g3_cbc$CD4T + g3_cbc$CD8T + g3_cbc$B + g3_cbc$NK) * 100
g3_cbc$Mono_pct   <- g3_cbc$Mono * 100
g3_cbc$Eosino_pct <- g3_cbc$Eosino * 100

cbc_pairs <- list(
  list(est = "Neutro_pct", cbc = "NE_PER", label = "Neutrophils (Neutro vs NE_PER)"),
  list(est = "Lymph_pct",  cbc = "LY_PER", label = "Lymphocytes (CD4T+CD8T+B+NK vs LY_PER)"),
  list(est = "Mono_pct",   cbc = "MO_PER", label = "Monocytes (Mono vs MO_PER)"),
  list(est = "Eosino_pct", cbc = "EO_PER", label = "Eosinophils (Eosino vs EO_PER)")
)

cat("--- Clinical CBC Validation Correlations (FHS Gen 3 Exam 2) ---\n")
for (cp in cbc_pairs) {
  val_est <- g3_cbc[[cp$est]]
  val_cbc <- g3_cbc[[cp$cbc]]
  valid_idx <- which(!is.na(val_est) & !is.na(val_cbc))
  
  n_valid <- length(valid_idx)
  ctest <- cor.test(val_est[valid_idx], val_cbc[valid_idx])
  sctest <- cor.test(val_est[valid_idx], val_cbc[valid_idx], method = "spearman")
  
  cat(sprintf("  %s: N = %d, Pearson r = %.4f (p = %.2e), Spearman rho = %.4f\n",
              cp$label, n_valid, ctest$estimate, ctest$p.value, sctest$estimate))
  
  add_qc("Step4.6", "Clinical_CBC_Validation", "Pearson_r", "FHS_Gen3", "Exam_2", cp$label,
         n_valid, sprintf("%.4f", ctest$estimate),
         sprintf("p = %.2e, Spearman rho = %.4f (Mean CBC: %.2f%%, Mean Est: %.2f%%)",
                 ctest$p.value, sctest$estimate, mean(val_cbc[valid_idx]), mean(val_est[valid_idx])))
}

# Also do CBC validation on Post-ComBat for completeness
g3_post <- fhs_post_res[fhs_post_res$batch == "Gen 3", ]
g3_post$Share_ID <- as.character(g3_post$Share_ID)
g3_post_cbc <- merge(g3_post, cbc_df, by = "Share_ID")

g3_post_cbc$Neutro_pct <- g3_post_cbc$Neutro * 100
g3_post_cbc$Lymph_pct  <- (g3_post_cbc$CD4T + g3_post_cbc$CD8T + g3_post_cbc$B + g3_post_cbc$NK) * 100
g3_post_cbc$Mono_pct   <- g3_post_cbc$Mono * 100
g3_post_cbc$Eosino_pct <- g3_post_cbc$Eosino * 100

for (cp in cbc_pairs) {
  val_est <- g3_post_cbc[[cp$est]]
  val_cbc <- g3_post_cbc[[cp$cbc]]
  valid_idx <- which(!is.na(val_est) & !is.na(val_cbc))
  
  n_valid <- length(valid_idx)
  ctest <- cor.test(val_est[valid_idx], val_cbc[valid_idx])
  sctest <- cor.test(val_est[valid_idx], val_cbc[valid_idx], method = "spearman")
  
  add_qc("Step4.6", "Clinical_CBC_Validation_PostComBat", "Pearson_r", "FHS_Gen3", "Exam_2_postComBat", cp$label,
         n_valid, sprintf("%.4f", ctest$estimate),
         sprintf("p = %.2e, Spearman rho = %.4f", ctest$p.value, sctest$estimate))
}

# Save qc_tables.csv (Deliverable 3)
qc_df <- do.call(rbind, qc_records)
file_qc_out <- file.path(out_dir, "qc_tables.csv")
write.csv(qc_df, file_qc_out, row.names = FALSE)
cat(sprintf("\nSaved Deliverable 3: %s (%d records)\n", file_qc_out, nrow(qc_df)))

# ------------------------------------------------------------------------------
# 6. Step 5: Comparison with Earlier Outputs
# ------------------------------------------------------------------------------
cat("\n--- Step 5: Comparison with Earlier Outputs ---\n")
cat("Search findings:\n")
cat("  1) Earlier script Cell_compostion_processing.ipynb referenced FHS_epidish_cellcounts_blood.csv,\n")
cat("     but that file was not saved/present on disk.\n")
cat("  2) Earlier batch-correction script Limma.R used limma::removeBatchEffect (~ age + sex).\n")
cat("  3) Earlier ComBat notebook Batch_correction.ipynb used neuroCombat (~ age + sex).\n")
cat("  -> No prior deconvolution output CSV files exist to compare against.\n\n")

# ------------------------------------------------------------------------------
# 7. Session Information (Deliverable 2)
# ------------------------------------------------------------------------------
cat("--- Saving sessionInfo.txt ---\n")
s_info_file <- file.path(out_dir, "sessionInfo.txt")
writeLines(capture.output(sessionInfo()), s_info_file)
cat(sprintf("Saved %s\n", s_info_file))

cat("\n==============================================================================\n")
cat("Deconvolution Pipeline Successfully Finished!\n")
cat("==============================================================================\n")
