# Input data

No individual-level data are included in this repository. The Framingham Heart Study (FHS, dbGaP accession [phs000007](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000007)) and the Women's Health Initiative (WHI, dbGaP accession [phs000200](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000200)) datasets are controlled-access. Access must be requested directly through the NCBI dbGaP portal. Both studies utilized the Illumina HumanMethylation450 BeadChip.

## dbGaP Data Access and Compliance

### Data Use Certification (DUC) & Sharing Restrictions
Data from dbGaP are governed by NIH Genomic Data Sharing (GDS) policies and study-specific Data Use Certifications (DUCs).
- **No Individual-Level Data:** Neither raw dbGaP files nor derived individual-level datasets (including processed CSV, Parquet, or feature matrices) can be distributed in this repository.
- **Controlled Sharing:** Individual-level data, including intermediate predictions and biological clock scores, may only be analyzed within approved computing environments authorized by the investigator's Data Access Committee (DAC) and Institutional Review Board (IRB).
- **Smoke Testing:** To test scripts and pipelines without controlled-access data, use the synthetic data generator (`python tools/make_synthetic_data.py`), which outputs mock data files matching the required schemas.

## How to Obtain the Data

To request access to the cohorts:
1. Log in to the [dbGaP Authorized Access portal](https://dbgap.ncbi.nlm.nih.gov/).
2. Submit a Data Access Request (DAR) for:
   - **Framingham Heart Study (FHS):** dbGaP accession `phs000007`
   - **Women's Health Initiative (WHI):** dbGaP accession `phs000200`
3. Await approval from the respective NIH Data Access Committees.

### Dataset Identifiers and Raw File Mapping

> **TODO [AUTHOR] (dbGaP Accession Specifics):**
> Please verify and fill in the exact version numbers (`.vN.pN`), dataset accessions (`pht`), and consent groups for all files required to run `01_preprocessing/` from scratch:
>
> | Dataset | Accession / File Pattern | Consent Group | Script | Purpose |
> |---|---|---|---|---|
> | FHS DNA Methylation | `phs000007.vN.pN...` *(TODO: specify exact accession/tar)* | e.g., `c1` (HMB-IRB-MDS) | `01_fhs_raw_processing.py` | Raw 450K beta matrices for Gen 3, JHU, UMN batches |
> | WHI DNA Methylation | `phs000200.vN.pN...` *(TODO: specify exact accession/tar)* | *(TODO: specify consent group)* | `02_whi_raw_processing_knn_imputation.py` | WHI BA23 450K methylation matrix and sample manifest |
> | FHS Demographics & Labs | `phs000007.vN.phtXXXXXX.vN.pN...` *(TODO)* | e.g., `c1` | `04_build_model_inputs.py` | Age, sex, lipid panel (Total Cholesterol, Triglycerides, HDL), fasting glucose |
> | FHS Mortality Follow-up | `phs000007.vN.phtXXXXXX.vN.pN...` *(TODO)* | e.g., `c1` | `04_build_model_inputs.py` | Survival time (years) and all-cause mortality event indicator |
> | FHS CVD Adjudicated Events | `phs000007.vN.phtXXXXXX.vN.pN...` *(TODO)* | e.g., `c1` | `04_build_model_inputs.py` | Events and censoring times for MI, AP, CVA, IC, CHF |
> | FHS Pedigree | `phs000007.v32.pht001415.v4.p12.c1...Framingham_Pedigree.MULTI.txt` | `c1` (HMB-IRB-MDS) | `06_revision_analyses/...` | Pedigree structure for family-leakage audit |
> | FHS Complete Blood Count | `phs000007.v33.pht002889.v5.p14.c1...l_fhslab_2011_m_0656s...` | `c1` (HMB-IRB-MDS) | `06_revision_analyses/.../deconvolution.R` | Clinical CBC differential counts for deconvolution validation |
> | WHI Clinical & Outcomes | `phs000200.vN.phtXXXXXX.vN.pN...` *(TODO)* | *(TODO)* | `04_build_model_inputs.py` | Baseline age, lipid panel, survival time, all-cause mortality |

## Expected Modeling Data Files

The downstream modeling code (`02_model_training/` onward) expects these files in `data/` (or the directory specified by `EPICLOCK_DATA_DIR`):

| File | One row per | Columns |
|---|---|---|
| `FHS_non_normalized.csv` | FHS participant (n = 3,493) | `Share_ID`, `dbGaP_Subject_ID`, `sex`, `age`, `event`, `batch`, `time`, `Total_Cholesterol`, `Triglycerides`, `Glucose`, `HDL`, then the 10 CpGs |
| `WHI_non_normalized.csv` | WHI participant (n = 196) | `SUBJECT_ID`, `ID`, `age`, `sex`, `event`, `time`, `Total_Cholesterol`, `Triglycerides`, `Glucose`, `HDL`, then the 10 CpGs |
| `Disease_data_with_censoring.csv` | FHS participant | `dbGaP_Subject_ID`, `sex`, `age`, `batch`, then event flag and time for `MI`, `AP`, `CVA`, `IC`, `CHF` |
| `phs000007...Framingham_Pedigree.MULTI.txt` | FHS participant | dbGaP pedigree file (only for the family-leakage analysis) |

The 10 CpGs: `cg04604946`, `cg06639320`, `cg07955995`, `cg10246448`, `cg10501210`, `cg11617964`, `cg16867657`, `cg20591472`, `cg24724428`, `cg25590826` (beta values).

Coding: `event` = 1 for death, 0 if censored; `time` = years from the baseline exam to death or last contact; `batch` = FHS processing batch (Gen 3, JHU, UMN).

The CpG values in these files are not ComBat-corrected. `06_revision_analyses/r1_c4_combat_sensitivity/` compares results with and without correction.

The clock estimation (`03_epigenetic_clocks/`) and some revision analyses also need the full 450K matrices (`FHS_methylation_merged.parquet`, `WHI_full_imputed.parquet`) produced by `01_preprocessing/`.
