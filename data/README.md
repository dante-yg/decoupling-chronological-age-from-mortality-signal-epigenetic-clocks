# Input data

No data are included. FHS (dbGaP phs000007) and WHI (dbGaP phs000200) are controlled-access; request them through dbGaP. Both use the Illumina HumanMethylation450 BeadChip.

The modelling code (`02_model_training/` onward) expects these files in `data/` under the project root:

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
