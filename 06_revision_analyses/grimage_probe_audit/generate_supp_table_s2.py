import os
import pandas as pd

# Define rows based on our computed ground truth
# FHS N = 3,493; WHI N = 196
data = [
    {
        "Clock": "Horvath (2013)",
        "Cohort": "FHS",
        "n_required": 353,
        "n_present_complete": 330,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 23,
        "n_values_imputed": 23 * 3493, # 80,339
        "pct_availability": 93.48,
        "non_cpg_inputs": "None"
    },
    {
        "Clock": "Horvath (2013)",
        "Cohort": "WHI",
        "n_required": 353,
        "n_present_complete": 353,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 0,
        "n_values_imputed": 0,
        "pct_availability": 100.00,
        "non_cpg_inputs": "None"
    },
    {
        "Clock": "Hannum (2013)",
        "Cohort": "FHS",
        "n_required": 69,
        "n_present_complete": 61,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 8,
        "n_values_imputed": 8 * 3493, # 27,944
        "pct_availability": 88.41,
        "non_cpg_inputs": "2 non-CpG probes (ch.2.30415474F, ch.13.39564907R; present in 450K, 0 in FHS)"
    },
    {
        "Clock": "Hannum (2013)",
        "Cohort": "WHI",
        "n_required": 69,
        "n_present_complete": 69,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 0,
        "n_values_imputed": 0,
        "pct_availability": 100.00,
        "non_cpg_inputs": "2 non-CpG probes (both present in WHI)"
    },
    {
        "Clock": "DNAmPhenoAge (2018)",
        "Cohort": "FHS",
        "n_required": 513,
        "n_present_complete": 509,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 4,
        "n_values_imputed": 4 * 3493, # 13,972
        "pct_availability": 99.22,
        "non_cpg_inputs": "None"
    },
    {
        "Clock": "DNAmPhenoAge (2018)",
        "Cohort": "WHI",
        "n_required": 513,
        "n_present_complete": 513,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 0,
        "n_values_imputed": 0,
        "pct_availability": 100.00,
        "non_cpg_inputs": "None"
    },
    {
        "Clock": "DNAmGrimAge (2019)",
        "Cohort": "FHS",
        "n_required": 1030,
        "n_present_complete": 829,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 201,
        "n_values_imputed": 201 * 3493, # 702,093
        "pct_availability": 80.49,
        "non_cpg_inputs": "Chronological age (used); Female sex indicator (absent in vars, imputed as 1.0)"
    },
    {
        "Clock": "DNAmGrimAge (2019)",
        "Cohort": "WHI",
        "n_required": 1030,
        "n_present_complete": 1027,
        "n_partially_missing": 0,
        "max_pct_missing": 0.0,
        "median_pct_missing": 0.0,
        "n_absent": 3,
        "n_values_imputed": 3 * 196, # 588
        "pct_availability": 99.71,
        "non_cpg_inputs": "Chronological age (used); Female sex indicator (all female cohort; imputed as 1.0)"
    }
]

df = pd.DataFrame(data)
out_path = r"C:\Yugan_Gogul\datasets\combined\data\grimage_audit\Supplementary_Table_S2.csv"
df.to_csv(out_path, index=False)
print(f"Saved Supplementary_Table_S2.csv to {out_path}")
print(df.to_string(index=False))
