# About this Project

## Scientific Motivation
Epigenetic clocks developed from DNA methylation (DNAm) data are widely used as biomarkers of biological aging and mortality risk. However, first-generation clocks (e.g., Horvath 2013, Hannum 2013) were trained solely to predict chronological age, whereas second-generation clocks (e.g., DNAm PhenoAge, DNAm GrimAge) incorporated clinical biomarkers and mortality-related surrogates. 

This study addresses a fundamental question in biogerontology:
> **Do CpG sites originally selected strictly for chronological age prediction retain independent mortality risk signals once chronological age is rigorously adjusted for?**

## Study Design & Cohorts
- **Development Cohort:** Framingham Heart Study (FHS, *N* = 3,493; 634 deaths; Illumina HumanMethylation450 array)
  - Evaluated under 60% training (*N* = 2,095), 20% validation (*N* = 699), and 20% test splits (*N* = 699).
- **External Validation Cohort:** Women's Health Initiative (WHI, *N* = 196; 56 deaths; all-female independent cohort).

## Analytical Approach
1. **Feature Set (EpiClock):** Ten target CpGs (`cg04604946`, `cg06639320`, `cg07955995`, `cg10246448`, `cg10501210`, `cg11617964`, `cg16867657`, `cg20591472`, `cg24724428`, `cg25590826`) with and without four standard clinical biomarkers (Total Cholesterol, Triglycerides, HDL, Glucose).
2. **Survival Learners:** CoxNet (elastic net), Weibull Accelerated Failure Time (AFT), Survival Support Vector Machine (SSVM), Multi-Task Logistic Regression (MTLR), Random Survival Forests (RSF), and a Stacking Neural Network Ensemble (CoxMLP).
3. **Decoupling Analysis:** Multivariable Cox proportional hazards models evaluating hazard ratios (HR) and discrimination (Harrell's C-index) with and without chronological age and sex adjustment.
4. **Epigenetic Clock Benchmarking:** Standardized head-to-head comparison against four established clocks (Horvath, Hannum, PhenoAge, GrimAge) computed via `pyaging`.

## Research Team & Status
- **Affiliation:** Department of Biological Science, Chosun University, Gwangju, Republic of Korea
- **Status:** Code accompanying a manuscript under revision at *Archives of Gerontology and Geriatrics*.
