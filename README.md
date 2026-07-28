This repository contains the analysis codebase for the paper studying the decoupling of chronological-age signal from mortality signal across first and second-generation epigenetic clocks. The code implements the preprocessing, model training, stacking, pyaging comparison, and nested Cox analysis to assess whether the age-associated CpGs carry prognostic information for survival beyond chronological age and routine clinical biomarkers.

System requirements:
- Python 3.9.15
- R 4.3.2

Installation:
Create the Conda environment using the provided environment description file:
conda env create -f environment.yml
conda activate decoupling-chronological-age-from-mortality-signal-epigenetic-clocks

Alternatively, packages can be installed using pip:
pip install -r requirements.txt

Data availability:
The datasets used in this study (Framingham Heart Study and Women's Health Initiative) are access-restricted and must be obtained directly from dbGaP under accessions phs000007 and phs000200.

Run order:
1. Data preprocessing: python src/stage1_preprocessing/preprocess.py
2. Feature-set construction: python src/stage2_feature_construction/construct_features.py
3. Survival modeling:
   - train base models: python src/stage3_survival_modeling/train_base_models_cpg.py and python src/stage3_survival_modeling/train_base_models_combined.py
   - train stacking: python src/stage3_survival_modeling/train_stacking_cpg.py and python src/stage3_survival_modeling/train_stacking_combined.py
4. Clock comparison:
   - estimate clocks: python src/stage4_clock_comparison/estimate_biological_clocks.py
   - run comparison: python src/stage4_clock_comparison/evaluate_clock_comparison.py
5. Statistics and visualization:
   - cohort demographics: python src/stage5_statistics_and_visualization/cohort_demographics.py
   - nested Cox analysis: python src/stage5_statistics_and_visualization/nested_cox_analysis.py
   - generate figures: python src/stage5_statistics_and_visualization/generate_figures.py
