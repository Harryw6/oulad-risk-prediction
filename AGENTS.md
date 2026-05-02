# AGENTS.md

You are working on a reproducible academic machine learning project for an education analytics paper.

Project topic:
Early prediction of at-risk students in online learning using the Open University Learning Analytics Dataset (OULAD).

Core requirement:
Build a complete, reproducible experimental pipeline. Do not fabricate results. All reported results must come from executed code.

Dataset:
The raw OULAD CSV files should be placed in:
data/oulad/raw/

Expected files:
- studentInfo.csv
- studentRegistration.csv
- studentAssessment.csv
- assessments.csv
- studentVle.csv
- vle.csv
- courses.csv

Research task:
Predict whether a student is academically at risk.

Binary label:
- at_risk = 1 if final_result is Fail or Withdrawn
- at_risk = 0 if final_result is Pass or Distinction

Critical academic rule:
Avoid data leakage. For each prediction window, only use data that occurred on or before that prediction day.

Prediction windows:
- 7 days
- 14 days
- 28 days
- 56 days
- full course period

Do not use:
- final_result as a feature
- id_student as a predictive feature
- future VLE interactions after the prediction window
- future assessment scores after the prediction window
- post-withdrawal behavior if it would leak the outcome

Preferred Python stack:
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost
- lightgbm
- catboost
- shap
- optuna if useful

Code requirements:
- Use modular Python scripts under src/
- Use fixed random seeds
- Make all outputs reproducible
- Save tables to outputs/tables/
- Save figures to outputs/figures/
- Save written reports to outputs/reports/
- Add clear README instructions

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Balanced Accuracy
- Confusion matrix

Validation protocols:
1. Stratified train-test split
2. GroupKFold by code_module + code_presentation
3. Leave-one-course-presentation-out evaluation if feasible

Required outputs:
- outputs/tables/dataset_summary.csv
- outputs/tables/model_comparison.csv
- outputs/tables/time_window_results.csv
- outputs/tables/feature_ablation_results.csv
- outputs/tables/shap_top_features.csv
- outputs/figures/label_distribution.png
- outputs/figures/time_window_performance.png
- outputs/figures/roc_curves.png
- outputs/figures/pr_curves.png
- outputs/figures/confusion_matrix_best_model.png
- outputs/figures/shap_summary.png
- outputs/reports/leakage_audit.md
- outputs/reports/experiment_summary.md
- README.md with reproduction instructions

When making design choices:
Make reasonable decisions independently, document the decision, and continue. Do not stop unless the dataset is missing or execution is impossible.

Final response expected from Codex:
Summarize:
1. What code was created
2. What experiments were run
3. What results were obtained
4. What files were generated
5. Any limitations or failed steps
6. Exact commands to reproduce everything