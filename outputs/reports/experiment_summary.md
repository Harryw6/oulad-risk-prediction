# Experiment Summary

## Objective
Predict academically at-risk students in OULAD using multi-window demographic, VLE, and assessment features.

## Label
`at_risk = 1` for Fail or Withdrawn; `at_risk = 0` for Pass or Distinction.

## Implementation Decisions

- The main model comparison uses the 56-day early-warning window with all feature groups combined.
- The selected tree-based model is reused for time-window and feature-ablation experiments; when tree models are statistically tied within 0.001 PR-AUC, XGBoost/LightGBM are preferred for stable behavior on sparse early-window ablations.
- Registration timing is treated as a window-aware administrative feature; registration dates after the prediction day are encoded as missing.
- The full-course window is reported as an upper-bound comparison rather than an early-warning result.
- Optional libraries are included when importable; this run included the models present in `model_comparison.csv`.

## Best Results

- Best model comparison row: CatBoost with PR-AUC=0.9186, ROC-AUC=0.8920, F1=0.8090.
- Best tree-based model selected for window/ablation experiments: XGBoost.
- Best time-window result: full with PR-AUC=0.9895, ROC-AUC=0.9858, F1=0.9474.
- Best ablation result: full / demographic_vle_assessment with PR-AUC=0.9895.

## Validation Protocols

- Stratified train-test split is grouped by `id_student` to avoid repeated-student overlap.
- GroupKFold groups by `code_module + code_presentation` and is summarized in `model_comparison.csv`.
- Leave-one-course-presentation-out is summarized in `model_comparison.csv`.

## Runtime

Completed in 3.80 minutes.

## Known Limitations

- Students who withdraw before a prediction day remain in the dataset because the target definition includes Withdrawn; the withdrawal date itself is never used.
- No hyperparameter optimization was run; model settings are conservative defaults for reproducibility and runtime.
- SHAP is computed on a held-out sample of at most 1,200 rows for runtime.
