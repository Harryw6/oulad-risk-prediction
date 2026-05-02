# Leakage Audit

Reviewer stance: strict. PASS means the current code satisfies the check under the implemented experimental design; WARN means a defensible limitation remains; FAIL means results should not be used until fixed.

## Risk Register

| risk | status | evidence | changes_made | remaining_limitations |
| --- | --- | --- | --- | --- |
| 1. General data leakage | PASS | src/build_features.py constructs window-specific feature frames; src/leakage_audit.py checks forbidden fields and event dates for each window. | Added stricter feature-frame checks and made registration features window-aware. | Observed inactivity can still reflect students who already withdrew; date_unregistration is excluded, but cohort censoring is a separate sensitivity analysis. |
| 2. Incorrect train-test splitting | PASS | src/run_experiments.py::_stratified_indices uses StratifiedGroupKFold with groups=id_student; split audit checks zero student overlap. | Replaced instance-level StratifiedShuffleSplit with student-grouped StratifiedGroupKFold. | Main split is student-disjoint but not course-disjoint; course-disjoint performance is reported separately with GroupKFold and leave-one-course-presentation-out. |
| 3. Future VLE data | PASS | src/build_features.py::build_vle_features filters studentVle with date <= cutoff_day, or date <= module_presentation_length for the full upper-bound window. | Excluded internal vle_prediction_day from model feature groups to avoid unnecessary course-window encoding. | The full window is not an early-warning setting and should be interpreted only as an upper bound. |
| 4. Future assessment scores | PASS | src/build_features.py::build_assessment_features requires assessment_date <= cutoff_day and date_submitted <= cutoff_day before using score features. | Audit now explicitly checks assessment_last_submitted_day <= prediction_day for every window. | Assessment due schedules are course design metadata and remain available to the model through due-count/weight features. |
| 5. final_result as a feature | PASS | src/load_data.py creates at_risk from final_result; src/build_features.py::get_feature_group_columns never includes final_result. | Audit forbids final_result in model feature columns. | None identified. |
| 6. id_student as a feature | PASS | id_student is retained for joins, split grouping, and SHAP identifiers; src/build_features.py::get_feature_group_columns excludes it. | Audit forbids id_student in model feature columns and uses it only for train/test grouping. | Repeated students are now separated across train/test, but repeated-course histories are not explicitly modeled. |
| 7. Preprocessing fitted on full dataset | PASS | src/train_models.py::build_pipeline wraps ColumnTransformer and estimator in one sklearn Pipeline; src/train_models.py::fit_evaluate calls model.fit only on x_train/y_train. | No code change required; report now documents this explicitly. | Feature engineering itself is deterministic aggregation and not learned preprocessing. |
| 8. Class imbalance handling | PASS | src/train_models.py uses class_weight='balanced' or balanced_subsample where supported; XGBoost scale_pos_weight is computed from y_train only. | No SMOTE or full-data resampling is used. | Threshold is the default classifier threshold; operating-point optimization is future work. |
| 9. Misleading metrics for imbalance | PASS | src/evaluate.py computes PR-AUC, ROC-AUC, F1, recall, precision, balanced accuracy, and confusion matrix; model selection sorts by PR-AUC, ROC-AUC, then F1. | No code change required; accuracy is reported but not used alone. | Calibration and decision-curve analysis are not yet included. |
| 10. Non-reproducible figures or tables | PASS | src/run_experiments.py generates every required output from raw CSV files using RANDOM_STATE=42; outputs are saved under outputs/. | Audit now checks required generated artifacts exist. | All required tables, figures, and reports exist under outputs/. |
| 11. Withdrawal date leakage | PASS | src/load_data.py reads date_unregistration, but create_modeling_base does not merge it; audit forbids date_unregistration in feature columns. | No predictive use of withdrawal date is allowed. | Withdrawal labels remain part of the target by study definition. |
| 12. Future registration date leakage | PASS | src/build_features.py::build_registration_features sets days_before_start_registered to missing unless date_registration <= prediction_day. | Moved registration feature creation out of static base construction and made it prediction-window aware. | Rows for students not yet registered by a prediction day remain in the analytical cohort with registration timing missing. |

## Window-Level Checks

| window | check | status | detail |
| --- | --- | --- | --- |
| day_7 | forbidden model fields absent | PASS | No target, student id, course id, or withdrawal fields in model feature set. |
| day_7 | VLE activity occurs on/before prediction day | PASS | violations=0 |
| day_7 | assessment submissions occur on/before prediction day | PASS | violations=0 |
| day_7 | registration date feature unavailable before known registration | PASS | future-or-missing registration rows=138; violations=0 |
| day_14 | forbidden model fields absent | PASS | No target, student id, course id, or withdrawal fields in model feature set. |
| day_14 | VLE activity occurs on/before prediction day | PASS | violations=0 |
| day_14 | assessment submissions occur on/before prediction day | PASS | violations=0 |
| day_14 | registration date feature unavailable before known registration | PASS | future-or-missing registration rows=103; violations=0 |
| day_28 | forbidden model fields absent | PASS | No target, student id, course id, or withdrawal fields in model feature set. |
| day_28 | VLE activity occurs on/before prediction day | PASS | violations=0 |
| day_28 | assessment submissions occur on/before prediction day | PASS | violations=0 |
| day_28 | registration date feature unavailable before known registration | PASS | future-or-missing registration rows=61; violations=0 |
| day_56 | forbidden model fields absent | PASS | No target, student id, course id, or withdrawal fields in model feature set. |
| day_56 | VLE activity occurs on/before prediction day | PASS | violations=0 |
| day_56 | assessment submissions occur on/before prediction day | PASS | violations=0 |
| day_56 | registration date feature unavailable before known registration | PASS | future-or-missing registration rows=54; violations=0 |
| full | forbidden model fields absent | PASS | No target, student id, course id, or withdrawal fields in model feature set. |
| full | VLE activity occurs on/before prediction day | PASS | violations=0 |
| full | assessment submissions occur on/before prediction day | PASS | violations=0 |
| full | registration date feature unavailable before known registration | PASS | future-or-missing registration rows=45; violations=0 |

## Split Audit

| window | train_rows | test_rows | train_unique_students | test_unique_students | overlapping_students |
| --- | --- | --- | --- | --- | --- |
| day_7 | 26074 | 6519 | 23029 | 5756 | 0 |
| day_14 | 26074 | 6519 | 23029 | 5756 | 0 |
| day_28 | 26074 | 6519 | 23029 | 5756 | 0 |
| day_56 | 26074 | 6519 | 23029 | 5756 | 0 |
| full | 26074 | 6519 | 23029 | 5756 | 0 |

## Summary of Changes Made in This Audit

- Changed the primary stratified train-test split from instance-level splitting to `StratifiedGroupKFold` grouped by `id_student`.
- Made `days_before_start_registered` prediction-window aware; future registration dates are now missing rather than used as known features.
- Excluded `vle_prediction_day` from model feature groups because it is an internal helper, not a behavioral feature.
- Expanded this report from a short leakage checklist into a reviewer-style risk register with code evidence, changes, and limitations.

## Remaining Limitations

- The main stratified split is student-disjoint, while GroupKFold and leave-one-course-presentation-out are course-disjoint but can include the same student in different folds if the student appears in multiple course presentations.
- The full-course window uses all within-course interactions and assessments and should be treated only as an upper-bound comparison.
- The cohort still includes students who may have withdrawn before a prediction day; excluding those students should be a sensitivity experiment.
- The pipeline reports discrimination metrics but does not yet report calibration, confidence intervals, or decision-threshold utility.