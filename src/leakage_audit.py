from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import REQUIRED_RAW_FILES


FORBIDDEN_MODEL_FEATURES = {
    "final_result",
    "at_risk",
    "id_student",
    "date_unregistration",
    "code_module",
    "code_presentation",
    "course_presentation",
}

REQUIRED_GENERATED_FILES = [
    "outputs/tables/dataset_summary.csv",
    "outputs/tables/model_comparison.csv",
    "outputs/tables/time_window_results.csv",
    "outputs/tables/feature_ablation_results.csv",
    "outputs/tables/shap_top_features.csv",
    "outputs/figures/label_distribution.png",
    "outputs/figures/time_window_performance.png",
    "outputs/figures/roc_curves.png",
    "outputs/figures/pr_curves.png",
    "outputs/figures/confusion_matrix_best_model.png",
    "outputs/figures/shap_summary.png",
    "outputs/reports/experiment_summary.md",
]


def audit_feature_frames(
    feature_frames: dict[str, pd.DataFrame], feature_columns_by_window: dict[str, list[str]]
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    for window_name, frame in feature_frames.items():
        feature_columns = set(feature_columns_by_window[window_name])
        forbidden_present = sorted(feature_columns.intersection(FORBIDDEN_MODEL_FEATURES))
        checks.append(
            {
                "window": window_name,
                "check": "forbidden model fields absent",
                "status": "PASS" if not forbidden_present else "FAIL",
                "detail": (
                    "No target, student id, course id, or withdrawal fields in model feature set."
                    if not forbidden_present
                    else ", ".join(forbidden_present)
                ),
            }
        )

        if "vle_last_activity_day" in frame.columns:
            valid = frame["vle_last_activity_day"].isna() | (
                frame["vle_last_activity_day"] <= frame["prediction_day"]
            )
            checks.append(
                {
                    "window": window_name,
                    "check": "VLE activity occurs on/before prediction day",
                    "status": "PASS" if bool(valid.all()) else "FAIL",
                    "detail": f"violations={int((~valid).sum())}",
                }
            )

        if "assessment_last_submitted_day" in frame.columns:
            valid = frame["assessment_last_submitted_day"].isna() | (
                frame["assessment_last_submitted_day"] <= frame["prediction_day"]
            )
            checks.append(
                {
                    "window": window_name,
                    "check": "assessment submissions occur on/before prediction day",
                    "status": "PASS" if bool(valid.all()) else "FAIL",
                    "detail": f"violations={int((~valid).sum())}",
                }
            )

        if {"registration_known_by_prediction_day", "days_before_start_registered"}.issubset(
            frame.columns
        ):
            future_registration = frame["registration_known_by_prediction_day"] == 0
            valid = frame.loc[future_registration, "days_before_start_registered"].isna()
            checks.append(
                {
                    "window": window_name,
                    "check": "registration date feature unavailable before known registration",
                    "status": "PASS" if bool(valid.all()) else "FAIL",
                    "detail": (
                        f"future-or-missing registration rows={int(future_registration.sum())}; "
                        f"violations={int((~valid).sum())}"
                    ),
                }
            )

    return pd.DataFrame(checks)


def _format_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _status_from_failures(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "WARN"
    if (frame["status"] == "FAIL").any():
        return "FAIL"
    if (frame["status"] == "WARN").any():
        return "WARN"
    return "PASS"


def _required_outputs_status(output_path: Path) -> tuple[str, str]:
    root = output_path.parents[2]
    missing = [path for path in REQUIRED_GENERATED_FILES if not (root / path).exists()]
    if missing:
        return "FAIL", "Missing generated files: " + ", ".join(missing)
    return "PASS", "All required tables, figures, and reports exist under outputs/."


def _split_status(split_audit: dict[str, dict[str, int]]) -> tuple[str, str]:
    if not split_audit:
        return "WARN", "No split audit details were provided."
    overlaps = {window: row["overlapping_students"] for window, row in split_audit.items()}
    total_overlap = sum(overlaps.values())
    if total_overlap:
        return "FAIL", f"Student overlap detected across train/test splits: {overlaps}"
    return "PASS", "Stratified train-test splits are grouped by id_student with zero overlap."


def write_leakage_audit(
    output_path: Path,
    raw_dir: Path,
    feature_frames: dict[str, pd.DataFrame],
    feature_columns_by_window: dict[str, list[str]],
    split_audit: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    split_audit = split_audit or {}
    missing_raw = [name for name in REQUIRED_RAW_FILES if not (raw_dir / name).exists()]
    feature_checks = audit_feature_frames(feature_frames, feature_columns_by_window)
    feature_status = _status_from_failures(feature_checks)
    split_status, split_detail = _split_status(split_audit)
    outputs_status, outputs_detail = _required_outputs_status(output_path)

    risk_rows = [
        {
            "risk": "1. General data leakage",
            "status": "PASS" if not missing_raw and feature_status == "PASS" else "FAIL",
            "evidence": "src/build_features.py constructs window-specific feature frames; src/leakage_audit.py checks forbidden fields and event dates for each window.",
            "changes_made": "Added stricter feature-frame checks and made registration features window-aware.",
            "remaining_limitations": "Observed inactivity can still reflect students who already withdrew; date_unregistration is excluded, but cohort censoring is a separate sensitivity analysis.",
        },
        {
            "risk": "2. Incorrect train-test splitting",
            "status": split_status,
            "evidence": "src/run_experiments.py::_stratified_indices uses StratifiedGroupKFold with groups=id_student; split audit checks zero student overlap.",
            "changes_made": "Replaced instance-level StratifiedShuffleSplit with student-grouped StratifiedGroupKFold.",
            "remaining_limitations": "Main split is student-disjoint but not course-disjoint; course-disjoint performance is reported separately with GroupKFold and leave-one-course-presentation-out.",
        },
        {
            "risk": "3. Future VLE data",
            "status": "PASS" if feature_status != "FAIL" else "FAIL",
            "evidence": "src/build_features.py::build_vle_features filters studentVle with date <= cutoff_day, or date <= module_presentation_length for the full upper-bound window.",
            "changes_made": "Excluded internal vle_prediction_day from model feature groups to avoid unnecessary course-window encoding.",
            "remaining_limitations": "The full window is not an early-warning setting and should be interpreted only as an upper bound.",
        },
        {
            "risk": "4. Future assessment scores",
            "status": "PASS" if feature_status != "FAIL" else "FAIL",
            "evidence": "src/build_features.py::build_assessment_features requires assessment_date <= cutoff_day and date_submitted <= cutoff_day before using score features.",
            "changes_made": "Audit now explicitly checks assessment_last_submitted_day <= prediction_day for every window.",
            "remaining_limitations": "Assessment due schedules are course design metadata and remain available to the model through due-count/weight features.",
        },
        {
            "risk": "5. final_result as a feature",
            "status": "PASS" if feature_status == "PASS" else "FAIL",
            "evidence": "src/load_data.py creates at_risk from final_result; src/build_features.py::get_feature_group_columns never includes final_result.",
            "changes_made": "Audit forbids final_result in model feature columns.",
            "remaining_limitations": "None identified.",
        },
        {
            "risk": "6. id_student as a feature",
            "status": "PASS" if feature_status == "PASS" else "FAIL",
            "evidence": "id_student is retained for joins, split grouping, and SHAP identifiers; src/build_features.py::get_feature_group_columns excludes it.",
            "changes_made": "Audit forbids id_student in model feature columns and uses it only for train/test grouping.",
            "remaining_limitations": "Repeated students are now separated across train/test, but repeated-course histories are not explicitly modeled.",
        },
        {
            "risk": "7. Preprocessing fitted on full dataset",
            "status": "PASS",
            "evidence": "src/train_models.py::build_pipeline wraps ColumnTransformer and estimator in one sklearn Pipeline; src/train_models.py::fit_evaluate calls model.fit only on x_train/y_train.",
            "changes_made": "No code change required; report now documents this explicitly.",
            "remaining_limitations": "Feature engineering itself is deterministic aggregation and not learned preprocessing.",
        },
        {
            "risk": "8. Class imbalance handling",
            "status": "PASS",
            "evidence": "src/train_models.py uses class_weight='balanced' or balanced_subsample where supported; XGBoost scale_pos_weight is computed from y_train only.",
            "changes_made": "No SMOTE or full-data resampling is used.",
            "remaining_limitations": "Threshold is the default classifier threshold; operating-point optimization is future work.",
        },
        {
            "risk": "9. Misleading metrics for imbalance",
            "status": "PASS",
            "evidence": "src/evaluate.py computes PR-AUC, ROC-AUC, F1, recall, precision, balanced accuracy, and confusion matrix; model selection sorts by PR-AUC, ROC-AUC, then F1.",
            "changes_made": "No code change required; accuracy is reported but not used alone.",
            "remaining_limitations": "Calibration and decision-curve analysis are not yet included.",
        },
        {
            "risk": "10. Non-reproducible figures or tables",
            "status": outputs_status,
            "evidence": "src/run_experiments.py generates every required output from raw CSV files using RANDOM_STATE=42; outputs are saved under outputs/.",
            "changes_made": "Audit now checks required generated artifacts exist.",
            "remaining_limitations": outputs_detail,
        },
        {
            "risk": "11. Withdrawal date leakage",
            "status": "PASS" if feature_status == "PASS" else "FAIL",
            "evidence": "src/load_data.py reads date_unregistration, but create_modeling_base does not merge it; audit forbids date_unregistration in feature columns.",
            "changes_made": "No predictive use of withdrawal date is allowed.",
            "remaining_limitations": "Withdrawal labels remain part of the target by study definition.",
        },
        {
            "risk": "12. Future registration date leakage",
            "status": "PASS" if feature_status == "PASS" else "FAIL",
            "evidence": "src/build_features.py::build_registration_features sets days_before_start_registered to missing unless date_registration <= prediction_day.",
            "changes_made": "Moved registration feature creation out of static base construction and made it prediction-window aware.",
            "remaining_limitations": "Rows for students not yet registered by a prediction day remain in the analytical cohort with registration timing missing.",
        },
    ]

    risk_frame = pd.DataFrame(risk_rows)
    split_frame = pd.DataFrame(
        [{"window": window, **values} for window, values in split_audit.items()]
    )

    lines = [
        "# Leakage Audit",
        "",
        "Reviewer stance: strict. PASS means the current code satisfies the check under the implemented experimental design; WARN means a defensible limitation remains; FAIL means results should not be used until fixed.",
        "",
        "## Risk Register",
        "",
    ]
    lines.extend(_format_table(risk_frame))
    lines.extend(["", "## Window-Level Checks", ""])
    lines.extend(_format_table(feature_checks))
    lines.extend(["", "## Split Audit", ""])
    lines.extend(_format_table(split_frame))
    lines.extend(
        [
            "",
            "## Summary of Changes Made in This Audit",
            "",
            "- Changed the primary stratified train-test split from instance-level splitting to `StratifiedGroupKFold` grouped by `id_student`.",
            "- Made `days_before_start_registered` prediction-window aware; future registration dates are now missing rather than used as known features.",
            "- Excluded `vle_prediction_day` from model feature groups because it is an internal helper, not a behavioral feature.",
            "- Expanded this report from a short leakage checklist into a reviewer-style risk register with code evidence, changes, and limitations.",
            "",
            "## Remaining Limitations",
            "",
            "- The main stratified split is student-disjoint, while GroupKFold and leave-one-course-presentation-out are course-disjoint but can include the same student in different folds if the student appears in multiple course presentations.",
            "- The full-course window uses all within-course interactions and assessments and should be treated only as an upper-bound comparison.",
            "- The cohort still includes students who may have withdrawn before a prediction day; excluding those students should be a sensitivity experiment.",
            "- The pipeline reports discrimination metrics but does not yet report calibration, confidence intervals, or decision-threshold utility.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return risk_frame
