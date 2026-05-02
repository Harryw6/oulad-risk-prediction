from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold

from .build_features import build_features_for_window, get_feature_group_columns
from .config import (
    FIGURES_DIR,
    GROUP_COLUMN,
    MAIN_COMPARISON_WINDOW,
    PREDICTION_WINDOWS,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .evaluate import (
    compute_metrics,
    get_prediction_scores,
    plot_confusion_matrix,
    plot_label_distribution,
    plot_roc_pr_curves,
    plot_time_window_performance,
)
from .leakage_audit import write_leakage_audit
from .load_data import create_dataset_summary, create_modeling_base, load_raw_data
from .shap_analysis import run_shap_analysis
from .train_models import find_model_spec, fit_evaluate, get_model_specs
from .utils import configure_threading, ensure_output_dirs, set_global_seed


def _split_xy(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return frame[columns].copy(), frame[TARGET_COLUMN].astype(int).copy()


def _stratified_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = frame[TARGET_COLUMN].astype(int)
    groups = frame["id_student"].to_numpy()
    n_splits = max(2, round(1 / TEST_SIZE))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return next(splitter.split(np.zeros(len(frame)), y, groups))


def _split_overlap_summary(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, int]:
    train_students = set(frame.iloc[train_idx]["id_student"].astype(int))
    test_students = set(frame.iloc[test_idx]["id_student"].astype(int))
    return {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_unique_students": int(len(train_students)),
        "test_unique_students": int(len(test_students)),
        "overlapping_students": int(len(train_students.intersection(test_students))),
    }


def _choose_best_tree(model_results: pd.DataFrame) -> str:
    tree_names = {"Decision Tree", "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Stacking Ensemble"}
    candidates = model_results[model_results["model"].isin(tree_names)].copy()
    if candidates.empty:
        return "Random Forest"
    candidates = candidates.sort_values(["pr_auc", "roc_auc", "f1"], ascending=False)
    best_pr_auc = float(candidates.iloc[0]["pr_auc"])
    near_best = candidates[candidates["pr_auc"] >= best_pr_auc - 0.001]
    for stable_model in ["XGBoost", "LightGBM", "Random Forest", "CatBoost", "Decision Tree"]:
        if stable_model in set(near_best["model"]):
            return stable_model
    return str(candidates.iloc[0]["model"])


def _run_group_validation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    validation_name: str,
    leave_one_group_out: bool,
) -> dict[str, object]:
    x, y = _split_xy(frame, feature_columns)
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if leave_one_group_out:
        splitter = LeaveOneGroupOut()
        splits = splitter.split(x, y, groups)
    else:
        splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
        splits = splitter.split(x, y, groups)

    fold_rows = []
    confusion = np.array([0, 0, 0, 0], dtype=int)
    total_folds = len(unique_groups) if leave_one_group_out else min(5, len(unique_groups))
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        print(f"[validation] {validation_name} fold {fold_id}/{total_folds}", flush=True)
        if len(np.unique(y.iloc[train_idx])) < 2:
            continue
        spec = find_model_spec(model_name, y.iloc[train_idx])
        model, metrics, _, _ = fit_evaluate(
            spec,
            x.iloc[train_idx],
            y.iloc[train_idx],
            x.iloc[test_idx],
            y.iloc[test_idx],
        )
        metrics["fold"] = fold_id
        metrics["test_groups"] = ",".join(sorted(set(groups[test_idx])))
        fold_rows.append(metrics)
        confusion += np.array([metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]])

    fold_frame = pd.DataFrame(fold_rows)
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    ]
    summary = {metric: float(fold_frame[metric].mean()) for metric in metric_columns}
    summary.update(
        {
            "validation": validation_name,
            "model": model_name,
            "n_folds": len(fold_frame),
            "tn": int(confusion[0]),
            "fp": int(confusion[1]),
            "fn": int(confusion[2]),
            "tp": int(confusion[3]),
        }
    )
    return summary


def _write_experiment_summary(
    output_path,
    model_comparison: pd.DataFrame,
    time_window_results: pd.DataFrame,
    feature_ablation_results: pd.DataFrame,
    best_model_name: str,
    best_tree_name: str,
    elapsed_seconds: float,
) -> None:
    best_model = model_comparison.sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).iloc[0]
    best_window = time_window_results.sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).iloc[0]
    best_ablation = feature_ablation_results.sort_values(
        ["pr_auc", "roc_auc", "f1"], ascending=False
    ).iloc[0]

    lines = [
        "# Experiment Summary",
        "",
        "## Objective",
        "Predict academically at-risk students in OULAD using multi-window demographic, VLE, and assessment features.",
        "",
        "## Label",
        "`at_risk = 1` for Fail or Withdrawn; `at_risk = 0` for Pass or Distinction.",
        "",
        "## Implementation Decisions",
        "",
        "- The main model comparison uses the 56-day early-warning window with all feature groups combined.",
        "- The selected tree-based model is reused for time-window and feature-ablation experiments; when tree models are statistically tied within 0.001 PR-AUC, XGBoost/LightGBM are preferred for stable behavior on sparse early-window ablations.",
        "- Registration timing is treated as a window-aware administrative feature; registration dates after the prediction day are encoded as missing.",
        "- The full-course window is reported as an upper-bound comparison rather than an early-warning result.",
        "- Optional libraries are included when importable; this run included the models present in `model_comparison.csv`.",
        "",
        "## Best Results",
        "",
        f"- Best model comparison row: {best_model['model']} with PR-AUC={best_model['pr_auc']:.4f}, ROC-AUC={best_model['roc_auc']:.4f}, F1={best_model['f1']:.4f}.",
        f"- Best tree-based model selected for window/ablation experiments: {best_tree_name}.",
        f"- Best time-window result: {best_window['prediction_window']} with PR-AUC={best_window['pr_auc']:.4f}, ROC-AUC={best_window['roc_auc']:.4f}, F1={best_window['f1']:.4f}.",
        f"- Best ablation result: {best_ablation['prediction_window']} / {best_ablation['feature_group']} with PR-AUC={best_ablation['pr_auc']:.4f}.",
        "",
        "## Validation Protocols",
        "",
        "- Stratified train-test split is grouped by `id_student` to avoid repeated-student overlap.",
        "- GroupKFold groups by `code_module + code_presentation` and is summarized in `model_comparison.csv`.",
        "- Leave-one-course-presentation-out is summarized in `model_comparison.csv`.",
        "",
        "## Runtime",
        "",
        f"Completed in {elapsed_seconds / 60:.2f} minutes.",
        "",
        "## Known Limitations",
        "",
        "- Students who withdraw before a prediction day remain in the dataset because the target definition includes Withdrawn; the withdrawal date itself is never used.",
        "- No hyperparameter optimization was run; model settings are conservative defaults for reproducibility and runtime.",
        "- SHAP is computed on a held-out sample of at most 1,200 rows for runtime.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(skip_shap: bool = False) -> None:
    start = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    configure_threading()
    ensure_output_dirs()
    set_global_seed()

    data = load_raw_data()
    base = create_modeling_base(data)
    create_dataset_summary(data, base).to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
    plot_label_distribution(base, FIGURES_DIR / "label_distribution.png")

    feature_frames: dict[str, pd.DataFrame] = {}
    full_feature_columns_by_window: dict[str, list[str]] = {}
    for window_name, cutoff_day in PREDICTION_WINDOWS.items():
        print(f"[features] building {window_name}", flush=True)
        frame = build_features_for_window(data, window_name, cutoff_day)
        feature_frames[window_name] = frame
        full_feature_columns_by_window[window_name] = get_feature_group_columns(frame)[
            "demographic_vle_assessment"
        ]

    split_audit = {
        window_name: _split_overlap_summary(frame, *_stratified_indices(frame))
        for window_name, frame in feature_frames.items()
    }
    write_leakage_audit(
        REPORTS_DIR / "leakage_audit.md",
        RAW_DATA_DIR,
        feature_frames,
        full_feature_columns_by_window,
        split_audit,
    )

    comparison_frame = feature_frames[MAIN_COMPARISON_WINDOW]
    feature_groups = get_feature_group_columns(comparison_frame)
    main_columns = feature_groups["demographic_vle_assessment"]
    x, y = _split_xy(comparison_frame, main_columns)
    train_idx, test_idx = _stratified_indices(comparison_frame)
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_rows = []
    curve_rows = []
    fitted_models = {}
    predictions = {}
    print("[models] running stratified model comparison", flush=True)
    for spec in get_model_specs(y_train, include_stacking=True):
        print(f"[models] fitting {spec.name}", flush=True)
        model, metrics, y_pred, y_score = fit_evaluate(spec, x_train, y_train, x_test, y_test)
        row = {
            "validation": "stratified_student_group_train_test",
            "prediction_window": MAIN_COMPARISON_WINDOW,
            "feature_group": "demographic_vle_assessment",
            "model": spec.name,
            **metrics,
        }
        model_rows.append(row)
        fitted_models[spec.name] = model
        predictions[spec.name] = (y_pred, y_score)
        curve_rows.append({"label": spec.name, "y_true": y_test.to_numpy(), "y_score": y_score})

    stratified_results = pd.DataFrame(model_rows)
    best_model_name = str(
        stratified_results.sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).iloc[0]["model"]
    )
    best_tree_name = _choose_best_tree(stratified_results)

    print("[validation] running grouped validation", flush=True)
    grouped_summaries = [
        {
            "prediction_window": MAIN_COMPARISON_WINDOW,
            "feature_group": "demographic_vle_assessment",
            **_run_group_validation(
                comparison_frame,
                main_columns,
                best_tree_name,
                "groupkfold_course_presentation",
                leave_one_group_out=False,
            ),
        },
        {
            "prediction_window": MAIN_COMPARISON_WINDOW,
            "feature_group": "demographic_vle_assessment",
            **_run_group_validation(
                comparison_frame,
                main_columns,
                best_tree_name,
                "leave_one_course_presentation_out",
                leave_one_group_out=True,
            ),
        },
    ]

    model_comparison = pd.concat(
        [stratified_results, pd.DataFrame(grouped_summaries)], ignore_index=True
    )
    model_comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    plot_roc_pr_curves(
        curve_rows,
        FIGURES_DIR / "roc_curves.png",
        FIGURES_DIR / "pr_curves.png",
    )
    best_y_pred, _ = predictions[best_model_name]
    plot_confusion_matrix(
        y_test,
        best_y_pred,
        FIGURES_DIR / "confusion_matrix_best_model.png",
        f"{best_model_name} Confusion Matrix",
    )

    print("[windows] evaluating selected tree model across prediction windows", flush=True)
    time_rows = []
    for window_name, frame in feature_frames.items():
        columns = get_feature_group_columns(frame)["demographic_vle_assessment"]
        x_window, y_window = _split_xy(frame, columns)
        train_idx, test_idx = _stratified_indices(frame)
        spec = find_model_spec(best_tree_name, y_window.iloc[train_idx])
        _, metrics, _, _ = fit_evaluate(
            spec,
            x_window.iloc[train_idx],
            y_window.iloc[train_idx],
            x_window.iloc[test_idx],
            y_window.iloc[test_idx],
        )
        time_rows.append(
            {
                "validation": "stratified_student_group_train_test",
                "prediction_window": window_name,
                "feature_group": "demographic_vle_assessment",
                "model": best_tree_name,
                **metrics,
            }
        )
    time_window_results = pd.DataFrame(time_rows)
    time_window_results.to_csv(TABLES_DIR / "time_window_results.csv", index=False)
    plot_time_window_performance(
        time_window_results, FIGURES_DIR / "time_window_performance.png"
    )

    print("[ablation] evaluating feature groups", flush=True)
    ablation_rows = []
    for window_name, frame in feature_frames.items():
        groups = get_feature_group_columns(frame)
        for group_name, columns in groups.items():
            x_group, y_group = _split_xy(frame, columns)
            train_idx, test_idx = _stratified_indices(frame)
            spec = find_model_spec(best_tree_name, y_group.iloc[train_idx])
            _, metrics, _, _ = fit_evaluate(
                spec,
                x_group.iloc[train_idx],
                y_group.iloc[train_idx],
                x_group.iloc[test_idx],
                y_group.iloc[test_idx],
            )
            ablation_rows.append(
                {
                    "validation": "stratified_student_group_train_test",
                    "prediction_window": window_name,
                    "feature_group": group_name,
                    "model": best_tree_name,
                    **metrics,
                }
            )
    feature_ablation_results = pd.DataFrame(ablation_rows)
    feature_ablation_results.to_csv(TABLES_DIR / "feature_ablation_results.csv", index=False)

    if not skip_shap:
        print("[shap] running explainability analysis", flush=True)
        best_tree_model = fitted_models.get(best_tree_name)
        if best_tree_model is None:
            spec = find_model_spec(best_tree_name, y_train)
            best_tree_model, _, _, _ = fit_evaluate(spec, x_train, y_train, x_test, y_test)
        identifiers = comparison_frame.iloc[test_idx][
            ["code_module", "code_presentation", "id_student", TARGET_COLUMN]
        ].reset_index(drop=True)
        try:
            run_shap_analysis(
                best_tree_model,
                x_test.reset_index(drop=True),
                y_test.reset_index(drop=True),
                identifiers,
                TABLES_DIR / "shap_top_features.csv",
                FIGURES_DIR / "shap_summary.png",
                FIGURES_DIR,
                TABLES_DIR / "shap_local_high_risk.csv",
            )
        except Exception as exc:
            pd.DataFrame(
                [{"feature": "SHAP_FAILED", "mean_abs_shap": np.nan, "error": str(exc)}]
            ).to_csv(TABLES_DIR / "shap_top_features.csv", index=False)
            (REPORTS_DIR / "shap_failure.txt").write_text(str(exc), encoding="utf-8")
            blank = pd.DataFrame({"x": [0], "y": [0]})
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "SHAP failed; see outputs/reports/shap_failure.txt", ha="center")
            ax.axis("off")
            fig.savefig(FIGURES_DIR / "shap_summary.png", dpi=200)
            plt.close(fig)
            blank.to_csv(TABLES_DIR / "shap_local_high_risk.csv", index=False)

    _write_experiment_summary(
        REPORTS_DIR / "experiment_summary.md",
        model_comparison,
        time_window_results,
        feature_ablation_results,
        best_model_name,
        best_tree_name,
        time.time() - start,
    )

    print("[done] experiment complete", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OULAD at-risk prediction experiments.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(skip_shap=args.skip_shap)
