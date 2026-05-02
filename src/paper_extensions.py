from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .build_features import build_features_for_window, get_feature_group_columns
from .config import (
    FIGURES_DIR,
    PREDICTION_WINDOWS,
    RANDOM_STATE,
    REPORTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .evaluate import compute_metrics, get_prediction_scores
from .load_data import load_raw_data
from .train_models import find_model_spec, fit_evaluate
from .utils import configure_threading, ensure_output_dirs, set_global_seed


def stratified_student_group_split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = frame[TARGET_COLUMN].astype(int)
    groups = frame["id_student"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=max(2, round(1 / TEST_SIZE)),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return next(splitter.split(np.zeros(len(frame)), y, groups))


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns = get_feature_group_columns(frame)["demographic_vle_assessment"]
    return frame[columns].copy(), frame[TARGET_COLUMN].astype(int).copy()


def train_xgboost(frame: pd.DataFrame):
    x, y = split_xy(frame)
    train_idx, test_idx = stratified_student_group_split(frame)
    spec = find_model_spec("XGBoost", y.iloc[train_idx])
    model, metrics, y_pred, y_score = fit_evaluate(
        spec,
        x.iloc[train_idx],
        y.iloc[train_idx],
        x.iloc[test_idx],
        y.iloc[test_idx],
    )
    return model, metrics, y.iloc[test_idx].to_numpy(), y_pred, y_score, test_idx


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_STATE,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    values = {metric: [] for metric in metrics}
    n = len(y_true)
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[sample_idx])) < 2:
            continue
        sample_metrics = compute_metrics(
            y_true[sample_idx],
            y_pred[sample_idx],
            y_score[sample_idx],
        )
        for metric in metrics:
            value = sample_metrics[metric]
            if not math.isnan(float(value)):
                values[metric].append(float(value))

    ci = {}
    for metric, metric_values in values.items():
        if not metric_values:
            ci[metric] = (math.nan, math.nan)
        else:
            ci[metric] = (
                float(np.percentile(metric_values, 2.5)),
                float(np.percentile(metric_values, 97.5)),
            )
    return ci


def run_bootstrap_time_windows(
    feature_frames: dict[str, pd.DataFrame], n_bootstrap: int
) -> pd.DataFrame:
    rows = []
    for window_name, frame in feature_frames.items():
        print(f"[extensions] bootstrap CI for {window_name}", flush=True)
        _, metrics, y_true, y_pred, y_score, _ = train_xgboost(frame)
        ci = bootstrap_ci(y_true, y_pred, y_score, n_bootstrap=n_bootstrap)
        row = {
            "prediction_window": window_name,
            "model": "XGBoost",
            "n_test": int(len(y_true)),
            "n_bootstrap": int(n_bootstrap),
        }
        for metric, value in metrics.items():
            if metric in {"tn", "fp", "fn", "tp"}:
                row[metric] = int(value)
            else:
                row[metric] = float(value)
                row[f"{metric}_ci_lower"] = ci[metric][0]
                row[f"{metric}_ci_upper"] = ci[metric][1]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "bootstrap_ci_time_window_results.csv", index=False)
    return result


def run_per_course_leave_one_out(frame: pd.DataFrame) -> pd.DataFrame:
    print("[extensions] per-course leave-one-course-presentation-out", flush=True)
    x, y = split_xy(frame)
    groups = frame["course_presentation"].astype(str).to_numpy()
    splitter = LeaveOneGroupOut()
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        group_name = sorted(set(groups[test_idx]))[0]
        print(f"[extensions] LOO {fold_id}: {group_name}", flush=True)
        spec = find_model_spec("XGBoost", y.iloc[train_idx])
        _, metrics, _, _ = fit_evaluate(
            spec,
            x.iloc[train_idx],
            y.iloc[train_idx],
            x.iloc[test_idx],
            y.iloc[test_idx],
        )
        rows.append(
            {
                "fold": fold_id,
                "held_out_course_presentation": group_name,
                "n_test": int(len(test_idx)),
                "at_risk_rate": float(y.iloc[test_idx].mean()),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "per_course_presentation_results.csv", index=False)
    return result


def add_unregistration_date(
    data: dict[str, pd.DataFrame], frame: pd.DataFrame
) -> pd.DataFrame:
    registration = data["studentRegistration"][
        ["code_module", "code_presentation", "id_student", "date_unregistration"]
    ].copy()
    result = frame.merge(
        registration,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    return result


def run_withdrawal_sensitivity(
    data: dict[str, pd.DataFrame], feature_frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for window_name, frame in feature_frames.items():
        print(f"[extensions] withdrawal sensitivity for {window_name}", flush=True)
        with_unregistration = add_unregistration_date(data, frame)
        before_n = len(with_unregistration)
        early_withdrawn = with_unregistration["date_unregistration"].notna() & (
            with_unregistration["date_unregistration"] <= with_unregistration["prediction_day"]
        )
        filtered = with_unregistration.loc[~early_withdrawn].drop(columns=["date_unregistration"])
        after_n = len(filtered)
        if filtered[TARGET_COLUMN].nunique() < 2:
            continue
        _, metrics, _, _, _, _ = train_xgboost(filtered)
        rows.append(
            {
                "prediction_window": window_name,
                "model": "XGBoost",
                "cohort": "exclude_withdrawn_on_or_before_prediction_day",
                "n_before_filter": int(before_n),
                "n_after_filter": int(after_n),
                "n_excluded": int(before_n - after_n),
                "at_risk_rate_after_filter": float(filtered[TARGET_COLUMN].mean()),
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "withdrawal_sensitivity_results.csv", index=False)
    return result


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> tuple[float, pd.DataFrame]:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1], right=True)
    rows = []
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        mean_pred = float(np.mean(y_score[mask]))
        observed = float(np.mean(y_true[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(mean_pred - observed)
        rows.append(
            {
                "bin": bin_id + 1,
                "bin_lower": float(bins[bin_id]),
                "bin_upper": float(bins[bin_id + 1]),
                "n": int(np.sum(mask)),
                "mean_predicted_risk": mean_pred,
                "observed_at_risk_rate": observed,
                "absolute_gap": abs(mean_pred - observed),
            }
        )
    return float(ece), pd.DataFrame(rows)


def run_calibration_and_threshold(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[extensions] calibration and threshold analysis", flush=True)
    model, metrics, y_true, y_pred, y_score, _ = train_xgboost(frame)

    brier = float(brier_score_loss(y_true, y_score))
    ece, calibration_bins = expected_calibration_error(y_true, y_score, n_bins=10)
    calibration_bins.to_csv(TABLES_DIR / "calibration_bins_day56.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "prediction_window": "day_56",
                "model": "XGBoost",
                "brier_score": brier,
                "expected_calibration_error_10_bins": ece,
                "mean_predicted_risk": float(np.mean(y_score)),
                "observed_at_risk_rate": float(np.mean(y_true)),
                **metrics,
            }
        ]
    )
    summary.to_csv(TABLES_DIR / "calibration_summary_day56.csv", index=False)

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="XGBoost")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed at-risk rate")
    ax.set_title("Calibration Curve (day 56)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve_day56.png", dpi=200)
    plt.close(fig)

    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        threshold_pred = (y_score >= threshold).astype(int)
        threshold_metrics = compute_metrics(y_true, threshold_pred, y_score)
        threshold_rows.append({"threshold": round(float(threshold), 2), **threshold_metrics})
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(TABLES_DIR / "threshold_analysis_day56.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(threshold_frame["threshold"], threshold_frame["precision"], marker="o", label="Precision")
    ax.plot(threshold_frame["threshold"], threshold_frame["recall"], marker="o", label="Recall")
    ax.plot(threshold_frame["threshold"], threshold_frame["f1"], marker="o", label="F1")
    ax.plot(
        threshold_frame["threshold"],
        threshold_frame["balanced_accuracy"],
        marker="o",
        label="Balanced accuracy",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Threshold Trade-off (day 56)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_tradeoff_day56.png", dpi=200)
    plt.close(fig)
    return summary, threshold_frame


def write_readiness_report(
    bootstrap: pd.DataFrame,
    per_course: pd.DataFrame,
    sensitivity: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    day56_bootstrap = bootstrap[bootstrap["prediction_window"] == "day_56"].iloc[0]
    per_course_pr_min = float(per_course["pr_auc"].min())
    per_course_pr_max = float(per_course["pr_auc"].max())
    per_course_pr_mean = float(per_course["pr_auc"].mean())
    day56_sensitivity = sensitivity[sensitivity["prediction_window"] == "day_56"].iloc[0]
    calibration_row = calibration.iloc[0]

    lines = [
        "# EI 投稿准备度评估与补充实验",
        "",
        "## 1. 结论",
        "",
        "在原有实验基础上，当前项目已经更接近 EI 论文投稿所需的实验完整性。原结果已经包含多模型比较、多窗口预测、特征消融、SHAP 解释和泄漏审计；扩展实验新增了 bootstrap 置信区间、逐课程开课学期留一验证明细、预测日前已退课样本排除敏感性分析，以及校准与阈值分析。这些补充能够更好回应审稿人关于稳健性、跨课程泛化和实际预警阈值的质疑。",
        "",
        "多随机种子重复实验和统计显著性检验由 `src.repeated_significance.py` 生成，结果见 outputs/tables/repeated_seed_model_summary.csv、outputs/tables/repeated_seed_window_summary.csv 和 outputs/tables/significance_tests.csv。若重新运行本扩展脚本，建议随后重新运行 `.venv/bin/python -m src.repeated_significance --n-seeds 10` 以刷新统计检验报告。",
        "",
        "## 2. 新增结果文件",
        "",
        "- outputs/tables/bootstrap_ci_time_window_results.csv：多窗口指标 bootstrap 95% 置信区间。",
        "- outputs/tables/per_course_presentation_results.csv：逐课程开课学期 leave-one-course-presentation-out 明细。",
        "- outputs/tables/withdrawal_sensitivity_results.csv：排除预测日前已退课样本后的敏感性分析。",
        "- outputs/tables/calibration_summary_day56.csv：第 56 天 XGBoost 校准汇总。",
        "- outputs/tables/calibration_bins_day56.csv：第 56 天校准分箱明细。",
        "- outputs/tables/threshold_analysis_day56.csv：第 56 天不同阈值下的 precision、recall、F1 和 balanced accuracy。",
        "- outputs/tables/repeated_seed_model_summary.csv：多随机种子模型比较汇总，由 `src.repeated_significance.py` 生成。",
        "- outputs/tables/repeated_seed_window_summary.csv：多随机种子窗口比较汇总，由 `src.repeated_significance.py` 生成。",
        "- outputs/tables/significance_tests.csv：配对 Wilcoxon 检验与 Holm 校正结果，由 `src.repeated_significance.py` 生成。",
        "- outputs/figures/calibration_curve_day56.png：校准曲线。",
        "- outputs/figures/threshold_tradeoff_day56.png：阈值权衡曲线。",
        "",
        "## 3. 关键补充发现",
        "",
        f"第 56 天 XGBoost 的 PR-AUC 为 {day56_bootstrap['pr_auc']:.4f}，bootstrap 95% CI 为 [{day56_bootstrap['pr_auc_ci_lower']:.4f}, {day56_bootstrap['pr_auc_ci_upper']:.4f}]；F1 为 {day56_bootstrap['f1']:.4f}，95% CI 为 [{day56_bootstrap['f1_ci_lower']:.4f}, {day56_bootstrap['f1_ci_upper']:.4f}]（outputs/tables/bootstrap_ci_time_window_results.csv）。",
        "",
        f"逐课程开课学期留一验证显示，不同课程开课学期之间存在明显性能差异。PR-AUC 的均值为 {per_course_pr_mean:.4f}，最小值为 {per_course_pr_min:.4f}，最大值为 {per_course_pr_max:.4f}（outputs/tables/per_course_presentation_results.csv）。这说明跨课程泛化并不均匀，论文中应避免只报告总体均值。",
        "",
        f"在第 56 天排除预测日前已退课样本后，样本量从 {int(day56_sensitivity['n_before_filter'])} 变为 {int(day56_sensitivity['n_after_filter'])}，排除 {int(day56_sensitivity['n_excluded'])} 个实例；该敏感性设置下 PR-AUC = {day56_sensitivity['pr_auc']:.4f}，F1 = {day56_sensitivity['f1']:.4f}（outputs/tables/withdrawal_sensitivity_results.csv）。该结果应作为对退课样本影响的稳健性补充。",
        "",
        f"第 56 天 XGBoost 的 Brier score 为 {calibration_row['brier_score']:.4f}，10-bin ECE 为 {calibration_row['expected_calibration_error_10_bins']:.4f}（outputs/tables/calibration_summary_day56.csv; outputs/figures/calibration_curve_day56.png）。阈值分析结果保存在 outputs/tables/threshold_analysis_day56.csv，可用于讨论不同预警策略下 precision 与 recall 的权衡。",
        "",
        "## 4. 对论文写作的影响",
        "",
        "建议在论文实验部分新增一个“稳健性与实际部署分析”小节，依次报告置信区间、跨课程开课学期明细、退课敏感性分析和阈值/校准分析；同时新增“重复实验与统计检验”小节，引用 `src.repeated_significance.py` 生成的结果。这样可以将当前工作从单次性能报告提升为更完整的实验评估。",
        "",
        "## 5. 仍需谨慎表述的内容",
        "",
        "- 不应声称模型具有因果解释能力；SHAP 仅说明模型关联性贡献。",
        "- 完整课程窗口只能作为性能上界，不是早期预警结果。",
        "- 当前未进行超参数搜索，模型设置以可复现和合理运行时间为主。",
        "- 多随机种子重复实验主要反映不同学生分组随机划分造成的方差，不能替代外部数据验证或嵌套超参数优化。",
    ]
    (REPORTS_DIR / "paper_readiness_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def run_extensions(n_bootstrap: int = 1000) -> None:
    start = time.time()
    configure_threading()
    ensure_output_dirs()
    set_global_seed()

    data = load_raw_data()
    feature_frames = {}
    for window_name, cutoff_day in PREDICTION_WINDOWS.items():
        print(f"[extensions] building {window_name}", flush=True)
        feature_frames[window_name] = build_features_for_window(data, window_name, cutoff_day)

    bootstrap = run_bootstrap_time_windows(feature_frames, n_bootstrap=n_bootstrap)
    per_course = run_per_course_leave_one_out(feature_frames["day_56"])
    sensitivity = run_withdrawal_sensitivity(data, feature_frames)
    calibration, _ = run_calibration_and_threshold(feature_frames["day_56"])
    write_readiness_report(bootstrap, per_course, sensitivity, calibration)
    print(f"[extensions] complete in {(time.time() - start) / 60:.2f} minutes", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper robustness extension analyses.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extensions(n_bootstrap=args.n_bootstrap)
