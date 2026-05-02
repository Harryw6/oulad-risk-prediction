from __future__ import annotations

import argparse
import math
import time
import warnings
from collections.abc import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold

from .build_features import build_features_for_window, get_feature_group_columns
from .config import (
    FIGURES_DIR,
    MAIN_COMPARISON_WINDOW,
    PREDICTION_WINDOWS,
    RANDOM_STATE,
    REPORTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .load_data import load_raw_data
from .train_models import find_model_spec, fit_evaluate, get_model_specs
from .utils import configure_threading, ensure_output_dirs, set_global_seed


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
]

DEFAULT_MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Linear SVM",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns = get_feature_group_columns(frame)["demographic_vle_assessment"]
    return frame[columns].copy(), frame[TARGET_COLUMN].astype(int).copy()


def stratified_student_group_split(
    frame: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    y = frame[TARGET_COLUMN].astype(int)
    groups = frame["id_student"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=max(2, round(1 / TEST_SIZE)),
        shuffle=True,
        random_state=seed,
    )
    return next(splitter.split(np.zeros(len(frame)), y, groups))


def holm_adjust_pvalues(p_values: Iterable[float]) -> list[float]:
    values = [float(value) if pd.notna(value) else math.nan for value in p_values]
    adjusted = [math.nan] * len(values)
    finite = [(index, value) for index, value in enumerate(values) if math.isfinite(value)]
    if not finite:
        return adjusted

    finite.sort(key=lambda item: item[1])
    m = len(finite)
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(finite):
        correction_factor = m - rank
        corrected = min(1.0, p_value * correction_factor)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    return adjusted


def paired_wilcoxon_tests(
    frame: pd.DataFrame,
    item_column: str,
    baseline_item: str,
    comparison_items: list[str],
    metrics: list[str],
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        baseline = frame.loc[
            frame[item_column] == baseline_item, ["seed", metric]
        ].rename(columns={metric: "baseline_value"})
        for comparison_item in comparison_items:
            comparator = frame.loc[
                frame[item_column] == comparison_item, ["seed", metric]
            ].rename(columns={metric: "comparator_value"})
            paired = baseline.merge(comparator, on="seed", how="inner").dropna()
            n_pairs = int(len(paired))
            if n_pairs < 2:
                statistic = math.nan
                p_value = math.nan
                baseline_mean = math.nan
                comparator_mean = math.nan
                mean_delta = math.nan
            else:
                deltas = (
                    paired["baseline_value"].to_numpy(dtype=float)
                    - paired["comparator_value"].to_numpy(dtype=float)
                )
                baseline_mean = float(paired["baseline_value"].mean())
                comparator_mean = float(paired["comparator_value"].mean())
                mean_delta = float(np.mean(deltas))
                if np.allclose(deltas, 0.0):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    test = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
                    statistic = float(test.statistic)
                    p_value = float(test.pvalue)

            rows.append(
                {
                    "family": family,
                    "comparison": f"{baseline_item} vs {comparison_item}",
                    "baseline": baseline_item,
                    "comparator": comparison_item,
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "baseline_mean": baseline_mean,
                    "comparator_mean": comparator_mean,
                    "mean_delta": mean_delta,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        result["p_value_holm"] = []
        return result

    result["p_value_holm"] = math.nan
    for (family_name, metric), indexes in result.groupby(["family", "metric"]).groups.items():
        del family_name, metric
        adjusted = holm_adjust_pvalues(result.loc[indexes, "p_value"])
        result.loc[indexes, "p_value_holm"] = adjusted
    return result


def summarize_repeated_results(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    aggregations = {}
    for metric in METRIC_COLUMNS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")
    aggregations["n_repeats"] = ("seed", "nunique")
    result = frame.groupby(group_columns, dropna=False).agg(**aggregations).reset_index()
    return result


def available_model_names(y_train: pd.Series, include_stacking: bool) -> list[str]:
    available = [spec.name for spec in get_model_specs(y_train, include_stacking=include_stacking)]
    ordered = [name for name in DEFAULT_MODEL_ORDER if name in available]
    if include_stacking and "Stacking Ensemble" in available:
        ordered.append("Stacking Ensemble")
    return ordered


def run_repeated_model_comparison(
    frame: pd.DataFrame,
    seeds: list[int],
    include_stacking: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    x, y = split_xy(frame)
    for seed in seeds:
        train_idx, test_idx = stratified_student_group_split(frame, seed)
        model_names = available_model_names(y.iloc[train_idx], include_stacking=include_stacking)
        for model_name in model_names:
            print(f"[repeats] seed {seed} model {model_name}", flush=True)
            spec = find_model_spec(model_name, y.iloc[train_idx])
            _, metrics, _, _ = fit_evaluate(
                spec,
                x.iloc[train_idx],
                y.iloc[train_idx],
                x.iloc[test_idx],
                y.iloc[test_idx],
            )
            rows.append(
                {
                    "validation": "repeated_stratified_student_group_train_test",
                    "seed": int(seed),
                    "prediction_window": MAIN_COMPARISON_WINDOW,
                    "feature_group": "demographic_vle_assessment",
                    "model": model_name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_at_risk_rate": float(y.iloc[test_idx].mean()),
                    **metrics,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "repeated_seed_model_results.csv", index=False)
    summarize_repeated_results(result, ["prediction_window", "feature_group", "model"]).to_csv(
        TABLES_DIR / "repeated_seed_model_summary.csv", index=False
    )
    return result


def run_repeated_time_windows(
    feature_frames: dict[str, pd.DataFrame],
    seeds: list[int],
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for window_name, frame in feature_frames.items():
        x, y = split_xy(frame)
        for seed in seeds:
            print(f"[repeats] seed {seed} window {window_name} model {model_name}", flush=True)
            train_idx, test_idx = stratified_student_group_split(frame, seed)
            spec = find_model_spec(model_name, y.iloc[train_idx])
            _, metrics, _, _ = fit_evaluate(
                spec,
                x.iloc[train_idx],
                y.iloc[train_idx],
                x.iloc[test_idx],
                y.iloc[test_idx],
            )
            rows.append(
                {
                    "validation": "repeated_stratified_student_group_train_test",
                    "seed": int(seed),
                    "prediction_window": window_name,
                    "feature_group": "demographic_vle_assessment",
                    "model": model_name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_at_risk_rate": float(y.iloc[test_idx].mean()),
                    **metrics,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLES_DIR / "repeated_seed_window_results.csv", index=False)
    summarize_repeated_results(result, ["prediction_window", "feature_group", "model"]).to_csv(
        TABLES_DIR / "repeated_seed_window_summary.csv", index=False
    )
    return result


def build_significance_tests(
    model_results: pd.DataFrame,
    window_results: pd.DataFrame,
    baseline_model: str = "XGBoost",
) -> pd.DataFrame:
    model_names = sorted(set(model_results["model"]) - {baseline_model})
    model_tests = paired_wilcoxon_tests(
        frame=model_results,
        item_column="model",
        baseline_item=baseline_model,
        comparison_items=model_names,
        metrics=["pr_auc", "f1", "roc_auc", "balanced_accuracy"],
        family="day56_model_comparison",
    )

    earlier_windows = [
        window for window in ["day_7", "day_14", "day_28"] if window in set(window_results["prediction_window"])
    ]
    early_window_tests = paired_wilcoxon_tests(
        frame=window_results,
        item_column="prediction_window",
        baseline_item="day_56",
        comparison_items=earlier_windows,
        metrics=["pr_auc", "f1", "roc_auc", "balanced_accuracy"],
        family="xgboost_day56_vs_earlier_windows",
    )

    full_window_tests = pd.DataFrame()
    if "full" in set(window_results["prediction_window"]):
        full_window_tests = paired_wilcoxon_tests(
            frame=window_results,
            item_column="prediction_window",
            baseline_item="full",
            comparison_items=["day_56"],
            metrics=["pr_auc", "f1", "roc_auc", "balanced_accuracy"],
            family="xgboost_full_upper_bound_vs_day56",
        )

    result = pd.concat([model_tests, early_window_tests, full_window_tests], ignore_index=True)
    result.to_csv(TABLES_DIR / "significance_tests.csv", index=False)
    return result


def plot_repeated_model_pr_auc(model_results: pd.DataFrame) -> None:
    summary = summarize_repeated_results(
        model_results, ["prediction_window", "feature_group", "model"]
    )
    order = (
        summary.sort_values("pr_auc_mean", ascending=False)["model"].astype(str).tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=model_results, x="model", y="pr_auc", order=order, ax=ax, color="#8FB9A8")
    sns.stripplot(
        data=model_results,
        x="model",
        y="pr_auc",
        order=order,
        ax=ax,
        color="#2F4858",
        alpha=0.65,
        size=4,
    )
    ax.set_xlabel("")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Repeated-Seed Day-56 Model PR-AUC")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "repeated_seed_pr_auc_boxplot.png", dpi=200)
    plt.close(fig)


def plot_repeated_window_pr_auc(window_results: pd.DataFrame) -> None:
    order = ["day_7", "day_14", "day_28", "day_56", "full"]
    plot_frame = window_results.copy()
    plot_frame["prediction_window"] = pd.Categorical(
        plot_frame["prediction_window"], categories=order, ordered=True
    )
    plot_frame = plot_frame.sort_values(["prediction_window", "seed"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_frame,
        x="prediction_window",
        y="pr_auc",
        order=order,
        ax=ax,
        color="#C6D8D3",
    )
    sns.pointplot(
        data=plot_frame,
        x="prediction_window",
        y="pr_auc",
        order=order,
        ax=ax,
        color="#B23A48",
        errorbar=None,
        markers="o",
        linestyles="-",
    )
    ax.set_xlabel("Prediction window")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Repeated-Seed XGBoost PR-AUC by Window")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "repeated_seed_window_pr_auc.png", dpi=200)
    plt.close(fig)


def _format_float(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    selected = frame[columns].copy()
    if max_rows is not None:
        selected = selected.head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def write_significance_report(
    model_summary: pd.DataFrame,
    window_summary: pd.DataFrame,
    significance: pd.DataFrame,
    seeds: list[int],
    include_stacking: bool,
    elapsed_seconds: float,
) -> None:
    model_ranked = model_summary.sort_values(["pr_auc_mean", "f1_mean"], ascending=False)
    window_order = ["day_7", "day_14", "day_28", "day_56", "full"]
    window_ranked = window_summary.copy()
    window_ranked["prediction_window"] = pd.Categorical(
        window_ranked["prediction_window"], categories=window_order, ordered=True
    )
    window_ranked = window_ranked.sort_values("prediction_window")

    best_model = model_ranked.iloc[0]
    day56 = window_ranked[window_ranked["prediction_window"].astype(str) == "day_56"].iloc[0]
    significant_rows = significance[
        significance["p_value_holm"].notna() & (significance["p_value_holm"] < 0.05)
    ]

    lines = [
        "# 多随机种子重复实验与统计显著性检验",
        "",
        "## 1. 实验目的",
        "",
        "为增强论文实验部分的统计可信度，本研究在原有单次学生分组训练-测试划分基础上，补充多随机种子重复实验和配对显著性检验。所有结果均由脚本 `src/repeated_significance.py` 生成，数值来源于 CSV 文件 outputs/tables/repeated_seed_model_results.csv、outputs/tables/repeated_seed_window_results.csv 和 outputs/tables/significance_tests.csv。",
        "",
        "## 2. 实验设置",
        "",
        f"重复实验使用 {len(seeds)} 个随机划分种子：{', '.join(str(seed) for seed in seeds)}。每个种子均采用 StratifiedGroupKFold 的首个划分，分组变量为 `id_student`，以避免同一学生出现在训练集和测试集中。特征构造、禁用特征、窗口截断和 sklearn 训练管道均复用主实验代码，因此预处理仍只在训练集上拟合。",
        "",
        "模型内部随机状态沿用主实验固定配置；因此本节主要评估不同学生分组划分造成的结果方差，而不是模型随机初始化方差。Stacking ensemble "
        + ("已包含在重复模型比较中。" if include_stacking else "未纳入默认重复比较，原因是其训练成本较高；主实验的单次比较仍报告了该模型。"),
        "",
        "## 3. 多模型重复比较",
        "",
        f"第 56 天窗口下，按平均 PR-AUC 排序的最佳模型为 {best_model['model']}，PR-AUC 均值为 {_format_float(best_model['pr_auc_mean'])}，标准差为 {_format_float(best_model['pr_auc_std'])}，F1 均值为 {_format_float(best_model['f1_mean'])}（outputs/tables/repeated_seed_model_summary.csv; outputs/figures/repeated_seed_pr_auc_boxplot.png）。",
        "",
        _markdown_table(
            model_ranked,
            [
                "model",
                "n_repeats",
                "pr_auc_mean",
                "pr_auc_std",
                "f1_mean",
                "f1_std",
                "roc_auc_mean",
                "balanced_accuracy_mean",
            ],
        ),
        "",
        "## 4. 多窗口重复比较",
        "",
        f"XGBoost 在第 56 天窗口的平均 PR-AUC 为 {_format_float(day56['pr_auc_mean'])}，标准差为 {_format_float(day56['pr_auc_std'])}；平均 F1 为 {_format_float(day56['f1_mean'])}，标准差为 {_format_float(day56['f1_std'])}（outputs/tables/repeated_seed_window_summary.csv; outputs/figures/repeated_seed_window_pr_auc.png）。完整课程窗口仍仅作为性能上界，不应解释为早期预警结果。",
        "",
        _markdown_table(
            window_ranked,
            [
                "prediction_window",
                "n_repeats",
                "pr_auc_mean",
                "pr_auc_std",
                "f1_mean",
                "f1_std",
                "roc_auc_mean",
                "balanced_accuracy_mean",
            ],
        ),
        "",
        "## 5. 显著性检验",
        "",
        "本研究使用相同随机种子下的配对 Wilcoxon signed-rank 检验，并在每个比较族和指标内进行 Holm 校正。完整检验结果见 outputs/tables/significance_tests.csv。",
        "",
        f"Holm 校正后 p < 0.05 的比较共有 {len(significant_rows)} 条。显著性检验结果应与均值和标准差共同解释；当模型之间平均差异很小但 p 值不显著时，论文中不应将其表述为明确优劣。",
        "",
        _markdown_table(
            significance.sort_values(["family", "metric", "p_value_holm"]),
            [
                "family",
                "comparison",
                "metric",
                "n_pairs",
                "baseline_mean",
                "comparator_mean",
                "mean_delta",
                "p_value",
                "p_value_holm",
            ],
            max_rows=30,
        ),
        "",
        "## 6. 对论文写作的影响",
        "",
        "新增重复实验使论文可以报告均值、标准差和配对显著性检验，而不只依赖单次划分结果。建议在实验部分新增“重复实验与统计检验”小节，引用 outputs/tables/repeated_seed_model_summary.csv、outputs/tables/repeated_seed_window_summary.csv、outputs/tables/significance_tests.csv、outputs/figures/repeated_seed_pr_auc_boxplot.png 和 outputs/figures/repeated_seed_window_pr_auc.png。",
        "",
        "## 7. 剩余限制",
        "",
        "- 重复实验改变的是训练-测试划分随机种子，模型超参数和模型内部随机状态保持固定。",
        "- 本节未进行嵌套交叉验证或超参数优化，因此不应用于声称模型已经达到最优。",
        "- 统计显著性不等同于教育实践显著性，论文仍需结合阈值分析、校准分析和干预资源约束讨论实际意义。",
        "",
        "## 8. 运行信息",
        "",
        f"脚本运行时间为 {elapsed_seconds / 60:.2f} 分钟。重复实验结果由执行代码生成，没有手工填写指标。",
    ]
    (REPORTS_DIR / "paper_significance_section.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_repeated_significance(
    seeds: list[int],
    include_stacking: bool = False,
) -> None:
    start = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    configure_threading()
    ensure_output_dirs()
    set_global_seed()

    print("[repeats] loading raw data", flush=True)
    data = load_raw_data()

    feature_frames: dict[str, pd.DataFrame] = {}
    for window_name, cutoff_day in PREDICTION_WINDOWS.items():
        print(f"[repeats] building features for {window_name}", flush=True)
        feature_frames[window_name] = build_features_for_window(data, window_name, cutoff_day)

    comparison_frame = feature_frames[MAIN_COMPARISON_WINDOW]
    model_results = run_repeated_model_comparison(
        comparison_frame,
        seeds=seeds,
        include_stacking=include_stacking,
    )
    window_results = run_repeated_time_windows(feature_frames, seeds=seeds, model_name="XGBoost")
    significance = build_significance_tests(
        model_results,
        window_results,
        baseline_model="XGBoost",
    )

    model_summary = summarize_repeated_results(
        model_results, ["prediction_window", "feature_group", "model"]
    )
    window_summary = summarize_repeated_results(
        window_results, ["prediction_window", "feature_group", "model"]
    )

    plot_repeated_model_pr_auc(model_results)
    plot_repeated_window_pr_auc(window_results)
    write_significance_report(
        model_summary,
        window_summary,
        significance,
        seeds=seeds,
        include_stacking=include_stacking,
        elapsed_seconds=time.time() - start,
    )
    print("[repeats] complete", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated-seed OULAD experiments and paired significance tests."
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of consecutive random seeds to run when --seeds is not supplied.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=RANDOM_STATE,
        help="First random seed used when --seeds is not supplied.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        help="Explicit random seeds. Overrides --n-seeds and --seed-start.",
    )
    parser.add_argument(
        "--include-stacking",
        action="store_true",
        help="Include stacking ensemble in repeated model comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_values = (
        args.seeds
        if args.seeds
        else list(range(args.seed_start, args.seed_start + args.n_seeds))
    )
    run_repeated_significance(
        seeds=[int(seed) for seed in seed_values],
        include_stacking=bool(args.include_stacking),
    )
