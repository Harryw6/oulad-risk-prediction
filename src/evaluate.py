from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def get_prediction_scores(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict(x)


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    if len(np.unique(y_true)) < 2:
        roc_auc = math.nan
        pr_auc = math.nan
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_fitted_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | int]:
    y_pred = model.predict(x_test)
    y_score = get_prediction_scores(model, x_test)
    return compute_metrics(y_test, y_pred, y_score)


def plot_label_distribution(base: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = base["at_risk"].map({0: "Not at risk", 1: "At risk"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=["#4C78A8", "#E45756"])
    ax.set_xlabel("")
    ax.set_ylabel("Student-course instances")
    ax.set_title("Binary Label Distribution")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_window_performance(results: pd.DataFrame, output_path: Path) -> None:
    order = ["day_7", "day_14", "day_28", "day_56", "full"]
    metric_names = ["f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    plot_frame = results[results["prediction_window"].isin(order)].copy()
    plot_frame["prediction_window"] = pd.Categorical(
        plot_frame["prediction_window"], categories=order, ordered=True
    )
    plot_frame = plot_frame.sort_values("prediction_window")

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metric_names:
        ax.plot(
            plot_frame["prediction_window"].astype(str),
            plot_frame[metric],
            marker="o",
            label=metric,
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Prediction window")
    ax.set_ylabel("Score")
    ax.set_title("Early-Warning Performance by Window")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_pr_curves(curve_rows: list[dict[str, object]], roc_path: Path, pr_path: Path) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 5))

    for row in curve_rows:
        y_true = row["y_true"]
        y_score = row["y_score"]
        label = row["label"]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax_roc.plot(fpr, tpr, label=label)
        ax_pr.plot(recall, precision, label=label)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.25)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.25)

    fig_roc.tight_layout()
    fig_pr.tight_layout()
    fig_roc.savefig(roc_path, dpi=200)
    fig_pr.savefig(pr_path, dpi=200)
    plt.close(fig_roc)
    plt.close(fig_pr)


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred not risk", "Pred risk"],
        yticklabels=["True not risk", "True risk"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

