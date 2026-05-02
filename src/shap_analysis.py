from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import slugify


def _as_2d_shap_values(values) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, 1] if array.shape[2] > 1 else array[:, :, 0]
    return array


def _clean_feature_name(name: str) -> str:
    name = name.replace("numeric__", "").replace("categorical__", "")
    return name


def run_shap_analysis(
    model,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
    identifiers: pd.DataFrame,
    output_table_path: Path,
    output_summary_path: Path,
    figures_dir: Path,
    local_output_path: Path,
    max_rows: int = 1200,
) -> pd.DataFrame:
    import shap

    if len(x_eval) > max_rows:
        x_sample = x_eval.sample(max_rows, random_state=42)
    else:
        x_sample = x_eval.copy()

    sample_positions = x_eval.index.get_indexer(x_sample.index)
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(x_sample)
    feature_names = [_clean_feature_name(name) for name in preprocessor.get_feature_names_out()]
    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=x_sample.index)

    explainer = shap.TreeExplainer(estimator)
    raw_values = explainer.shap_values(transformed)
    shap_values = _as_2d_shap_values(raw_values)

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_features = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_features.to_csv(output_table_path, index=False)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, transformed_frame, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    for feature in top_features["feature"].head(5):
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(feature, shap_values, transformed_frame, show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / f"shap_dependence_{slugify(feature)}.png", dpi=200)
        plt.close()

    y_pred = model.predict(x_eval)
    correct_high_risk = np.where((np.asarray(y_eval) == 1) & (np.asarray(y_pred) == 1))[0][:3]
    local_rows = []
    if len(correct_high_risk) > 0:
        eval_subset = x_eval.iloc[correct_high_risk]
        transformed_subset = preprocessor.transform(eval_subset)
        local_values = _as_2d_shap_values(explainer.shap_values(transformed_subset))
        for local_position, original_position in enumerate(correct_high_risk):
            row_id = identifiers.iloc[original_position].to_dict()
            contributions = pd.Series(local_values[local_position], index=feature_names)
            for feature, value in contributions.abs().sort_values(ascending=False).head(10).items():
                local_rows.append(
                    {
                        **row_id,
                        "feature": feature,
                        "shap_value": float(contributions[feature]),
                        "abs_shap_value": float(value),
                    }
                )
    pd.DataFrame(local_rows).to_csv(local_output_path, index=False)

    return top_features

