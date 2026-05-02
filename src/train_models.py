from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import MAX_MODEL_THREADS, RANDOM_STATE
from .evaluate import evaluate_fitted_model, get_prediction_scores


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    is_tree_based: bool


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = [column for column in x.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def positive_class_weight(y: pd.Series) -> float:
    positives = int(np.sum(np.asarray(y) == 1))
    negatives = int(np.sum(np.asarray(y) == 0))
    if positives == 0:
        return 1.0
    return max(1.0, negatives / positives)


def get_model_specs(y_train: pd.Series, include_stacking: bool = True) -> list[ModelSpec]:
    pos_weight = positive_class_weight(y_train)
    specs = [
        ModelSpec(
            "Logistic Regression",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        ModelSpec(
            "Decision Tree",
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=25,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=MAX_MODEL_THREADS,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        ModelSpec(
            "Linear SVM",
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
            False,
        ),
    ]

    try:
        from xgboost import XGBClassifier

        specs.append(
            ModelSpec(
                "XGBoost",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=pos_weight,
                    tree_method="hist",
                    n_jobs=MAX_MODEL_THREADS,
                    random_state=RANDOM_STATE,
                ),
                True,
            )
        )
    except Exception as exc:
        warnings.warn(f"XGBoost unavailable and will be skipped: {exc}")

    try:
        from lightgbm import LGBMClassifier

        specs.append(
            ModelSpec(
                "LightGBM",
                LGBMClassifier(
                    n_estimators=160,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=MAX_MODEL_THREADS,
                    verbose=-1,
                ),
                True,
            )
        )
    except Exception as exc:
        warnings.warn(f"LightGBM unavailable and will be skipped: {exc}")

    try:
        from catboost import CatBoostClassifier

        specs.append(
            ModelSpec(
                "CatBoost",
                CatBoostClassifier(
                    iterations=120,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    auto_class_weights="Balanced",
                    random_seed=RANDOM_STATE,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=MAX_MODEL_THREADS,
                ),
                True,
            )
        )
    except Exception as exc:
        warnings.warn(f"CatBoost unavailable and will be skipped: {exc}")

    if include_stacking:
        base_estimators = [
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=60,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    n_jobs=MAX_MODEL_THREADS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
        specs.append(
            ModelSpec(
                "Stacking Ensemble",
                StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                    cv=3,
                    n_jobs=1,
                    stack_method="auto",
                ),
                True,
            )
        )

    return specs


def find_model_spec(name: str, y_train: pd.Series) -> ModelSpec:
    for spec in get_model_specs(y_train, include_stacking=True):
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model spec: {name}")


def build_pipeline(spec: ModelSpec, x_train: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(x_train)),
            ("model", spec.estimator),
        ]
    )


def fit_evaluate(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, float | int], np.ndarray, np.ndarray]:
    model = build_pipeline(spec, x_train)
    model.fit(x_train, y_train)
    metrics = evaluate_fitted_model(model, x_test, y_test)
    y_pred = model.predict(x_test)
    y_score = get_prediction_scores(model, x_test)
    return model, metrics, y_pred, y_score
