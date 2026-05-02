from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COURSE_COLUMNS, DEMOGRAPHIC_FEATURES, KEY_COLUMNS
from .load_data import create_modeling_base
from .utils import slugify


def _prediction_day_for_frame(base: pd.DataFrame, cutoff_day: int | None) -> pd.Series:
    if cutoff_day is None:
        return base["module_presentation_length"].astype(float)
    return pd.Series(float(cutoff_day), index=base.index)


def _max_inactivity_gap(dates: np.ndarray, cutoff: float) -> float:
    if len(dates) == 0 or np.isnan(cutoff):
        return np.nan
    unique_dates = np.sort(np.unique(dates.astype(float)))
    start = min(0.0, float(unique_dates[0]))
    previous = start - 1.0
    max_gap = 0.0
    for day in unique_dates:
        max_gap = max(max_gap, float(day - previous - 1.0))
        previous = float(day)
    max_gap = max(max_gap, float(cutoff - previous))
    return max(0.0, max_gap)


def _trend_slope(weeks: np.ndarray, clicks: np.ndarray) -> float:
    if len(weeks) < 2:
        return 0.0
    x = weeks.astype(float)
    y = clicks.astype(float)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def build_vle_features(data: dict[str, pd.DataFrame], cutoff_day: int | None) -> pd.DataFrame:
    student_vle = data["studentVle"]
    vle_lookup = data["vle"][COURSE_COLUMNS + ["id_site", "activity_type"]]
    courses = data["courses"][COURSE_COLUMNS + ["module_presentation_length"]]

    df = student_vle.merge(vle_lookup, on=COURSE_COLUMNS + ["id_site"], how="left")
    df["activity_type"] = df["activity_type"].astype("object").fillna("unknown").map(slugify)

    if cutoff_day is None:
        df = df.merge(courses, on=COURSE_COLUMNS, how="left")
        df = df[df["date"] <= df["module_presentation_length"]].copy()
        cutoff_lookup = {
            (row.code_module, row.code_presentation): float(row.module_presentation_length)
            for row in courses.itertuples(index=False)
        }
    else:
        df = df[df["date"] <= cutoff_day].copy()
        cutoff_lookup = None

    if df.empty:
        return pd.DataFrame(columns=KEY_COLUMNS)

    grouped = df.groupby(KEY_COLUMNS, observed=True, sort=False)
    basic = grouped.agg(
        vle_total_clicks=("sum_click", "sum"),
        vle_unique_resources=("id_site", "nunique"),
        vle_activity_types_accessed=("activity_type", "nunique"),
        vle_first_activity_day=("date", "min"),
        vle_last_activity_day=("date", "max"),
    ).reset_index()

    daily = (
        df.groupby(KEY_COLUMNS + ["date"], observed=True, sort=False)["sum_click"]
        .sum()
        .reset_index()
    )
    active_days = (
        daily.groupby(KEY_COLUMNS, observed=True, sort=False)["date"]
        .nunique()
        .rename("vle_active_days")
        .reset_index()
    )

    gap_rows = []
    for key, group in daily.groupby(KEY_COLUMNS, observed=True, sort=False):
        cutoff = float(cutoff_day) if cutoff_day is not None else cutoff_lookup[(key[0], key[1])]
        gap_rows.append((*key, _max_inactivity_gap(group["date"].to_numpy(), cutoff)))
    gaps = pd.DataFrame(gap_rows, columns=KEY_COLUMNS + ["vle_max_inactivity_gap"])

    weekly = df.copy()
    weekly["week_index"] = np.floor(weekly["date"].astype(float) / 7.0).astype("int16")
    weekly = (
        weekly.groupby(KEY_COLUMNS + ["week_index"], observed=True, sort=False)["sum_click"]
        .sum()
        .reset_index()
    )
    slope_rows = []
    for key, group in weekly.groupby(KEY_COLUMNS, observed=True, sort=False):
        slope_rows.append(
            (
                *key,
                _trend_slope(
                    group["week_index"].to_numpy(dtype=float),
                    group["sum_click"].to_numpy(dtype=float),
                ),
            )
        )
    slopes = pd.DataFrame(slope_rows, columns=KEY_COLUMNS + ["vle_weekly_click_trend_slope"])

    activity_clicks = (
        df.groupby(KEY_COLUMNS + ["activity_type"], observed=True, sort=False)["sum_click"]
        .sum()
        .reset_index()
    )
    entropy_rows = []
    for key, group in activity_clicks.groupby(KEY_COLUMNS, observed=True, sort=False):
        values = group["sum_click"].to_numpy(dtype=float)
        total = values.sum()
        entropy = 0.0
        if total > 0:
            probabilities = values / total
            entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
        entropy_rows.append((*key, entropy))
    entropy = pd.DataFrame(entropy_rows, columns=KEY_COLUMNS + ["vle_click_entropy"])

    activity_pivot = activity_clicks.pivot_table(
        index=KEY_COLUMNS,
        columns="activity_type",
        values="sum_click",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )
    activity_pivot.columns = [f"vle_clicks_activity_{column}" for column in activity_pivot.columns]
    activity_pivot = activity_pivot.reset_index()

    features = basic
    for frame in [active_days, gaps, slopes, entropy, activity_pivot]:
        features = features.merge(frame, on=KEY_COLUMNS, how="left")

    features = features.merge(courses, on=COURSE_COLUMNS, how="left")
    if cutoff_day is None:
        features["vle_prediction_day"] = features["module_presentation_length"].astype(float)
    else:
        features["vle_prediction_day"] = float(cutoff_day)
    features = features.drop(columns=["module_presentation_length"])

    features["vle_avg_clicks_per_active_day"] = (
        features["vle_total_clicks"] / features["vle_active_days"].replace(0, np.nan)
    )
    features["vle_days_since_last_activity"] = (
        features["vle_prediction_day"] - features["vle_last_activity_day"]
    )
    first_day = features["vle_first_activity_day"].fillna(0).astype(float)
    features["vle_available_days"] = (
        features["vle_prediction_day"] - np.minimum(0.0, first_day) + 1.0
    )
    features["vle_active_day_ratio"] = (
        features["vle_active_days"] / features["vle_available_days"].replace(0, np.nan)
    )
    features["vle_has_activity"] = 1

    return features


def build_assessment_features(
    data: dict[str, pd.DataFrame], base: pd.DataFrame, cutoff_day: int | None
) -> pd.DataFrame:
    assessments = data["assessments"].copy()
    courses = data["courses"][COURSE_COLUMNS + ["module_presentation_length"]]

    if cutoff_day is None:
        due = assessments.merge(courses, on=COURSE_COLUMNS, how="left")
        due = due[due["date"] <= due["module_presentation_length"]].copy()
    else:
        due = assessments[assessments["date"] <= cutoff_day].copy()

    due_counts = due.groupby(COURSE_COLUMNS, observed=True, sort=False).agg(
        assessment_due_count=("id_assessment", "nunique"),
        assessment_due_weight=("weight", "sum"),
    ).reset_index()

    submitted = data["studentAssessment"].merge(
        assessments.rename(columns={"date": "assessment_date"}),
        on="id_assessment",
        how="left",
    )

    if cutoff_day is None:
        submitted = submitted.merge(courses, on=COURSE_COLUMNS, how="left")
        submitted = submitted[
            (submitted["assessment_date"] <= submitted["module_presentation_length"])
            & (submitted["date_submitted"] <= submitted["module_presentation_length"])
        ].copy()
    else:
        submitted = submitted[
            (submitted["assessment_date"] <= cutoff_day)
            & (submitted["date_submitted"] <= cutoff_day)
        ].copy()

    if submitted.empty:
        assessment_features = pd.DataFrame(columns=KEY_COLUMNS)
    else:
        submitted["assessment_delay"] = (
            submitted["date_submitted"].astype(float) - submitted["assessment_date"].astype(float)
        )
        submitted["assessment_late"] = (submitted["assessment_delay"] > 0).astype("int8")
        submitted["score_weight"] = np.where(submitted["score"].notna(), submitted["weight"], 0.0)
        submitted["weighted_score"] = submitted["score"].fillna(0.0) * submitted["score_weight"]

        grouped = submitted.groupby(KEY_COLUMNS, observed=True, sort=False)
        assessment_features = grouped.agg(
            assessment_submitted_count=("id_assessment", "nunique"),
            assessment_mean_score=("score", "mean"),
            assessment_completed_weight=("weight", "sum"),
            assessment_late_submission_count=("assessment_late", "sum"),
            assessment_average_delay=("assessment_delay", "mean"),
            assessment_maximum_delay=("assessment_delay", "max"),
            assessment_last_submitted_day=("date_submitted", "max"),
            _assessment_weighted_score_sum=("weighted_score", "sum"),
            _assessment_scored_weight_sum=("score_weight", "sum"),
        ).reset_index()
        assessment_features["assessment_weighted_mean_score"] = (
            assessment_features["_assessment_weighted_score_sum"]
            / assessment_features["_assessment_scored_weight_sum"].replace(0, np.nan)
        )
        assessment_features = assessment_features.drop(
            columns=["_assessment_weighted_score_sum", "_assessment_scored_weight_sum"]
        )

    result = base[KEY_COLUMNS].copy()
    result = result.merge(due_counts, on=COURSE_COLUMNS, how="left")
    result = result.merge(assessment_features, on=KEY_COLUMNS, how="left")
    expected_columns = [
        "assessment_due_count",
        "assessment_due_weight",
        "assessment_submitted_count",
        "assessment_mean_score",
        "assessment_completed_weight",
        "assessment_late_submission_count",
        "assessment_average_delay",
        "assessment_maximum_delay",
        "assessment_last_submitted_day",
        "assessment_weighted_mean_score",
    ]
    for column in expected_columns:
        if column not in result.columns:
            result[column] = np.nan
    result["assessment_due_count"] = result["assessment_due_count"].fillna(0)
    result["assessment_due_weight"] = result["assessment_due_weight"].fillna(0)
    result["assessment_submitted_count"] = result["assessment_submitted_count"].fillna(0)
    result["assessment_missing_due_count"] = (
        result["assessment_due_count"] - result["assessment_submitted_count"]
    ).clip(lower=0)
    result["assessment_completed_weight"] = result["assessment_completed_weight"].fillna(0)
    result["assessment_late_submission_count"] = result[
        "assessment_late_submission_count"
    ].fillna(0)
    result["assessment_average_delay"] = result["assessment_average_delay"].fillna(0)
    result["assessment_maximum_delay"] = result["assessment_maximum_delay"].fillna(0)

    return result


def build_registration_features(data: dict[str, pd.DataFrame], base: pd.DataFrame) -> pd.DataFrame:
    registration = data["studentRegistration"][KEY_COLUMNS + ["date_registration"]].copy()
    features = base[KEY_COLUMNS + ["prediction_day"]].merge(
        registration, on=KEY_COLUMNS, how="left"
    )
    registration_known = features["date_registration"].notna() & (
        features["date_registration"] <= features["prediction_day"]
    )
    features["days_before_start_registered"] = np.where(
        registration_known,
        -features["date_registration"].astype(float),
        np.nan,
    )
    features["registration_known_by_prediction_day"] = registration_known.astype("int8")
    return features.drop(columns=["prediction_day", "date_registration"])


def build_features_for_window(
    data: dict[str, pd.DataFrame], window_name: str, cutoff_day: int | None
) -> pd.DataFrame:
    base = create_modeling_base(data)
    base["prediction_window"] = window_name
    base["prediction_day"] = _prediction_day_for_frame(base, cutoff_day)

    registration_features = build_registration_features(data, base)
    vle_features = build_vle_features(data, cutoff_day)
    assessment_features = build_assessment_features(data, base, cutoff_day)

    features = base.merge(registration_features, on=KEY_COLUMNS, how="left")
    features = features.merge(vle_features, on=KEY_COLUMNS, how="left")
    features = features.merge(assessment_features, on=KEY_COLUMNS, how="left")

    vle_zero_columns = [
        column
        for column in features.columns
        if column.startswith("vle_")
        and column
        not in {
            "vle_first_activity_day",
            "vle_last_activity_day",
            "vle_days_since_last_activity",
            "vle_available_days",
            "vle_prediction_day",
        }
    ]
    features[vle_zero_columns] = features[vle_zero_columns].fillna(0)
    features["vle_has_activity"] = features["vle_has_activity"].fillna(0)
    features["vle_prediction_day"] = features["vle_prediction_day"].fillna(features["prediction_day"])
    features["vle_available_days"] = features["vle_available_days"].fillna(
        features["prediction_day"] + 1
    )
    features["vle_days_since_last_activity"] = features["vle_days_since_last_activity"].fillna(
        features["prediction_day"] + 1
    )

    assessment_count_columns = [
        column
        for column in features.columns
        if column.startswith("assessment_")
        and column
        not in {
            "assessment_mean_score",
            "assessment_weighted_mean_score",
            "assessment_last_submitted_day",
        }
    ]
    features[assessment_count_columns] = features[assessment_count_columns].fillna(0)
    return features


def get_feature_group_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    demographic = [column for column in DEMOGRAPHIC_FEATURES if column in frame.columns]
    excluded_vle_columns = {"vle_prediction_day"}
    vle = [
        column
        for column in frame.columns
        if column.startswith("vle_") and column not in excluded_vle_columns
    ]
    assessment = [column for column in frame.columns if column.startswith("assessment_")]

    return {
        "demographic_only": demographic,
        "vle_behavior_only": vle,
        "assessment_only": assessment,
        "demographic_vle": demographic + vle,
        "demographic_vle_assessment": demographic + vle + assessment,
    }
