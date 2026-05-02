from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import COURSE_COLUMNS, KEY_COLUMNS, RAW_DATA_DIR, REQUIRED_RAW_FILES


def validate_raw_data(raw_dir: Path = RAW_DATA_DIR) -> None:
    missing = [name for name in REQUIRED_RAW_FILES if not (raw_dir / name).exists()]
    if missing:
        missing_list = "\n".join(f"- {name}" for name in missing)
        raise FileNotFoundError(
            "Missing OULAD raw CSV files.\n"
            "Download the official Figshare dataset from:\n"
            "https://figshare.com/articles/dataset/OULAD_Open_University_Learning_Analytics_Dataset/5081998\n"
            f"Place these files in {raw_dir}:\n{missing_list}"
        )


def _read_csv(raw_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(raw_dir / name, na_values=["", "?", "NA"], keep_default_na=True, **kwargs)


def load_raw_data(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    validate_raw_data(raw_dir)

    student_info = _read_csv(
        raw_dir,
        "studentInfo.csv",
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "id_student": "int32",
            "gender": "category",
            "region": "category",
            "highest_education": "category",
            "imd_band": "category",
            "age_band": "category",
            "num_of_prev_attempts": "int16",
            "studied_credits": "int16",
            "disability": "category",
            "final_result": "category",
        },
    )

    student_registration = _read_csv(
        raw_dir,
        "studentRegistration.csv",
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "id_student": "int32",
            "date_registration": "float32",
            "date_unregistration": "float32",
        },
    )

    student_assessment = _read_csv(
        raw_dir,
        "studentAssessment.csv",
        dtype={
            "id_assessment": "int32",
            "id_student": "int32",
            "date_submitted": "int16",
            "is_banked": "int8",
            "score": "float32",
        },
    )

    assessments = _read_csv(
        raw_dir,
        "assessments.csv",
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "id_assessment": "int32",
            "assessment_type": "category",
            "date": "float32",
            "weight": "float32",
        },
    )

    student_vle = _read_csv(
        raw_dir,
        "studentVle.csv",
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "id_student": "int32",
            "id_site": "int32",
            "date": "int16",
            "sum_click": "int32",
        },
    )

    vle = _read_csv(
        raw_dir,
        "vle.csv",
        dtype={
            "id_site": "int32",
            "code_module": "category",
            "code_presentation": "category",
            "activity_type": "category",
            "week_from": "float32",
            "week_to": "float32",
        },
    )

    courses = _read_csv(
        raw_dir,
        "courses.csv",
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "module_presentation_length": "int16",
        },
    )

    return {
        "studentInfo": student_info,
        "studentRegistration": student_registration,
        "studentAssessment": student_assessment,
        "assessments": assessments,
        "studentVle": student_vle,
        "vle": vle,
        "courses": courses,
    }


def create_modeling_base(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = data["studentInfo"].copy()
    base = base.merge(data["courses"], on=COURSE_COLUMNS, how="left")

    base["at_risk"] = base["final_result"].isin(["Fail", "Withdrawn"]).astype("int8")
    base["course_presentation"] = (
        base["code_module"].astype(str) + "_" + base["code_presentation"].astype(str)
    )
    return base


def create_dataset_summary(data: dict[str, pd.DataFrame], base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for name, frame in data.items():
        rows.append({"section": "raw_file", "metric": f"{name}_rows", "value": len(frame)})

    rows.extend(
        [
            {"section": "dataset", "metric": "student_course_instances", "value": len(base)},
            {
                "section": "dataset",
                "metric": "unique_students",
                "value": base["id_student"].nunique(),
            },
            {
                "section": "dataset",
                "metric": "course_presentations",
                "value": base["course_presentation"].nunique(),
            },
            {
                "section": "dataset",
                "metric": "at_risk_rate",
                "value": round(float(base["at_risk"].mean()), 6),
            },
        ]
    )

    for label, count in base["final_result"].value_counts(dropna=False).items():
        rows.append({"section": "final_result", "metric": str(label), "value": int(count)})

    for label, count in base["at_risk"].value_counts(dropna=False).sort_index().items():
        rows.append({"section": "binary_label", "metric": f"at_risk_{label}", "value": int(count)})

    course_counts = base.groupby("course_presentation", observed=True).size().sort_index()
    for group, count in course_counts.items():
        rows.append({"section": "course_presentation", "metric": str(group), "value": int(count)})

    return pd.DataFrame(rows)
