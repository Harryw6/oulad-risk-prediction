import os
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_MODEL_THREADS = min(64, os.cpu_count() or 1)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "oulad" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

REQUIRED_RAW_FILES = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "assessments.csv",
    "studentVle.csv",
    "vle.csv",
    "courses.csv",
]

PREDICTION_WINDOWS = {
    "day_7": 7,
    "day_14": 14,
    "day_28": 28,
    "day_56": 56,
    "full": None,
}

MAIN_COMPARISON_WINDOW = "day_56"

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
COURSE_COLUMNS = ["code_module", "code_presentation"]

DEMOGRAPHIC_FEATURES = [
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "days_before_start_registered",
]

TARGET_COLUMN = "at_risk"
GROUP_COLUMN = "course_presentation"
