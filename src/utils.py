import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import FIGURES_DIR, MAX_MODEL_THREADS, OUTPUT_DIR, RANDOM_STATE, REPORTS_DIR, TABLES_DIR


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_threading(max_threads: int = MAX_MODEL_THREADS) -> None:
    import os

    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ.setdefault(name, str(max_threads))


def import_optional(module_name: str) -> tuple[bool, Any]:
    try:
        module = __import__(module_name)
    except Exception:
        return False, None
    return True, module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def slugify(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"
