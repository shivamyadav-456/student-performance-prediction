"""
Data loading and preprocessing utilities for the Student Performance Prediction System.
Dataset Source: UCI Machine Learning Repository (Student Performance Dataset)
URL: https://archive.ics.uci.edu/dataset/320/student+performance
"""

import io
import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MATH_DATASET_NAME = "student-mat.csv"
POR_DATASET_NAME = "student-por.csv"
UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

# Target performance category definitions based on final grade G3 (0 - 20 scale)
GRADE_BOUNDARIES = {
    "Low Performance": (0, 9),      # Insufficient / Failing (< 10)
    "Medium Performance": (10, 13), # Satisfactory / Sufficient (10 - 13)
    "High Performance": (14, 20)    # Good to Excellent (14 - 20)
}


def download_dataset_if_needed(target_dir: Optional[Path] = None) -> Path:
    """
    Ensures student-mat.csv is present in target_dir. If missing, attempts
    to download and extract it from the official UCI repository.
    """
    dest_dir = Path(target_dir) if target_dir else DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    math_csv = dest_dir / MATH_DATASET_NAME

    if math_csv.exists() and math_csv.stat().st_size > 0:
        return math_csv

    print(f"Dataset not found locally at {math_csv}. Attempting automatic download from UCI...")
    try:
        response = requests.get(UCI_DOWNLOAD_URL, timeout=25)
        response.raise_for_status()

        # The outer zip archive contains 'student.zip'
        with zipfile.ZipFile(io.BytesIO(response.content)) as outer_zip:
            if "student.zip" in outer_zip.namelist():
                inner_zip_bytes = outer_zip.read("student.zip")
                with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zip:
                    inner_zip.extractall(dest_dir)
            else:
                outer_zip.extractall(dest_dir)

        if not math_csv.exists():
            raise FileNotFoundError(f"Extracted archive did not contain {MATH_DATASET_NAME}")

        print(f"Dataset successfully downloaded and extracted to {dest_dir}")
        return math_csv

    except Exception as e:
        error_msg = (
            f"Failed to automatically download dataset from {UCI_DOWNLOAD_URL}: {e}\n"
            f"MANUAL FALLBACK INSTRUCTIONS:\n"
            f"1. Download 'student+performance.zip' from https://archive.ics.uci.edu/dataset/320/student+performance\n"
            f"2. Extract 'student-mat.csv' (and optionally 'student-por.csv') into:\n"
            f"   {dest_dir.resolve()}\n"
        )
        raise FileNotFoundError(error_msg) from e


def load_raw_dataset(filepath: Optional[Path] = None, sep: str = ";") -> pd.DataFrame:
    """
    Loads the student dataset from CSV using the semicolon delimiter.
    """
    if filepath is None:
        filepath = DATA_DIR / MATH_DATASET_NAME

    path = Path(filepath)
    if not path.exists():
        download_dataset_if_needed(path.parent)

    df = pd.read_csv(path, sep=sep)
    return df


def categorize_final_grade(grade: float) -> str:
    """
    Categorizes the Portuguese 0-20 scale final grade (G3):
    - Low Performance: 0 to 9 (< 10, failing mark in Portuguese secondary education)
    - Medium Performance: 10 to 13 (Satisfactory / passing)
    - High Performance: 14 to 20 (Good to Excellent)
    """
    if grade < 10:
        return "Low Performance"
    elif grade <= 13:
        return "Medium Performance"
    else:
        return "High Performance"


def prepare_labeled_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw data and adds the derived target column 'performance_category'.
    """
    df = load_raw_dataset(filepath)
    if "G3" not in df.columns:
        raise ValueError("Dataset does not contain expected final grade column 'G3'")
    df = df.copy()
    df["performance_category"] = df["G3"].apply(categorize_final_grade)
    return df


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculates summary data health and distribution statistics.
    """
    category_counts = df["performance_category"].value_counts().to_dict() if "performance_category" in df.columns else {}
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "grade_mean_g1": float(df["G1"].mean()) if "G1" in df.columns else None,
        "grade_mean_g2": float(df["G2"].mean()) if "G2" in df.columns else None,
        "grade_mean_g3": float(df["G3"].mean()) if "G3" in df.columns else None,
        "category_distribution": category_counts,
        "grade_boundaries": GRADE_BOUNDARIES
    }
