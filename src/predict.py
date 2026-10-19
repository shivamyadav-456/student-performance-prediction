"""
Prediction logic, input validation, and result interpretation for the Student Performance Prediction System.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
from src.preprocessor import FEATURE_COLUMNS, TARGET_CLASSES, FEATURE_METADATA

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "student_rf_model.joblib"

# Pre-configured student profiles for instant demo testing and validation
SAMPLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "High Achiever Profile": {
        "school": "GP",
        "sex": "F",
        "age": 16,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "teacher",
        "Fjob": "services",
        "reason": "reputation",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "yes",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 5,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 2,
        "G1": 16,
        "G2": 17,
    },
    "Average Student Profile": {
        "school": "GP",
        "sex": "M",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 4,
        "absences": 4,
        "G1": 11,
        "G2": 12,
    },
    "At-Risk Student Profile": {
        "school": "MS",
        "sex": "M",
        "age": 18,
        "address": "R",
        "famsize": "GT3",
        "Pstatus": "A",
        "Medu": 1,
        "Fedu": 1,
        "Mjob": "at_home",
        "Fjob": "other",
        "reason": "home",
        "guardian": "other",
        "traveltime": 3,
        "studytime": 1,
        "failures": 2,
        "schoolsup": "no",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "nursery": "no",
        "higher": "no",
        "internet": "no",
        "romantic": "yes",
        "famrel": 2,
        "freetime": 4,
        "goout": 4,
        "Dalc": 3,
        "Walc": 4,
        "health": 2,
        "absences": 16,
        "G1": 6,
        "G2": 6,
    },
}


def load_model(model_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads the trained model payload from disk.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {path.resolve()}.\n"
            f"Please run 'python train_model.py' first to train and persist the model."
        )
    model_payload = joblib.load(path)
    return model_payload


def validate_and_format_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validates input features against FEATURE_METADATA schema, fills missing keys
    with defaults if necessary, and formats strictly into an ordered DataFrame.
    """
    formatted: Dict[str, Any] = {}
    missing_cols = []

    for col in FEATURE_COLUMNS:
        if col not in input_data:
            meta = FEATURE_METADATA.get(col, {})
            if "default" in meta:
                formatted[col] = meta["default"]
            else:
                missing_cols.append(col)
        else:
            formatted[col] = input_data[col]

    if missing_cols:
        raise ValueError(f"Missing required input features without defaults: {missing_cols}")

    # Build DataFrame strictly matching FEATURE_COLUMNS order
    df = pd.DataFrame([formatted])[FEATURE_COLUMNS]

    # Convert numeric fields to int/float
    for col in FEATURE_COLUMNS:
        meta = FEATURE_METADATA.get(col, {})
        if meta.get("type") in ("numerical", "ordinal"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                df[col] = meta.get("default", 0)
            # Enforce valid domain bounds if min/max are defined in metadata
            min_val = meta.get("min")
            max_val = meta.get("max")
            if min_val is not None:
                df[col] = df[col].clip(lower=min_val)
            if max_val is not None:
                df[col] = df[col].clip(upper=max_val)
        else:
            # Ensure categorical inputs are sanitized strings
            df[col] = df[col].astype(str)

    return df


def generate_interpretation(category: str, confidence: float, input_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates educational, objective interpretations of model outputs.
    """
    g1 = float(input_df["G1"].iloc[0])
    g2 = float(input_df["G2"].iloc[0])
    absences = int(input_df["absences"].iloc[0])
    failures = int(input_df["failures"].iloc[0])
    studytime = int(input_df["studytime"].iloc[0])

    if category == "High Performance":
        badge_color = "success"
        headline = "Projected High Academic Performance (14–20 Final Grade Range)"
        summary = (
            f"The model estimates with {confidence:.1%} confidence that the student is poised for "
            f"High Performance. Strong earlier grades (G1: {g1:.0f}, G2: {g2:.0f}), low course failures ({failures}), "
            f"and consistent academic habits provide solid positive support."
        )
        recommendations = [
            "Encourage participation in advanced or honors enrichment programs.",
            "Maintain current steady study schedule and attendance habits.",
            "Explore university pathways and scholarship opportunities.",
        ]
    elif category == "Medium Performance":
        badge_color = "info"
        headline = "Projected Medium Academic Performance (10–13 Final Grade Range)"
        summary = (
            f"The model estimates with {confidence:.1%} confidence that the student is performing "
            f"at a Satisfactory / Medium level. With current intermediate grades (G1: {g1:.0f}, G2: {g2:.0f}) "
            f"and attendance, consistent focused effort could push performance into the higher tier."
        )
        recommendations = [
            "Establish structured weekly study blocks (aiming for >5 hours/week).",
            "Focus revision on specific problem areas identified during G1 and G2 exams.",
            "Utilize peer study groups or teacher office hours for difficult concepts.",
        ]
    else:
        badge_color = "warning"
        headline = "Projected Low Academic Performance (0–9 Final Grade Range - At Risk)"
        summary = (
            f"The model estimates with {confidence:.1%} confidence that the student is currently "
            f"in an at-risk academic category (<10/20 passing threshold). Contributing risk indicators "
            f"include earlier grade performance (G1: {g1:.0f}, G2: {g2:.0f}), past class failures ({failures}), "
            f"and recorded absences ({absences} days)."
        )
        recommendations = [
            "Recommend structured remedial academic tutoring and concepts review.",
            "Address attendance patterns and engage parents/guardians early.",
            "Implement bi-weekly progress check-ins with academic counselors.",
        ]

    return {
        "badge_color": badge_color,
        "headline": headline,
        "summary": summary,
        "recommendations": recommendations,
    }


def predict_student_performance(
    model_payload: Dict[str, Any],
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes inference using the fitted Scikit-learn pipeline, calculates class probabilities,
    and returns a structured prediction response.
    """
    pipeline = model_payload["pipeline"]
    target_classes = model_payload.get("target_classes", TARGET_CLASSES)

    # 1. Format and validate input features
    input_df = validate_and_format_input(input_data)

    # 2. Run prediction and probability estimation
    predicted_class = str(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]

    # Map probabilities to classes
    prob_dict = {cls_name: float(prob) for cls_name, prob in zip(pipeline.classes_, probabilities)}

    # Ensure all target classes are represented in ordered dictionary
    ordered_probs = {cls_name: prob_dict.get(cls_name, 0.0) for cls_name in target_classes}
    confidence = ordered_probs.get(predicted_class, float(max(probabilities)))

    # 3. Generate educational interpretation
    interpretation = generate_interpretation(predicted_class, confidence, input_df)

    return {
        "predicted_category": predicted_class,
        "confidence": confidence,
        "class_probabilities": ordered_probs,
        "interpretation": interpretation,
        "input_features": input_df.to_dict(orient="records")[0],
    }
