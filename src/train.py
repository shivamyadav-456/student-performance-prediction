"""
Model training, evaluation, comparison, and persistence for the Student Performance Prediction System.
Dataset: UCI Student Performance Dataset (Mathematics: student-mat.csv)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_loader import prepare_labeled_dataset, GRADE_BOUNDARIES
from src.preprocessor import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CLASSES,
    build_preprocessor,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILENAME = "student_rf_model.joblib"
METADATA_FILENAME = "model_metadata.json"


def train_and_evaluate(
    data_path: Path = None,
    random_state: int = 42,
    test_size: float = 0.2,
    output_dir: Path = None,
) -> Tuple[Dict[str, Any], Path]:
    """
    Executes end-to-end model training, cross-validation, evaluation,
    comparison with baselines, and artifact persistence.
    """
    save_dir = Path(output_dir) if output_dir else MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and label dataset
    df = prepare_labeled_dataset(data_path)

    # Validate target presence and absence of target leakage
    if "G3" not in df.columns or "performance_category" not in df.columns:
        raise ValueError("Missing G3 or performance_category in prepared dataset")

    # Strictly drop G3 (the raw final grade) to prevent target leakage
    X = df[FEATURE_COLUMNS].copy()
    y = df["performance_category"].copy()

    # 2. Stratified train/test split (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 3. Train Baseline Models
    # Baseline A: Dummy Majority Classifier
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_pred = dummy_model.predict(X_test)
    dummy_acc = float(accuracy_score(y_test, dummy_pred))
    dummy_f1 = float(f1_score(y_test, dummy_pred, average="macro", zero_division=0))

    # Baseline B: Logistic Regression Pipeline
    lr_preprocessor = build_preprocessor()
    lr_pipeline = Pipeline(
        [
            ("preprocessor", lr_preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    # 5-Fold Cross Validation on Training Split
    lr_cv_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    lr_cv_f1 = cross_val_score(lr_pipeline, X_train, y_train, cv=skf, scoring="f1_macro")

    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    lr_acc = float(accuracy_score(y_test, lr_pred))
    lr_f1 = float(f1_score(y_test, lr_pred, average="macro", zero_division=0))

    # 4. Train Primary Model: Random Forest Pipeline
    rf_preprocessor = build_preprocessor()
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    )

    rf_pipeline = Pipeline(
        [
            ("preprocessor", rf_preprocessor),
            ("classifier", rf_classifier),
        ]
    )

    # 5-Fold Cross Validation on Training Split
    rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=skf, scoring="accuracy")
    rf_cv_f1 = cross_val_score(rf_pipeline, X_train, y_train, cv=skf, scoring="f1_macro")

    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)

    # 5. Calculate Actual Evaluation Metrics on Held-out Test Set
    test_acc = float(accuracy_score(y_test, rf_pred))
    test_prec_macro = float(precision_score(y_test, rf_pred, average="macro", zero_division=0))
    test_rec_macro = float(recall_score(y_test, rf_pred, average="macro", zero_division=0))
    test_f1_macro = float(f1_score(y_test, rf_pred, average="macro", zero_division=0))

    test_prec_weighted = float(precision_score(y_test, rf_pred, average="weighted", zero_division=0))
    test_rec_weighted = float(recall_score(y_test, rf_pred, average="weighted", zero_division=0))
    test_f1_weighted = float(f1_score(y_test, rf_pred, average="weighted", zero_division=0))

    # Confusion Matrix ordered by TARGET_CLASSES
    cm = confusion_matrix(y_test, rf_pred, labels=TARGET_CLASSES)
    cr_dict = classification_report(y_test, rf_pred, labels=TARGET_CLASSES, output_dict=True, zero_division=0)
    cr_text = classification_report(y_test, rf_pred, labels=TARGET_CLASSES, zero_division=0)

    # 6. Extract Feature Importances
    fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
    fitted_rf = rf_pipeline.named_steps["classifier"]

    try:
        encoded_feature_names = fitted_preprocessor.get_feature_names_out()
    except Exception:
        encoded_feature_names = [f"feature_{i}" for i in range(len(fitted_rf.feature_importances_))]

    feature_importances = []
    for name, imp in zip(encoded_feature_names, fitted_rf.feature_importances_):
        clean_name = name.replace("num__", "").replace("cat__", "")
        feature_importances.append({"feature": clean_name, "raw_name": str(name), "importance": float(imp)})

    feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    # 7. Construct Complete Model Payload
    model_payload = {
        "pipeline": rf_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_classes": TARGET_CLASSES,
        "grade_boundaries": GRADE_BOUNDARIES,
        "random_state": random_state,
        "training_timestamp": datetime.now().isoformat(),
        "dataset_name": "UCI Student Performance (Mathematics: student-mat.csv)",
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "primary_metrics": {
            "accuracy": test_acc,
            "precision_macro": test_prec_macro,
            "recall_macro": test_rec_macro,
            "f1_macro": test_f1_macro,
            "precision_weighted": test_prec_weighted,
            "recall_weighted": test_rec_weighted,
            "f1_weighted": test_f1_weighted,
            "confusion_matrix": cm.tolist(),
            "classification_report": cr_dict,
            "classification_report_text": cr_text,
            "cv_accuracy_mean": float(rf_cv_scores.mean()),
            "cv_accuracy_std": float(rf_cv_scores.std()),
            "cv_f1_mean": float(rf_cv_f1.mean()),
        },
        "baseline_metrics": {
            "dummy_majority": {
                "name": "Majority Class Baseline (DummyClassifier)",
                "accuracy": dummy_acc,
                "f1_macro": dummy_f1,
                "cv_accuracy": float(dummy_acc),
            },
            "logistic_regression": {
                "name": "Logistic Regression Baseline",
                "accuracy": lr_acc,
                "f1_macro": lr_f1,
                "cv_accuracy_mean": float(lr_cv_scores.mean()),
                "cv_accuracy_std": float(lr_cv_scores.std()),
                "cv_f1_mean": float(lr_cv_f1.mean()),
            },
            "random_forest": {
                "name": "Random Forest Classifier (Primary)",
                "accuracy": test_acc,
                "f1_macro": test_f1_macro,
                "cv_accuracy_mean": float(rf_cv_scores.mean()),
                "cv_accuracy_std": float(rf_cv_scores.std()),
                "cv_f1_mean": float(rf_cv_f1.mean()),
            },
        },
        "top_feature_importances": feature_importances[:15],
        "all_feature_importances": feature_importances,
    }

    # 8. Save Model Artifact and Metadata
    model_path = save_dir / MODEL_FILENAME
    metadata_path = save_dir / METADATA_FILENAME

    joblib.dump(model_payload, model_path)

    json_metadata = {
        "dataset_name": model_payload["dataset_name"],
        "training_timestamp": model_payload["training_timestamp"],
        "random_state": model_payload["random_state"],
        "train_size": model_payload["train_size"],
        "test_size": model_payload["test_size"],
        "grade_boundaries": model_payload["grade_boundaries"],
        "target_classes": model_payload["target_classes"],
        "feature_count": len(FEATURE_COLUMNS),
        "primary_metrics": {
            "test_accuracy": test_acc,
            "cv_accuracy_mean": float(rf_cv_scores.mean()),
            "cv_accuracy_std": float(rf_cv_scores.std()),
            "precision_macro": test_prec_macro,
            "recall_macro": test_rec_macro,
            "f1_macro": test_f1_macro,
            "precision_weighted": test_prec_weighted,
            "recall_weighted": test_rec_weighted,
            "f1_weighted": test_f1_weighted,
        },
        "baseline_comparison": {
            "Dummy Majority Baseline": {"test_accuracy": dummy_acc, "f1_macro": dummy_f1},
            "Logistic Regression Baseline": {
                "test_accuracy": lr_acc,
                "cv_accuracy_mean": float(lr_cv_scores.mean()),
                "f1_macro": lr_f1,
            },
            "Random Forest (Primary)": {
                "test_accuracy": test_acc,
                "cv_accuracy_mean": float(rf_cv_scores.mean()),
                "f1_macro": test_f1_macro,
            },
        },
        "top_5_features": [f["feature"] for f in feature_importances[:5]],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(json_metadata, f, indent=4)

    return model_payload, model_path
