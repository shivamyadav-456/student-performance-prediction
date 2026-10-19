"""
Comprehensive automated tests for Student Performance Prediction System.
Run:
    python -m unittest tests/test_pipeline.py
"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_raw_dataset,
    prepare_labeled_dataset,
    categorize_final_grade,
    GRADE_BOUNDARIES,
)
from src.preprocessor import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_CLASSES,
    FEATURE_METADATA,
)
from src.predict import (
    load_model,
    predict_student_performance,
    validate_and_format_input,
    SAMPLE_PROFILES,
    DEFAULT_MODEL_PATH,
)


class TestStudentPerformancePipeline(unittest.TestCase):
    """Test suite for data loading, preprocessing, model inference, and integrity."""

    def test_01_dataset_loading_and_shape(self):
        """Verify UCI Math dataset loads correctly with semicolon separator."""
        df = load_raw_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 395, "student-mat.csv should contain 395 records")
        self.assertEqual(len(df.columns), 33, "student-mat.csv should have 33 columns")
        self.assertIn("G3", df.columns, "G3 must be present in raw dataset")
        self.assertEqual(df.isnull().sum().sum(), 0, "Raw dataset should have 0 missing values")

    def test_02_grade_categorization_boundaries(self):
        """Verify strict adherence to defined grade boundary rules."""
        # Low Performance: < 10 (0 - 9)
        self.assertEqual(categorize_final_grade(0), "Low Performance")
        self.assertEqual(categorize_final_grade(5), "Low Performance")
        self.assertEqual(categorize_final_grade(9), "Low Performance")

        # Medium Performance: 10 to 13
        self.assertEqual(categorize_final_grade(10), "Medium Performance")
        self.assertEqual(categorize_final_grade(12), "Medium Performance")
        self.assertEqual(categorize_final_grade(13), "Medium Performance")

        # High Performance: 14 to 20
        self.assertEqual(categorize_final_grade(14), "High Performance")
        self.assertEqual(categorize_final_grade(18), "High Performance")
        self.assertEqual(categorize_final_grade(20), "High Performance")

    def test_03_labeled_dataset_target_distribution(self):
        """Verify labeled dataset creation and non-empty classes."""
        df = prepare_labeled_dataset()
        self.assertIn("performance_category", df.columns)
        counts = df["performance_category"].value_counts()
        for cls in TARGET_CLASSES:
            self.assertIn(cls, counts)
            self.assertGreater(counts[cls], 50, f"Class {cls} should have adequate samples")

    def test_04_target_leakage_prevention(self):
        """Ensure G3 and target label are strictly excluded from input feature columns."""
        self.assertNotIn("G3", FEATURE_COLUMNS, "G3 MUST NOT be an input feature (target leakage)")
        self.assertNotIn("performance_category", FEATURE_COLUMNS)
        self.assertEqual(
            len(FEATURE_COLUMNS),
            len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES),
            "Feature column count must equal sum of categorical and numerical features",
        )
        self.assertEqual(len(FEATURE_COLUMNS), 32)

    def test_05_saved_model_artifact(self):
        """Verify saved model artifact exists, loads, and contains valid metrics."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists(), f"Model file must exist at {DEFAULT_MODEL_PATH}")
        payload = load_model()
        self.assertIn("pipeline", payload)
        self.assertIn("primary_metrics", payload)
        self.assertIn("baseline_metrics", payload)
        self.assertIn("top_feature_importances", payload)

        metrics = payload["primary_metrics"]
        self.assertGreater(metrics["accuracy"], 0.70, "Accuracy should be realistically > 70%")
        self.assertGreater(metrics["f1_macro"], 0.70, "F1-score should be realistically > 70%")
        self.assertEqual(len(metrics["confusion_matrix"]), 3)

        # Baseline comparison check
        baselines = payload["baseline_metrics"]
        self.assertIn("dummy_majority", baselines)
        self.assertIn("logistic_regression", baselines)

    def test_06_predictions_and_probability_coherence(self):
        """Verify predictions on standard sample student profiles."""
        payload = load_model()

        # High Achiever Profile
        high_res = predict_student_performance(payload, SAMPLE_PROFILES["High Achiever Profile"])
        self.assertEqual(high_res["predicted_category"], "High Performance")
        self.assertGreaterEqual(high_res["confidence"], 0.70)

        # At-Risk Profile
        low_res = predict_student_performance(payload, SAMPLE_PROFILES["At-Risk Student Profile"])
        self.assertEqual(low_res["predicted_category"], "Low Performance")
        self.assertGreaterEqual(low_res["confidence"], 0.70)

        # Probabilities sum to 1.0 check
        for name, profile in SAMPLE_PROFILES.items():
            res = predict_student_performance(payload, profile)
            probs = res["class_probabilities"]
            total_prob = sum(probs.values())
            self.assertAlmostEqual(total_prob, 1.0, places=4, msg=f"Probabilities must sum to 1 for {name}")
            self.assertEqual(len(probs), 3)

    def test_07_input_formatting_and_defaults(self):
        """Verify that missing inputs fallback gracefully to sensible defaults."""
        partial_input = {"G1": 15, "G2": 15}
        formatted_df = validate_and_format_input(partial_input)
        self.assertEqual(list(formatted_df.columns), FEATURE_COLUMNS)
        self.assertEqual(formatted_df["G1"].iloc[0], 15)
        self.assertEqual(formatted_df["G2"].iloc[0], 15)
        self.assertEqual(formatted_df["school"].iloc[0], FEATURE_METADATA["school"]["default"])

    def test_08_metadata_json_integrity(self):
        """Verify that model_metadata.json is present and valid."""
        meta_path = DEFAULT_MODEL_PATH.parent / "model_metadata.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("primary_metrics", data)
        self.assertIn("baseline_comparison", data)
        self.assertIn("grade_boundaries", data)

    def test_09_numerical_clamping_and_sanitization(self):
        """Verify that invalid out-of-bound numerical inputs are clamped safely."""
        extreme_input = {
            "age": 99,        # valid range [15, 22] -> clamped to 22
            "G1": 30,         # valid range [0, 20] -> clamped to 20
            "G2": -5,         # valid range [0, 20] -> clamped to 0
            "absences": -10,  # valid range [0, 93] -> clamped to 0
        }
        formatted_df = validate_and_format_input(extreme_input)
        self.assertEqual(formatted_df["age"].iloc[0], 22)
        self.assertEqual(formatted_df["G1"].iloc[0], 20)
        self.assertEqual(formatted_df["G2"].iloc[0], 0)
        self.assertEqual(formatted_df["absences"].iloc[0], 0)

    def test_10_unseen_categorical_graceful_handling(self):
        """Verify that unseen categorical values are handled gracefully by OneHotEncoder."""
        payload = load_model()
        unseen_cat_input = {
            "school": "UNKNOWN_CAMPUS",
            "Mjob": "astronaut",
            "G1": 15,
            "G2": 15,
        }
        res = predict_student_performance(payload, unseen_cat_input)
        self.assertIn(res["predicted_category"], TARGET_CLASSES)
        probs = res["class_probabilities"]
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

    def test_11_probability_bounds_and_argmax_coherence(self):
        """Verify that all predicted probabilities are in [0, 1] and match argmax class."""
        payload = load_model()
        test_inputs = [
            SAMPLE_PROFILES["High Achiever Profile"],
            SAMPLE_PROFILES["Average Student Profile"],
            SAMPLE_PROFILES["At-Risk Student Profile"],
            {"G1": 0, "G2": 0},
            {"G1": 20, "G2": 20},
        ]
        for inp in test_inputs:
            res = predict_student_performance(payload, inp)
            probs = res["class_probabilities"]
            for cls_name, prob in probs.items():
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)
                self.assertTrue(np.isfinite(prob))
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
            # Confidence should equal the max probability and match predicted class
            max_class = max(probs, key=probs.get)
            self.assertEqual(res["predicted_category"], max_class)
            self.assertAlmostEqual(res["confidence"], probs[max_class], places=4)

    def test_12_prediction_determinism(self):
        """Verify that repeated predictions with the same inputs produce identical results."""
        payload = load_model()
        profile = SAMPLE_PROFILES["Average Student Profile"]
        res1 = predict_student_performance(payload, profile)
        res2 = predict_student_performance(payload, profile)
        self.assertEqual(res1["predicted_category"], res2["predicted_category"])
        self.assertEqual(res1["confidence"], res2["confidence"])
        self.assertEqual(res1["class_probabilities"], res2["class_probabilities"])


if __name__ == "__main__":
    unittest.main()
