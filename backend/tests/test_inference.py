"""
Lightweight inference tests for the final VitaMap pipeline artifact.
"""

import json
import unittest
from pathlib import Path

from backend.src.predict import (
    FINAL_ARTIFACT_PATH,
    load_pipeline,
    predict_resume,
)


class TestFinalInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline()

    def test_artifact_exists(self):
        self.assertTrue(FINAL_ARTIFACT_PATH.exists(), "Final pipeline artifact missing")

    def test_pipeline_reloads(self):
        reloaded = load_pipeline(str(FINAL_ARTIFACT_PATH))
        self.assertIsNotNone(reloaded)
        self.assertIn("preprocess", reloaded.named_steps)
        self.assertIn("tfidf", reloaded.named_steps)
        self.assertIn("classifier", reloaded.named_steps)

    def test_raw_text_prediction_valid_category(self):
        raw_resume = (
            "Experienced software engineer skilled in Python, SQL, AWS, "
            "and machine learning. Built REST APIs and deployed models."
        )
        result = predict_resume(raw_resume)
        metadata_path = FINAL_ARTIFACT_PATH.parent / "model_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            known_labels = json.load(f)["class_labels"]
        self.assertIn(result["category"], known_labels)
        self.assertIsInstance(result["decision_score"], float)
        self.assertTrue(len(result["top_predictions"]) >= 1)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            predict_resume("   ")
        with self.assertRaises(ValueError):
            predict_resume("")


if __name__ == "__main__":
    unittest.main()
