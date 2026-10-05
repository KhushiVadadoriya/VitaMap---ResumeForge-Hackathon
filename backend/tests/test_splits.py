"""
Tests for stratified splits, data integrity, and leakage prevention in VitaMap ResumeForge.
"""

import json
from pathlib import Path
import unittest
import pandas as pd


class TestSplitsAndLeakage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent
        cls.splits_dir = cls.base_dir / "data" / "processed" / "splits"
        cls.train_df = pd.read_csv(cls.splits_dir / "train.csv")
        cls.val_df = pd.read_csv(cls.splits_dir / "validation.csv")
        cls.test_df = pd.read_csv(cls.splits_dir / "test.csv")

    def test_split_counts_and_proportions(self):
        total = len(self.train_df) + len(self.val_df) + len(self.test_df)
        self.assertEqual(total, 2481)
        self.assertEqual(len(self.train_df), 1736)
        self.assertEqual(len(self.val_df), 372)
        self.assertEqual(len(self.test_df), 373)

        # Train ~70%, Val ~15%, Test ~15%
        self.assertAlmostEqual(len(self.train_df) / total, 0.70, places=2)
        self.assertAlmostEqual(len(self.val_df) / total, 0.15, places=2)
        self.assertAlmostEqual(len(self.test_df) / total, 0.15, places=2)

    def test_no_id_leakage(self):
        train_ids = set(self.train_df["ID"])
        val_ids = set(self.val_df["ID"])
        test_ids = set(self.test_df["ID"])

        self.assertEqual(len(train_ids.intersection(val_ids)), 0, "Train-Val ID leakage!")
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Train-Test ID leakage!")
        self.assertEqual(len(val_ids.intersection(test_ids)), 0, "Val-Test ID leakage!")

    def test_no_text_leakage(self):
        train_texts = set(self.train_df["text"])
        val_texts = set(self.val_df["text"])
        test_texts = set(self.test_df["text"])

        self.assertEqual(len(train_texts.intersection(val_texts)), 0, "Train-Val text leakage!")
        self.assertEqual(len(train_texts.intersection(test_texts)), 0, "Train-Test text leakage!")
        self.assertEqual(len(val_texts.intersection(test_texts)), 0, "Val-Test text leakage!")

    def test_all_24_classes_represented(self):
        self.assertEqual(self.train_df["Category"].nunique(), 24)
        self.assertEqual(self.val_df["Category"].nunique(), 24)
        self.assertEqual(self.test_df["Category"].nunique(), 24)

    def test_no_null_or_empty_text(self):
        for name, df in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
            self.assertEqual(df["text"].isna().sum(), 0, f"{name} contains null text!")
            self.assertEqual((df["text"].str.strip() == "").sum(), 0, f"{name} contains empty text!")
            self.assertEqual(df["Category"].isna().sum(), 0, f"{name} contains null Category!")

    def test_removed_duplicate_ids_not_present(self):
        all_split_ids = set(self.train_df["ID"]).union(self.val_df["ID"]).union(self.test_df["ID"])
        # Duplicates that should have been excluded
        self.assertNotIn(28398216, all_split_ids)
        self.assertNotIn(37473139, all_split_ids)
        # Empty resume that should have been excluded
        self.assertNotIn(12632728, all_split_ids)

    def test_metadata_consistency(self):
        meta_path = self.splits_dir / "split_metadata.json"
        self.assertTrue(meta_path.exists())
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["final_clean_records"], 2481)
        self.assertEqual(meta["final_class_count"], 24)
        self.assertEqual(meta["leakage_verification"]["id_overlap"], 0)
        self.assertEqual(meta["leakage_verification"]["text_overlap"], 0)
        self.assertFalse(meta["leakage_verification"]["learned_transformations_fitted"])


if __name__ == "__main__":
    unittest.main()
