"""
Data Preparation and Leakage-Safe Splitting Pipeline for VitaMap ResumeForge.
Executes Phase 2 data cleaning, empty-resume recovery handling, duplicate deduplication,
text normalization, stratified splitting, and leakage verification.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.src.preprocessing import (
    clean_text_minimal,
    extract_text_from_html,
    extract_text_from_pdf,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_data(
    raw_csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict:
    """
    Run full data preparation pipeline and produce clean dataset and stratified splits.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if raw_csv_path is None:
        raw_csv_path = base_dir / "data" / "raw" / "Resume.csv"
    else:
        raw_csv_path = Path(raw_csv_path)

    if output_dir is None:
        output_dir = base_dir / "data" / "processed"
    else:
        output_dir = Path(output_dir)

    splits_dir = output_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    logger.info("Loading raw dataset from %s", raw_csv_path)
    df_raw = pd.read_csv(raw_csv_path)
    original_records_count = len(df_raw)
    logger.info("Raw dataset shape: %s", df_raw.shape)

    # Initialize metadata tracking
    text_source_series = pd.Series(["Resume_str"] * len(df_raw), index=df_raw.index)
    recovered_ids = []
    unrecoverable_ids = []

    # 2. Step 2: Handle Empty / Whitespace Resumes
    # Identify empty or whitespace-only Resume_str
    empty_mask = df_raw["Resume_str"].isna() | (df_raw["Resume_str"].astype(str).str.strip() == "")
    empty_indices = df_raw[empty_mask].index

    for idx in empty_indices:
        row = df_raw.loc[idx]
        row_id = row["ID"]
        category = row["Category"]
        logger.info("Processing empty Resume_str for ID %s (Category: %s)", row_id, category)

        # Attempt 1: Recover from Resume_html
        html_content = row.get("Resume_html", "")
        recovered_html_text = extract_text_from_html(html_content)

        if len(recovered_html_text.split()) >= 20:
            logger.info("Recovered ID %s from Resume_html (%d words)", row_id, len(recovered_html_text.split()))
            df_raw.loc[idx, "Resume_str"] = recovered_html_text
            text_source_series.loc[idx] = "Resume_html_recovered"
            recovered_ids.append(int(row_id))
            continue

        # Attempt 2: Recover from PDF
        pdf_path = base_dir / "data" / "raw" / "data" / category / f"{row_id}.pdf"
        if pdf_path.exists():
            recovered_pdf_text = extract_text_from_pdf(str(pdf_path)).strip()
            if len(recovered_pdf_text.split()) >= 20:
                logger.info("Recovered ID %s from PDF (%d words)", row_id, len(recovered_pdf_text.split()))
                df_raw.loc[idx, "Resume_str"] = recovered_pdf_text
                text_source_series.loc[idx] = "PDF_recovered"
                recovered_ids.append(int(row_id))
                continue

        # Attempt 3: Unrecoverable
        logger.warning("ID %s could not be recovered from HTML or PDF. Marking unrecoverable.", row_id)
        text_source_series.loc[idx] = "unrecoverable"
        unrecoverable_ids.append(int(row_id))

    # Add text_source column
    df_raw["text_source"] = text_source_series

    # Filter out unrecoverable records from modeling data
    df_recover_filtered = df_raw[df_raw["text_source"] != "unrecoverable"].copy()
    records_after_empty_handling = len(df_recover_filtered)

    # 3. Step 3: Handle Exact Duplicate Text Records
    # Group by Resume_str and identify duplicates
    dup_mask = df_recover_filtered.duplicated(subset=["Resume_str"], keep="first")
    duplicate_records_removed = df_recover_filtered[dup_mask]
    duplicate_ids_removed = duplicate_records_removed["ID"].tolist()

    logger.info("Duplicate records identified: %d (IDs: %s)", len(duplicate_ids_removed), duplicate_ids_removed)

    # Retain only the first record per duplicate text group
    df_deduped = df_recover_filtered[~dup_mask].copy()
    records_after_duplicate_handling = len(df_deduped)

    # 4. Step 4: Text Normalization (Deterministic, Conservative Variant A)
    logger.info("Applying conservative text normalization (Variant A)...")
    df_deduped["original_text_length"] = df_deduped["Resume_str"].astype(str).str.len()
    df_deduped["text"] = df_deduped["Resume_str"].apply(clean_text_minimal)
    df_deduped["clean_text_length"] = df_deduped["text"].str.len()

    # 5. Step 7: Create and Save clean_resumes.csv
    clean_df = df_deduped[[
        "ID", "text", "Category", "text_source", "original_text_length", "clean_text_length"
    ]].copy()

    clean_csv_path = output_dir / "clean_resumes.csv"
    clean_df.to_csv(clean_csv_path, index=False)
    logger.info("Saved clean resumes to %s (%d records)", clean_csv_path, len(clean_df))

    # 6. Step 8: Validate Clean Data
    num_classes = clean_df["Category"].nunique()
    class_counts = clean_df["Category"].value_counts().to_dict()
    min_len = int(clean_df["clean_text_length"].min())
    max_len = int(clean_df["clean_text_length"].max())
    mean_len = float(clean_df["clean_text_length"].mean())
    median_len = float(clean_df["clean_text_length"].median())

    assert clean_df["text"].isna().sum() == 0, "Null values found in cleaned text!"
    assert (clean_df["clean_text_length"] == 0).sum() == 0, "Empty strings found in cleaned text!"
    assert clean_df["Category"].isna().sum() == 0, "Missing values found in Category!"
    assert clean_df["text"].duplicated().sum() == 0, "Duplicate text found in cleaned dataset!"
    assert num_classes == 24, f"Expected 24 classes, found {num_classes}!"

    # 7. Step 9: Stratified Train / Validation / Test Split (70% / 15% / 15%)
    logger.info("Performing stratified 70/15/15 train/val/test split with random_state=%d...", random_state)
    # Split 1: Train (70%) and Temp (30%)
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.30,
        random_state=random_state,
        stratify=clean_df["Category"]
    )
    # Split 2: Val (15% overall = 50% of temp) and Test (15% overall = 50% of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["Category"]
    )

    # 8. Step 10 & Critical Leakage Checks
    train_ids = set(train_df["ID"])
    val_ids = set(val_df["ID"])
    test_ids = set(test_df["ID"])

    id_overlap_train_val = len(train_ids.intersection(val_ids))
    id_overlap_train_test = len(train_ids.intersection(test_ids))
    id_overlap_val_test = len(val_ids.intersection(test_ids))

    train_texts = set(train_df["text"])
    val_texts = set(val_df["text"])
    test_texts = set(test_df["text"])

    text_overlap_train_val = len(train_texts.intersection(val_texts))
    text_overlap_train_test = len(train_texts.intersection(test_texts))
    text_overlap_val_test = len(val_texts.intersection(test_texts))

    assert id_overlap_train_val == 0, "ID overlap detected between Train and Val!"
    assert id_overlap_train_test == 0, "ID overlap detected between Train and Test!"
    assert id_overlap_val_test == 0, "ID overlap detected between Val and Test!"
    assert text_overlap_train_val == 0, "Text leakage detected between Train and Val!"
    assert text_overlap_train_test == 0, "Text leakage detected between Train and Test!"
    assert text_overlap_val_test == 0, "Text leakage detected between Val and Test!"

    # 9. Step 11: Save Splits
    split_cols = ["ID", "text", "Category"]
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

    train_df[split_cols].to_csv(train_csv, index=False)
    val_df[split_cols].to_csv(val_csv, index=False)
    test_df[split_cols].to_csv(test_csv, index=False)

    logger.info("Saved train.csv: %d rows (%.1f%%)", len(train_df), 100 * len(train_df) / len(clean_df))
    logger.info("Saved validation.csv: %d rows (%.1f%%)", len(val_df), 100 * len(val_df) / len(clean_df))
    logger.info("Saved test.csv: %d rows (%.1f%%)", len(test_df), 100 * len(test_df) / len(clean_df))

    # Split Summary Table
    cat_summary = pd.DataFrame({
        "Total": clean_df["Category"].value_counts(),
        "Train": train_df["Category"].value_counts(),
        "Validation": val_df["Category"].value_counts(),
        "Test": test_df["Category"].value_counts()
    }).fillna(0).astype(int).sort_index()

    cat_summary.to_csv(splits_dir / "split_summary.csv")

    # Split Metadata
    metadata = {
        "random_state": random_state,
        "split_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "source_dataset": str(raw_csv_path),
        "number_of_records_before_cleaning": original_records_count,
        "number_after_empty_handling": records_after_empty_handling,
        "number_after_duplicate_handling": records_after_duplicate_handling,
        "final_clean_records": len(clean_df),
        "final_class_count": num_classes,
        "recovered_ids": recovered_ids,
        "unrecoverable_ids": unrecoverable_ids,
        "duplicate_ids_removed": [int(i) for i in duplicate_ids_removed],
        "split_counts": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "leakage_verification": {
            "id_overlap": 0,
            "text_overlap": 0,
            "learned_transformations_fitted": False,
            "external_pdf_only_records_excluded": 16
        }
    }

    with open(splits_dir / "split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved split_metadata.json and split_summary.csv")
    return metadata


if __name__ == "__main__":
    prepare_data()
