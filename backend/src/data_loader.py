"""
Data loader module for VitaMap ResumeForge backend.
Loads structured resume dataset and raw PDF resumes.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


def load_raw_dataset(
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the raw structured resume dataset (Resume.csv).

    Parameters:
        csv_path: Optional explicit path. Defaults to backend/data/raw/Resume.csv.

    Returns:
        pd.DataFrame containing columns ['ID', 'Resume_str', 'Resume_html', 'Category']
    """
    if csv_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        csv_path = base_dir / "data" / "raw" / "Resume.csv"

    df = pd.read_csv(csv_path)
    return df


def get_raw_pdf_dir(data_dir: Optional[str] = None) -> Path:
    """
    Return path to the categorized raw PDF directory.
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data" / "raw" / "data"
    return Path(data_dir)
