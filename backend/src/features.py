"""
Feature engineering module for VitaMap ResumeForge.
Provides standardized TF-IDF configurations and feature utilities.
"""

from typing import Any, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A": {
        "description": "Unigram only, min_df=2, max_df=0.95, sublinear_tf=True",
        "params": {
            "ngram_range": (1, 1),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
    },
    "B": {
        "description": "Unigram + Bigram, min_df=2, max_df=0.95, sublinear_tf=True",
        "params": {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
    },
    "C": {
        "description": "Unigram + Bigram, min_df=1, max_df=0.95, sublinear_tf=True",
        "params": {
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
    },
    "D": {
        "description": "Unigram + Bigram, min_df=2, max_df=1.0, sublinear_tf=True",
        "params": {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 1.0,
            "sublinear_tf": True,
        },
    },
}


def build_tfidf_vectorizer(
    config_name: str = "B",
    custom_params: Optional[Dict[str, Any]] = None
) -> TfidfVectorizer:
    """
    Build a TfidfVectorizer instance from a predefined configuration.

    Parameters:
        config_name: Key from TFIDF_CONFIGS ('A', 'B', 'C', 'D').
        custom_params: Optional dictionary of parameter overrides.

    Returns:
        Configured unfitted TfidfVectorizer.
    """
    cfg_key = config_name.upper().strip()
    if cfg_key not in TFIDF_CONFIGS:
        raise ValueError(
            f"Unknown TF-IDF config '{config_name}'. Expected one of {list(TFIDF_CONFIGS.keys())}."
        )

    params = dict(TFIDF_CONFIGS[cfg_key]["params"])
    if custom_params:
        params.update(custom_params)

    return TfidfVectorizer(**params)
