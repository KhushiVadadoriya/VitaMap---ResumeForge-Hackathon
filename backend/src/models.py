"""
Classical Machine Learning models and Pipeline definitions for VitaMap ResumeForge.
Implements Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.src.features import build_tfidf_vectorizer


def build_multinomial_nb(**kwargs) -> MultinomialNB:
    """Build MultinomialNB with optional kwargs."""
    params = {"alpha": 1.0}
    params.update(kwargs)
    return MultinomialNB(**params)


def build_logistic_regression(**kwargs) -> LogisticRegression:
    """Build LogisticRegression with class_weight='balanced' and random_state=42."""
    params = {
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
        "solver": "lbfgs",
    }
    params.update(kwargs)
    return LogisticRegression(**params)


def build_linear_svc(**kwargs) -> LinearSVC:
    """Build LinearSVC with class_weight='balanced' and random_state=42."""
    params = {
        "class_weight": "balanced",
        "random_state": 42,
        "max_iter": 5000,
        "dual": "auto",
    }
    params.update(kwargs)
    return LinearSVC(**params)


MODEL_BUILDERS = {
    "nb": build_multinomial_nb,
    "multinomial_nb": build_multinomial_nb,
    "logreg": build_logistic_regression,
    "logistic_regression": build_logistic_regression,
    "svm": build_linear_svc,
    "linear_svc": build_linear_svc,
}


def build_model(model_name: str, **kwargs) -> Any:
    """Build classifier by key."""
    key = model_name.lower().strip()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[key](**kwargs)


def build_pipeline(
    tfidf_config: str = "B",
    model_name: str = "linear_svc",
    custom_tfidf_params: Optional[Dict[str, Any]] = None,
    custom_model_params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Build coupled scikit-learn Pipeline with TF-IDF Vectorizer and Classifier.
    """
    vectorizer = build_tfidf_vectorizer(tfidf_config, custom_tfidf_params)
    classifier = build_model(model_name, **(custom_model_params or {}))

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def extract_top_features_by_class(
    pipeline: Pipeline,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Extract top high-weight features per class for linear models (LinearSVC, LogisticRegression).

    Returns:
        pd.DataFrame with columns ['class', 'rank', 'feature', 'weight']
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        raise ValueError("Classifier does not have coef_ attribute (e.g. Naive Bayes).")

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = classifier.classes_
    coefs = classifier.coef_

    records = []
    for idx, class_name in enumerate(classes):
        class_coefs = coefs[idx]
        # Top positive indices
        top_indices = np.argsort(class_coefs)[::-1][:top_n]
        for rank, feat_idx in enumerate(top_indices, start=1):
            records.append({
                "class": class_name,
                "rank": rank,
                "feature": feature_names[feat_idx],
                "weight": float(class_coefs[feat_idx]),
            })

    return pd.DataFrame(records)
