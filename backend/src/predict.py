"""
End-to-end inference pipeline for VitaMap ResumeForge.

Final inference path (identical to the winning Phase 3 configuration):

    Raw resume text
        -> Variant A preprocessing (clean_text_minimal)
        -> TF-IDF (ngram_range=(1,1), min_df=2, max_df=0.95, sublinear_tf=True)
        -> LinearSVC (class_weight='balanced', random_state=42, dual='auto')
        -> predicted category + decision scores

The final artifact bundles Variant A preprocessing inside the sklearn
Pipeline, so raw resume text can be passed directly to it.

The Phase 3 artifact (models/vitamap_tfidf_classifier.joblib) is preserved
unchanged for reproducibility. The final artifact is a separate file:
models/vitamap_final_pipeline.joblib.

IMPORTANT:
- The TF-IDF vocabulary and LinearSVC weights are reused AS-IS from the
  Phase 3 artifact (fitted ONLY on backend/data/processed/splits/train.csv
  with the exact winning configuration). Nothing is refit on validation
  or test data.
- LinearSVC does NOT produce probabilities. Scores exposed here are raw
  decision_function values, NOT calibrated confidences.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from backend.src.preprocessing import apply_variant_a

_BACKEND_DIR = Path(__file__).resolve().parent.parent
PHASE3_ARTIFACT_PATH = _BACKEND_DIR / "models" / "vitamap_tfidf_classifier.joblib"
FINAL_ARTIFACT_PATH = _BACKEND_DIR / "models" / "vitamap_final_pipeline.joblib"
TEST_METRICS_PATH = _BACKEND_DIR / "data" / "processed" / "modeling" / "test_metrics.json"

_PIPELINE: Optional[Pipeline] = None


def build_final_pipeline(phase3_artifact_path: Optional[str] = None) -> Pipeline:
    """Wrap the Phase 3 TF-IDF -> LinearSVC pipeline with Variant A preprocessing.

    Reuses the already-fitted TF-IDF vectorizer and LinearSVC estimator
    verbatim; no refitting is performed, so the Phase 3 test metrics
    remain valid for this artifact.
    """
    path = Path(phase3_artifact_path) if phase3_artifact_path else PHASE3_ARTIFACT_PATH
    phase3_pipeline = joblib.load(path)

    tfidf = phase3_pipeline.named_steps["tfidf"]
    classifier = phase3_pipeline.named_steps["classifier"]

    return Pipeline([
        ("preprocess", FunctionTransformer(apply_variant_a)),
        ("tfidf", tfidf),
        ("classifier", classifier),
    ])


def save_final_pipeline(output_path: Optional[str] = None) -> Path:
    """Build the final pipeline from the Phase 3 artifact and persist it."""
    final_pipeline = build_final_pipeline()
    out_path = Path(output_path) if output_path else FINAL_ARTIFACT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, out_path)
    return out_path


def load_pipeline(artifact_path: Optional[str] = None) -> Pipeline:
    """Load (and cache) the final inference pipeline from disk."""
    global _PIPELINE
    if artifact_path is not None:
        return joblib.load(artifact_path)
    if _PIPELINE is None:
        path = Path(artifact_path) if artifact_path else FINAL_ARTIFACT_PATH
        if not path.exists():
            save_final_pipeline()
        _PIPELINE = joblib.load(path)
    return _PIPELINE


def predict_resume(resume_text: str, top_k: int = 5) -> Dict[str, Any]:
    """Classify a raw resume text into one of the 24 known categories.

    Parameters:
        resume_text: Raw (unpreprocessed) resume text.
        top_k: Number of top predictions to include (by decision score).

    Returns:
        {
            "category": <predicted category>,
            "decision_score": <float, raw LinearSVC decision score>,
            "top_predictions": [{"category": ..., "decision_score": ...}, ...]
        }

    Raises:
        ValueError: if the input is empty or whitespace-only.
    """
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("resume_text cannot be empty")

    pipeline = load_pipeline()
    classifier = pipeline.named_steps["classifier"]
    classes: List[str] = list(classifier.classes_)

    scores = pipeline.decision_function([resume_text])[0]
    order = np.argsort(scores)[::-1]

    top = [
        {"category": classes[i], "decision_score": float(scores[i])}
        for i in order[: max(1, top_k)]
    ]
    return {
        "category": top[0]["category"],
        "decision_score": top[0]["decision_score"],
        "top_predictions": top,
    }


def get_final_test_metrics() -> Dict[str, float]:
    """Return the already-computed final test metrics (no re-evaluation)."""
    with open(TEST_METRICS_PATH, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {
        "accuracy": m["accuracy"],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
    }


if __name__ == "__main__":
    sample = (
        "Senior Software Engineer with 6 years experience in Python, Django, "
        "PostgreSQL, Docker, REST APIs, and AWS cloud infrastructure."
    )
    result = predict_resume(sample)
    print(json.dumps(result, indent=2))
