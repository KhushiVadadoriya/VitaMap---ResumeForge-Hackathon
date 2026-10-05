# VitaMap — ResumeForge Backend

Backend and Machine Learning pipeline for the VitaMap Resume Classification System.

## Project Structure
```
backend/
├── data/
│   ├── raw/                  # Original raw data (Resume.csv and categorized PDFs)
│   └── processed/            # Processed artifacts & EDA outputs
├── notebooks/                # Jupyter exploration & experiment notebooks
├── src/                      # Core backend Python package
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading utilities
│   ├── preprocessing.py      # Text cleaning and preprocessing
│   ├── features.py           # Feature engineering & embeddings
│   ├── models.py             # Model architectures and training routines
│   ├── evaluation.py         # Evaluation metrics and error analysis
│   └── predict.py            # Prediction inference pipeline
├── models/                   # Serialized model artifacts
├── api/                      # FastAPI application
│   └── main.py
└── requirements.txt          # Python dependencies
```

## Setup & Virtual Environment
Virtual environment: `backend/.venv` (Python 3.11)
Install dependencies: `pip install -r backend/requirements.txt`

## Inference & API

Final inference artifact: `backend/models/vitamap_final_pipeline.joblib`

Inference path:
```
Raw resume text -> Variant A preprocessing -> TF-IDF (unigram, min_df=2,
max_df=0.95, sublinear_tf=True) -> LinearSVC -> category + decision score
```
Variant A preprocessing is bundled inside the final pipeline, so raw resume
text can be passed directly. `backend/models/vitamap_tfidf_classifier.joblib`
remains the untouched Phase 3 artifact (TF-IDF -> LinearSVC, no preprocessing
step) and is kept for reproducibility.

Start the API from the repository root:

```bash
backend\.venv\Scripts\uvicorn backend.api.main:app --reload
```

### GET /health
```json
{ "status": "ok" }
```

### POST /predict
Request:
```json
{ "resume_text": "John Doe ... Python ... SQL ..." }
```
Response:
```json
{
  "category": "INFORMATION-TECHNOLOGY",
  "decision_score": 2.84,
  "top_predictions": [
    { "category": "INFORMATION-TECHNOLOGY", "decision_score": 2.84 }
  ]
}
```
Empty or whitespace-only input returns HTTP 400:
```json
{ "detail": "resume_text cannot be empty" }
```

**Note on `decision_score`:** this is the raw LinearSVC `decision_function`
value. It is NOT a probability and NOT a confidence percentage. Larger (more
positive) values indicate stronger model support for the class, but the
values are not bounded to [0, 1].

### GET /metrics
Returns the already-computed final test metrics (no re-evaluation is run):
```json
{
  "accuracy": 0.6863,
  "macro_precision": 0.7194,
  "macro_recall": 0.6629,
  "macro_f1": 0.6658,
  "weighted_f1": 0.6757
}
```
These are held-out test evaluation metrics for the frozen model, not
prediction confidence.

## Model Artifact Used
- `backend/models/vitamap_final_pipeline.joblib` (inference)
- `backend/models/vitamap_tfidf_classifier.joblib` (Phase 3 reproducibility)
- `backend/models/model_metadata.json` (model config + metrics)

## Preprocessing Used
Variant A (`clean_text_minimal` in `backend/src/preprocessing.py`): HTML
entity unescape, tag strip, Unicode NFKC normalization, control-char removal,
bullet/quote/dash normalization, whitespace collapse. Case and technical
tokens (C++, C#, .NET, SQL, Python, AWS, ...) are preserved. No lowercasing,
no stopword removal, no stemming/lemmatization.
