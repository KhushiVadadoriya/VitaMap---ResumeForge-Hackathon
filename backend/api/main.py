"""
FastAPI application entrypoint for VitaMap ResumeForge.

Thin HTTP layer only — all inference logic lives in src.predict.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.src.predict import get_final_test_metrics, predict_resume

app = FastAPI(title="VitaMap ResumeForge API", version="1.0.0")


class PredictRequest(BaseModel):
    resume_text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.resume_text or not request.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")
    try:
        result = predict_resume(request.resume_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/metrics")
def metrics():
    return get_final_test_metrics()