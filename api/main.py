"""FastAPI application serving the Heart Disease classifier.

Endpoints:
    GET  /            -> service metadata
    GET  /health      -> liveness/readiness probe
    POST /predict     -> single prediction (JSON in, prediction + confidence out)
    POST /predict/batch -> batch predictions
    GET  /metrics     -> Prometheus metrics (via instrumentator)

Request/response logging and a custom Prometheus counter are included for the
monitoring requirement.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from heart_mlops import config
from heart_mlops.predict import load_model, predict_batch, predict_one

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart_api")

# ---------------------------------------------------------------------------
# Custom Prometheus metrics
# ---------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter("heart_predictions_total", "Total predictions served", ["outcome"])
PREDICTION_LATENCY = Histogram("heart_prediction_latency_seconds", "Latency of /predict handler")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts the risk of heart disease from patient health data.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    age: float = Field(..., example=63, description=config.FEATURE_DESCRIPTIONS["age"])
    sex: int = Field(..., example=1, description=config.FEATURE_DESCRIPTIONS["sex"])
    cp: int = Field(..., example=3, description=config.FEATURE_DESCRIPTIONS["cp"])
    trestbps: float = Field(..., example=145, description=config.FEATURE_DESCRIPTIONS["trestbps"])
    chol: float = Field(..., example=233, description=config.FEATURE_DESCRIPTIONS["chol"])
    fbs: int = Field(..., example=1, description=config.FEATURE_DESCRIPTIONS["fbs"])
    restecg: int = Field(..., example=0, description=config.FEATURE_DESCRIPTIONS["restecg"])
    thalach: float = Field(..., example=150, description=config.FEATURE_DESCRIPTIONS["thalach"])
    exang: int = Field(..., example=0, description=config.FEATURE_DESCRIPTIONS["exang"])
    oldpeak: float = Field(..., example=2.3, description=config.FEATURE_DESCRIPTIONS["oldpeak"])
    slope: int = Field(..., example=0, description=config.FEATURE_DESCRIPTIONS["slope"])
    ca: float = Field(..., example=0, description=config.FEATURE_DESCRIPTIONS["ca"])
    thal: float = Field(..., example=1, description=config.FEATURE_DESCRIPTIONS["thal"])


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    confidence: float


class BatchRequest(BaseModel):
    records: list[PatientFeatures]


# ---------------------------------------------------------------------------
# Middleware: request logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ---------------------------------------------------------------------------
# Lifecycle: warm the model, install Prometheus instrumentation
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    try:
        load_model()
        logger.info("Model loaded successfully at startup.")
    except FileNotFoundError as exc:
        logger.warning("Model not available at startup: %s", exc)


Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    return {
        "service": "Heart Disease Risk Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/predict/batch", "/metrics", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": config.MODEL_FILE.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    try:
        with PREDICTION_LATENCY.time():
            result = predict_one(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    PREDICTIONS_TOTAL.labels(outcome=result["label"]).inc()
    logger.info("Prediction: %s (p=%.3f)", result["label"], result["probability"])
    return PredictionResponse(**result)


@app.post("/predict/batch")
def predict_many(payload: BatchRequest) -> dict:
    try:
        results = predict_batch([r.model_dump() for r in payload.records])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    for r in results:
        PREDICTIONS_TOTAL.labels(outcome=r["label"]).inc()
    return {"count": len(results), "predictions": results}
