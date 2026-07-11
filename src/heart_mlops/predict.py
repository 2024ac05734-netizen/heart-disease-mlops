"""Inference helpers: load the trained pipeline and produce predictions."""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import pandas as pd

from heart_mlops import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained sklearn pipeline from disk."""
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_FILE}. Run `python -m heart_mlops.train` first."
        )
    logger.info("Loading model from %s", config.MODEL_FILE)
    return joblib.load(config.MODEL_FILE)


def predict_one(features: dict) -> dict:
    """Predict for a single patient record given a feature dict.

    Returns a dict with the binary prediction, human label and probability.
    """
    model = load_model()
    row = {col: features.get(col) for col in config.FEATURE_COLUMNS}
    X = pd.DataFrame([row], columns=config.FEATURE_COLUMNS)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "label": "Heart Disease" if pred == 1 else "No Heart Disease",
        "probability": round(proba, 4),
        "confidence": round(proba if pred == 1 else 1 - proba, 4),
    }


def predict_batch(records: list[dict]) -> list[dict]:
    return [predict_one(r) for r in records]
