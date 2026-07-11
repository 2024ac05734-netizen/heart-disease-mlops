"""Pytest configuration: ensure src/ is importable and a model exists."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    """Guarantee a trained model exists before any test runs.

    On a clean checkout / CI runner there is no ``models/model.joblib`` (it is
    a build artifact, not committed), which would make the API return 503 on
    ``/predict``. To keep the test-suite self-contained and network-free we
    train a small logistic-regression pipeline on a synthetic, schema-accurate
    dataset when the real model is absent. When the model already exists this
    fixture is a no-op.
    """
    import joblib

    from heart_mlops import config
    from heart_mlops.predict import load_model

    if not config.MODEL_FILE.exists():
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        from heart_mlops.preprocessing import build_preprocessor

        rng = np.random.default_rng(config.RANDOM_STATE)
        n = 300
        X = pd.DataFrame(
            {
                "age": rng.integers(29, 77, n),
                "sex": rng.integers(0, 2, n),
                "cp": rng.integers(0, 4, n),
                "trestbps": rng.integers(94, 200, n),
                "chol": rng.integers(126, 564, n),
                "fbs": rng.integers(0, 2, n),
                "restecg": rng.integers(0, 3, n),
                "thalach": rng.integers(71, 202, n),
                "exang": rng.integers(0, 2, n),
                "oldpeak": rng.uniform(0, 6.2, n).round(1),
                "slope": rng.integers(0, 3, n),
                "ca": rng.integers(0, 4, n),
                "thal": rng.integers(1, 4, n),
            },
            columns=config.FEATURE_COLUMNS,
        )
        logits = (X["age"] - 50) / 10 + X["oldpeak"] + X["ca"] + (X["cp"] == 0).astype(int) - 2
        y = (logits + rng.normal(0, 1, n) > 0).astype(int)

        pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
        pipe.fit(X, y)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, config.MODEL_FILE)

    # Drop any cached FileNotFound/stale handle so the API picks up the model.
    load_model.cache_clear()
    return config.MODEL_FILE


@pytest.fixture(scope="session")
def sample_patient() -> dict:
    return {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
