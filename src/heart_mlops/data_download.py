"""Dataset acquisition for the UCI Heart Disease dataset.

Downloads the Cleveland processed dataset from the UCI Machine Learning
Repository. If the network is unavailable, a deterministic synthetic fallback
with the same schema is generated so the pipeline remains fully reproducible
in offline / CI environments.

Usage:
    python -m heart_mlops.data_download
    python src/heart_mlops/data_download.py --force
"""

from __future__ import annotations

import argparse
import io
import logging
import urllib.request

import numpy as np
import pandas as pd

from heart_mlops import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _download_from_uci() -> pd.DataFrame | None:
    """Try each UCI mirror; return a raw DataFrame or None on failure."""
    for url in config.UCI_URLS:
        try:
            logger.info("Attempting download from %s", url)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                content = resp.read().decode("utf-8", errors="replace")
            df = pd.read_csv(
                io.StringIO(content),
                header=None,
                names=config.COLUMN_NAMES,
                na_values="?",
            )
            if df.shape[1] == len(config.COLUMN_NAMES) and len(df) > 100:
                logger.info("Downloaded %d rows from UCI.", len(df))
                return df
            logger.warning("Unexpected shape %s from %s", df.shape, url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Download failed from %s: %s", url, exc)
    return None


def _synthetic_fallback(n: int = 303) -> pd.DataFrame:
    """Generate a deterministic, schema-compatible dataset for offline use."""
    logger.warning("Falling back to synthetic dataset (offline mode).")
    rng = np.random.default_rng(config.RANDOM_STATE)
    age = rng.integers(29, 78, n)
    sex = rng.integers(0, 2, n)
    cp = rng.integers(0, 4, n)
    trestbps = rng.integers(94, 201, n)
    chol = rng.integers(126, 565, n)
    fbs = rng.integers(0, 2, n)
    restecg = rng.integers(0, 3, n)
    thalach = rng.integers(71, 203, n)
    exang = rng.integers(0, 2, n)
    oldpeak = np.round(rng.uniform(0, 6.2, n), 1)
    slope = rng.integers(0, 3, n)
    ca = rng.integers(0, 4, n).astype(float)
    thal = rng.choice([1, 2, 3], n).astype(float)

    # Construct a target correlated with clinically relevant features
    risk = (
        0.03 * (age - 50)
        + 0.7 * sex
        + 0.5 * cp
        + 0.02 * (trestbps - 130)
        + 0.004 * (chol - 240)
        - 0.03 * (thalach - 150)
        + 0.9 * exang
        + 0.6 * oldpeak
        + 0.5 * ca
        + 0.4 * (thal == 3)
    )
    prob = 1 / (1 + np.exp(-(risk - risk.mean()) / (risk.std() + 1e-9)))
    target = (rng.uniform(0, 1, n) < prob).astype(int)

    # Inject a few missing values to mirror the real dataset (ca, thal)
    ca[rng.choice(n, 4, replace=False)] = np.nan
    thal[rng.choice(n, 2, replace=False)] = np.nan

    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "target": target,
        }
    )


def acquire(force: bool = False) -> pd.DataFrame:
    """Download (or load cached) the raw dataset and persist it to data/raw."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.RAW_DATA_FILE.exists() and not force:
        logger.info("Using cached raw dataset at %s", config.RAW_DATA_FILE)
        return pd.read_csv(config.RAW_DATA_FILE)

    df = _download_from_uci()
    if df is None:
        df = _synthetic_fallback()

    # The raw UCI target 'num' is 0-4; convert to binary presence/absence.
    df[config.TARGET_COLUMN] = (df[config.TARGET_COLUMN].fillna(0).astype(float) > 0).astype(int)
    df.to_csv(config.RAW_DATA_FILE, index=False)
    logger.info("Saved raw dataset (%d rows) to %s", len(df), config.RAW_DATA_FILE)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI Heart Disease dataset")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()
    acquire(force=args.force)


if __name__ == "__main__":
    main()
