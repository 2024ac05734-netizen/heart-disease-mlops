"""Data cleaning, preprocessing pipeline and train/test splitting.

Exposes a reusable sklearn ``ColumnTransformer`` so the exact same
preprocessing is applied at training and inference time (reproducibility).
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_mlops import config

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, coerce numeric types, binarise target."""
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    for col in config.FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure a binary target even if raw values are 0-4.
    df[config.TARGET_COLUMN] = (
        pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce").fillna(0) > 0
    ).astype(int)

    logger.info("Cleaned data: %d rows, %d cols", df.shape[0], df.shape[1])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer: impute+scale numeric, impute+one-hot categorical."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def split_data(df: pd.DataFrame):
    """Stratified train/test split returning X_train, X_test, y_train, y_test."""
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def prepare_and_save() -> None:
    """End-to-end: load raw -> clean -> split -> persist processed CSVs."""
    from heart_mlops.data_download import acquire

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_data(acquire())
    X_train, X_test, y_train, y_test = split_data(df)
    train = X_train.copy()
    train[config.TARGET_COLUMN] = y_train.values
    test = X_test.copy()
    test[config.TARGET_COLUMN] = y_test.values
    train.to_csv(config.PROCESSED_TRAIN_FILE, index=False)
    test.to_csv(config.PROCESSED_TEST_FILE, index=False)
    logger.info("Saved processed train (%d) and test (%d) sets.", len(train), len(test))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare_and_save()
