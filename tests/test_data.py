"""Unit tests for data acquisition, cleaning and the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from heart_mlops import config
from heart_mlops.data_download import _synthetic_fallback
from heart_mlops.preprocessing import build_preprocessor, clean_data, split_data


def _raw_frame() -> pd.DataFrame:
    return _synthetic_fallback(n=200)


def test_synthetic_fallback_schema():
    df = _raw_frame()
    assert list(df.columns) == config.COLUMN_NAMES
    assert len(df) == 200
    # Missing values were injected into ca/thal
    assert df[["ca", "thal"]].isna().sum().sum() > 0


def test_clean_data_binarises_target():
    df = _raw_frame()
    df.loc[0, config.TARGET_COLUMN] = 4  # raw UCI target can be 0-4
    cleaned = clean_data(df)
    assert set(cleaned[config.TARGET_COLUMN].unique()).issubset({0, 1})


def test_clean_data_removes_duplicates():
    df = _raw_frame()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(dup)
    assert len(cleaned) <= len(dup)


def test_split_is_stratified_and_correct_size():
    df = clean_data(_raw_frame())
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_test) == pytest.approx(len(df) * config.TEST_SIZE, abs=2)
    assert len(X_train) + len(X_test) == len(df)
    # stratification keeps class ratio roughly stable
    assert abs(y_train.mean() - y_test.mean()) < 0.15


def test_preprocessor_handles_missing_and_outputs_numeric():
    df = clean_data(_raw_frame())
    X_train, X_test, y_train, _ = split_data(df)
    pre = build_preprocessor()
    transformed = pre.fit_transform(X_train, y_train)
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(arr).any()
    assert arr.shape[0] == len(X_train)
    # numeric features should be scaled (mean ~0)
    assert abs(arr[:, 0].mean()) < 1.0
