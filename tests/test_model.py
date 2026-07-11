"""Unit tests for model training candidates and inference."""

import numpy as np
from sklearn.pipeline import Pipeline

from heart_mlops.data_download import _synthetic_fallback
from heart_mlops.preprocessing import build_preprocessor, clean_data, split_data
from heart_mlops.train import get_candidate_models


def test_candidate_models_present():
    names = {m.name for m in get_candidate_models()}
    # At least the two required classifiers must be available
    assert {"logistic_regression", "random_forest"}.issubset(names)


def test_pipeline_trains_and_predicts_probabilities():
    df = clean_data(_synthetic_fallback(n=300))
    X_train, X_test, y_train, y_test = split_data(df)
    cand = get_candidate_models()[0]  # logistic regression
    pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", cand.estimator)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert np.all((proba >= 0) & (proba <= 1))


def test_trained_model_beats_random():
    df = clean_data(_synthetic_fallback(n=400))
    X_train, X_test, y_train, y_test = split_data(df)
    cand = get_candidate_models()[1]  # random forest
    pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("classifier", cand.estimator)])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    assert acc > 0.6
