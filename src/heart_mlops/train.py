"""Model training, hyperparameter tuning, evaluation and MLflow tracking.

Trains Logistic Regression, Random Forest and (optionally) XGBoost inside a
single sklearn Pipeline (preprocessor + classifier). Uses GridSearchCV with
stratified k-fold cross-validation, logs everything to MLflow, and persists the
best pipeline to ``models/model.joblib``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from heart_mlops import config  # noqa: E402
from heart_mlops.data_download import acquire  # noqa: E402
from heart_mlops.preprocessing import build_preprocessor, clean_data, split_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # noqa: BLE001
    _HAS_XGB = False


@dataclass
class CandidateModel:
    name: str
    estimator: object
    param_grid: dict


def get_candidate_models() -> list[CandidateModel]:
    """Return the candidate classifiers and their hyperparameter search grids."""
    models = [
        CandidateModel(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            param_grid={
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
            },
        ),
        CandidateModel(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=config.RANDOM_STATE),
            param_grid={
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
            },
        ),
    ]
    if _HAS_XGB:
        models.append(
            CandidateModel(
                name="xgboost",
                estimator=XGBClassifier(
                    random_state=config.RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
                param_grid={
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [3, 5],
                    "classifier__learning_rate": [0.05, 0.1],
                },
            )
        )
    return models


def _evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def _plot_roc(y_test, y_proba, name: str):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"roc_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_confusion(pipe, X_test, y_test, name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"confusion_{name}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train(experiment_name: str = "heart-disease-classification") -> dict:
    """Run the full training + tuning + tracking workflow. Returns best metrics."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_data(acquire())
    X_train, X_test, y_train, y_test = split_data(df)

    mlflow.set_experiment(experiment_name)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    best_auc = -1.0
    best_name = None
    best_pipe = None
    best_metrics: dict = {}
    leaderboard: dict[str, dict] = {}

    for cand in get_candidate_models():
        with mlflow.start_run(run_name=cand.name):
            pipe = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("classifier", cand.estimator),
                ]
            )
            search = GridSearchCV(
                pipe,
                param_grid=cand.param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            tuned = search.best_estimator_

            # Cross-validation score on the full training set for the tuned model
            cv_scores = cross_val_score(tuned, X_train, y_train, cv=cv, scoring="roc_auc")
            metrics = _evaluate(tuned, X_test, y_test)
            metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
            metrics["cv_roc_auc_std"] = float(cv_scores.std())

            # ---- MLflow logging ----
            mlflow.log_param("model_type", cand.name)
            mlflow.log_params(search.best_params_)
            mlflow.log_param("cv_folds", config.CV_FOLDS)
            mlflow.log_metrics(metrics)

            y_proba = tuned.predict_proba(X_test)[:, 1]
            roc_path = _plot_roc(y_test, y_proba, cand.name)
            cm_path = _plot_confusion(tuned, X_test, y_test, cand.name)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            mlflow.sklearn.log_model(tuned, artifact_path="model")

            leaderboard[cand.name] = metrics
            logger.info("%s -> %s", cand.name, {k: round(v, 4) for k, v in metrics.items()})

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_name = cand.name
                best_pipe = tuned
                best_metrics = metrics

    # ---- Persist the best pipeline + a feature-importance style summary ----
    assert best_pipe is not None
    joblib.dump(best_pipe, config.MODEL_FILE)
    summary = {
        "best_model": best_name,
        "best_metrics": best_metrics,
        "leaderboard": leaderboard,
        "xgboost_available": _HAS_XGB,
    }
    config.METRICS_FILE.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Best model: %s (ROC-AUC=%.4f). Saved to %s", best_name, best_auc, config.MODEL_FILE
    )

    # Register best model in MLflow with a dedicated run
    with mlflow.start_run(run_name=f"best_{best_name}"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_pipe, artifact_path="best_model")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train heart disease classifiers")
    parser.add_argument(
        "--experiment", default="heart-disease-classification", help="MLflow experiment name"
    )
    args = parser.parse_args()
    train(experiment_name=args.experiment)


if __name__ == "__main__":
    main()
