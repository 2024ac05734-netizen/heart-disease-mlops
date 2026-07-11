"""Central configuration and constants for the Heart Disease MLOps project."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_FILE = RAW_DIR / "heart_disease_raw.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DIR / "train.csv"
PROCESSED_TEST_FILE = PROCESSED_DIR / "test.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Dataset schema (UCI Heart Disease - Cleveland processed dataset, 14 columns)
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# Numerical (continuous) features
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Categorical / discrete features
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target"

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# UCI download locations (primary + mirror)
_UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
UCI_URLS = [
    f"{_UCI_BASE}/processed.cleveland.data",
    "https://archive.ics.uci.edu/static/public/45/data.csv",
]

# Human-readable descriptions used in the API schema / docs
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0-2)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment (0-2)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)",
}
