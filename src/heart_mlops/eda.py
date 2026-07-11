"""Exploratory Data Analysis: generates professional visualizations.

Produces histograms, correlation heatmap, class-balance plot, missing-value
analysis and feature-relationship plots, saving all figures to
``reports/figures`` and a text summary to ``reports/eda_summary.txt``.

Run: python -m heart_mlops.eda
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart_mlops import config  # noqa: E402
from heart_mlops.data_download import acquire  # noqa: E402
from heart_mlops.preprocessing import clean_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved %s", path)


def run_eda() -> pd.DataFrame:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_data(acquire())

    # 1. Histograms of all numeric features
    fig = df[config.NUMERIC_FEATURES].hist(figsize=(12, 8), bins=20, edgecolor="black")
    _save(plt.gcf(), "histograms.png")

    # 2. Correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "correlation_heatmap.png")

    # 3. Class balance
    fig, ax = plt.subplots(figsize=(6, 5))
    order = sorted(df[config.TARGET_COLUMN].unique())
    sns.countplot(
        x=config.TARGET_COLUMN, data=df, order=order, ax=ax, hue=config.TARGET_COLUMN, legend=False
    )
    ax.set_title("Target Class Distribution (0 = No disease, 1 = Disease)")
    ax.set_xlabel("Heart Disease Present")
    _save(fig, "class_balance.png")

    # 4. Missing-value analysis
    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=missing.values, y=missing.index, ax=ax, hue=missing.index, legend=False)
    ax.set_title("Missing Values per Feature")
    ax.set_xlabel("Count of missing values")
    _save(fig, "missing_values.png")

    # 5. Feature relationships: boxplots of numeric features vs target
    fig, axes = plt.subplots(1, len(config.NUMERIC_FEATURES), figsize=(20, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(
            x=config.TARGET_COLUMN, y=col, data=df, ax=ax, hue=config.TARGET_COLUMN, legend=False
        )
        ax.set_title(col)
    fig.suptitle("Numeric Features vs Target")
    _save(fig, "feature_vs_target_boxplots.png")

    # 6. Text summary
    summary_lines = [
        "HEART DISEASE UCI - EDA SUMMARY",
        "=" * 40,
        f"Rows: {len(df)}  |  Columns: {df.shape[1]}",
        "",
        "Class distribution:",
        df[config.TARGET_COLUMN].value_counts().to_string(),
        "",
        "Missing values per column:",
        missing[missing > 0].to_string() if missing.sum() else "None",
        "",
        "Numeric feature summary:",
        df[config.NUMERIC_FEATURES].describe().round(2).to_string(),
    ]
    summary_path = config.REPORTS_DIR / "eda_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("EDA complete. Summary at %s", summary_path)
    return df


if __name__ == "__main__":
    run_eda()
