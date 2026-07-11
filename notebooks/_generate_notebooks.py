"""Generate Jupyter notebooks (EDA, training, inference) for the project.

Run once: python notebooks/_generate_notebooks.py
Produces valid .ipynb files that call the reusable heart_mlops package.
"""
import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

SETUP = (
    "import sys, os\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))\n"
)


def build_eda():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 01 - Exploratory Data Analysis\n"
            "Heart Disease UCI dataset. Uses the reusable `heart_mlops` package."
        ),
        new_code_cell(SETUP + "from heart_mlops.data_download import acquire\n"
                              "from heart_mlops.preprocessing import clean_data\n"
                              "df = clean_data(acquire())\n"
                              "df.head()"),
        new_markdown_cell("## Shape, dtypes and missing values"),
        new_code_cell("print(df.shape)\n"
                      "print(df.dtypes)\n"
                      "df.isna().sum()"),
        new_markdown_cell("## Target class balance"),
        new_code_cell("df['target'].value_counts(normalize=True)"),
        new_markdown_cell("## Generate & save all EDA figures"),
        new_code_cell("from heart_mlops.eda import run_eda\n"
                      "run_eda()\n"
                      "print('Figures saved to reports/figures/')"),
        new_markdown_cell(
            "Figures produced: histograms, correlation heatmap, class balance, "
            "missing values, feature-vs-target boxplots."
        ),
    ]
    return nb


def build_training():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 02 - Feature Engineering, Modelling & MLflow Tracking\n"
            "Trains Logistic Regression, Random Forest and XGBoost with GridSearchCV "
            "and logs everything to MLflow."
        ),
        new_code_cell(SETUP + "from heart_mlops.preprocessing import build_preprocessor\n"
                              "pre = build_preprocessor()\n"
                              "pre"),
        new_markdown_cell("## Candidate models and hyperparameter grids"),
        new_code_cell(SETUP + "from heart_mlops.train import get_candidate_models\n"
                              "for m in get_candidate_models():\n"
                              "    print(m.name, '->', m.param_grid)"),
        new_markdown_cell("## Run the full training + tracking workflow"),
        new_code_cell(SETUP + "from heart_mlops.train import train\n"
                              "summary = train()\n"
                              "summary"),
        new_markdown_cell(
            "## Model comparison (leaderboard)\n"
            "Open the MLflow UI with `mlflow ui` to inspect runs, params and plots."
        ),
        new_code_cell("import pandas as pd\n"
                      "pd.DataFrame(summary['leaderboard']).T"),
    ]
    return nb


def build_inference():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 03 - Inference\n"
            "Loads the persisted pipeline and predicts heart-disease risk."
        ),
        new_code_cell(SETUP + "from heart_mlops.predict import predict_one\n"
                              "patient = {\n"
                              "    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,\n"
                              "    'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,\n"
                              "    'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1,\n"
                              "}\n"
                              "predict_one(patient)"),
        new_markdown_cell("## Batch inference"),
        new_code_cell(SETUP + "from heart_mlops.predict import predict_batch\n"
                              "predict_batch([patient, {**patient, 'age': 45, 'cp': 0, 'exang': 0}])"),
    ]
    return nb


def main():
    for name, builder in [
        ("01_eda.ipynb", build_eda),
        ("02_training.ipynb", build_training),
        ("03_inference.ipynb", build_inference),
    ]:
        nb = builder()
        with open(name, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", name)


if __name__ == "__main__":
    main()
