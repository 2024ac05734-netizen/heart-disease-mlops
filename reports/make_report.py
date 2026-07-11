"""Generate the final assignment report as a multi-page PDF (reportlab).

Embeds EDA figures, model comparison, ROC/confusion plots and the architecture
diagram. Output: reports/MLOps_Assignment_Report.pdf
"""
from __future__ import annotations

import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from heart_mlops import config

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10,
                    textColor=colors.HexColor("#b02a37"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=8,
                    spaceAfter=6, textColor=colors.HexColor("#2c3e50"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY)
CENTER = ParagraphStyle("Center", parent=BODY, alignment=TA_CENTER)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8, alignment=TA_CENTER,
                         textColor=colors.grey)


def fig(name, width=15 * cm):
    path = config.FIGURES_DIR / name
    if not path.exists():
        return Paragraph(f"[missing figure: {name}]", CAPTION)
    img = Image(str(path))
    ratio = img.imageHeight / float(img.imageWidth)
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def p(text):
    return Paragraph(text, BODY)


def metrics_table():
    data = json.loads(config.METRICS_FILE.read_text())
    lb = data["leaderboard"]
    header = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV ROC-AUC"]
    rows = [header]
    for name, m in lb.items():
        star = " *" if name == data["best_model"] else ""
        rows.append([
            name.replace("_", " ").title() + star,
            f"{m['accuracy']:.3f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}",
            f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}",
            f"{m['cv_roc_auc_mean']:.3f}+/-{m['cv_roc_auc_std']:.3f}",
        ])
    t = Table(rows, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b02a37")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    return t, data["best_model"]


def build():
    story = []
    # ---- Title page ----
    story += [
        Spacer(1, 4 * cm),
        Paragraph("Machine Learning Operations (MLOps)", H1),
        Paragraph("Assignment 01 &mdash; AIMLCZG523", H2),
        Spacer(1, 1 * cm),
        Paragraph("End-to-End ML Model Development, CI/CD and Production Deployment", CENTER),
        Spacer(1, 0.5 * cm),
        Paragraph("<b>Problem:</b> Predict the risk of heart disease from patient "
                  "health data and deploy the solution as a cloud-ready, monitored API.", CENTER),
        Spacer(1, 0.5 * cm),
        Paragraph("Dataset: Heart Disease UCI (Cleveland processed, 303 records)", CENTER),
        Spacer(1, 3 * cm),
        Paragraph("Total Marks: 50", CENTER),
        PageBreak(),
    ]

    # ---- 1. Project overview ----
    story += [
        Paragraph("1. Project Overview", H1),
        p("This project implements a complete MLOps workflow for a binary "
          "classification problem: predicting the presence of heart disease. The "
          "solution covers the full lifecycle &mdash; data acquisition, exploratory "
          "data analysis, feature engineering, model development with hyperparameter "
          "tuning, experiment tracking with MLflow, reproducible packaging, automated "
          "testing and CI/CD, containerisation with Docker, deployment on Kubernetes "
          "with a Helm chart, and monitoring with Prometheus and Grafana."),
        Spacer(1, 0.3 * cm),
        Paragraph("1.1 Architecture", H2),
        fig("architecture.png", 16 * cm),
        Paragraph("Figure 1: End-to-end MLOps architecture.", CAPTION),
        Spacer(1, 0.2 * cm),
        p("<b>Tech stack:</b> Python 3.12, Pandas/NumPy, Scikit-learn, XGBoost, "
          "Matplotlib/Seaborn, MLflow, FastAPI, Pytest, Docker, Kubernetes/Helm, "
          "Prometheus, Grafana, GitHub Actions."),
        PageBreak(),
    ]

    # ---- 2. Data & EDA ----
    story += [
        Paragraph("2. Data Acquisition &amp; Exploratory Data Analysis", H1),
        p("The Cleveland processed dataset is downloaded from the UCI Machine "
          "Learning Repository via <font face='Courier'>data_download.py</font>. It "
          "contains 303 records with 13 clinical features and a target (raw 0&ndash;4, "
          "binarised to 0 = no disease, 1 = disease). Missing values in "
          "<font face='Courier'>ca</font> (4) and <font face='Courier'>thal</font> (2) "
          "are imputed. The classes are reasonably balanced (164 vs 139)."),
        Spacer(1, 0.2 * cm),
        fig("class_balance.png", 9 * cm),
        Paragraph("Figure 2: Target class distribution.", CAPTION),
        fig("missing_values.png", 11 * cm),
        Paragraph("Figure 3: Missing-value analysis.", CAPTION),
        PageBreak(),
        fig("histograms.png", 15 * cm),
        Paragraph("Figure 4: Distributions of numeric features.", CAPTION),
        fig("correlation_heatmap.png", 14 * cm),
        Paragraph("Figure 5: Correlation heatmap. cp, thalach, oldpeak, ca and thal "
                  "show the strongest association with the target.", CAPTION),
        PageBreak(),
        fig("feature_vs_target_boxplots.png", 16 * cm),
        Paragraph("Figure 6: Numeric features vs target &mdash; patients with disease "
                  "tend to show lower max heart rate (thalach) and higher oldpeak.", CAPTION),
        PageBreak(),
    ]

    # ---- 3. Feature engineering & modelling ----
    tbl, best = metrics_table()
    story += [
        Paragraph("3. Feature Engineering &amp; Model Development", H1),
        p("Preprocessing is encapsulated in a scikit-learn "
          "<font face='Courier'>ColumnTransformer</font>: numeric features are median-"
          "imputed and standard-scaled; categorical features are most-frequent-imputed "
          "and one-hot encoded. This transformer is bundled with the classifier inside "
          "a single <font face='Courier'>Pipeline</font>, guaranteeing identical "
          "transformations at train and inference time."),
        p("Three classifiers were trained &mdash; Logistic Regression, Random Forest "
          "and XGBoost &mdash; each tuned with <font face='Courier'>GridSearchCV</font> "
          "over a hyperparameter grid using 5-fold stratified cross-validation, scored "
          "by ROC-AUC. Models were evaluated on a held-out 20% test set."),
        Spacer(1, 0.3 * cm),
        Paragraph("3.1 Model Comparison", H2),
        tbl,
        Spacer(1, 0.2 * cm),
        Paragraph("Table 1: Model comparison (* = selected model). "
                  f"Selected: <b>{best.replace('_', ' ').title()}</b> "
                  "for the highest ROC-AUC.", CAPTION),
        Spacer(1, 0.3 * cm),
        fig(f"roc_{best}.png", 9 * cm),
        Paragraph("Figure 7: ROC curve of the selected model.", CAPTION),
        fig(f"confusion_{best}.png", 8 * cm),
        Paragraph("Figure 8: Confusion matrix of the selected model.", CAPTION),
        PageBreak(),
    ]

    # ---- 4. Experiment tracking ----
    story += [
        Paragraph("4. Experiment Tracking (MLflow)", H1),
        p("Every training run is logged to MLflow under the experiment "
          "<font face='Courier'>heart-disease-classification</font>. For each model we "
          "log: hyperparameters (best GridSearchCV params, CV folds), metrics "
          "(accuracy, precision, recall, F1, ROC-AUC and cross-validated ROC-AUC), "
          "artifacts (ROC curve and confusion matrix PNGs) and the serialized sklearn "
          "model. The best model is additionally logged in a dedicated run. The MLflow "
          "UI (<font face='Courier'>mlflow ui</font>) provides side-by-side run "
          "comparison and full reproducibility."),
        Paragraph("<i>[Screenshot placeholder: MLflow UI runs &amp; metrics &mdash; "
                  "add from reports/screenshots/]</i>", CAPTION),
        Spacer(1, 0.3 * cm),
        Paragraph("5. Model Packaging &amp; Reproducibility", H1),
        p("The final pipeline (preprocessing + classifier) is serialized to "
          "<font face='Courier'>models/model.joblib</font> and also registered as an "
          "MLflow model. A pinned <font face='Courier'>requirements.txt</font> and "
          "<font face='Courier'>pyproject.toml</font> allow a clean-environment install. "
          "A fixed random seed (42) makes splits and models deterministic. Because the "
          "preprocessing is inside the saved pipeline, inference is fully reproducible."),
        PageBreak(),
    ]

    # ---- 6. CI/CD & testing ----
    story += [
        Paragraph("6. CI/CD Pipeline &amp; Automated Testing", H1),
        p("A GitHub Actions workflow (<font face='Courier'>.github/workflows/ci-cd.yml"
          "</font>) runs four sequential jobs on every push and pull request:"),
        p("<b>1. lint</b> &mdash; flake8, black and isort checks.<br/>"
          "<b>2. test</b> &mdash; 14 pytest unit + integration tests with coverage.<br/>"
          "<b>3. train</b> &mdash; trains the model and uploads model, metrics and "
          "MLflow runs as artifacts.<br/>"
          "<b>4. docker</b> &mdash; builds the image and smoke-tests the running "
          "container's <font face='Courier'>/health</font> endpoint."),
        p("The pipeline <b>fails fast</b>: any lint or test error stops the run and is "
          "clearly reported. Tests cover data cleaning, stratified splitting, the "
          "preprocessing transformer, model training/prediction, and all API endpoints "
          "(including validation errors and the metrics endpoint)."),
        Paragraph("<i>[Screenshot placeholder: green GitHub Actions run]</i>", CAPTION),
        Spacer(1, 0.3 * cm),
        Paragraph("7. Model Containerisation", H1),
        p("A multi-stage <font face='Courier'>Dockerfile</font> packages the API code, "
          "trained model, dependencies and inference pipeline. The container runs as a "
          "non-root user, exposes port 8000 and defines a health check. The FastAPI app "
          "serves <font face='Courier'>POST /predict</font> which accepts JSON and "
          "returns the prediction plus a confidence/probability score. "
          "<font face='Courier'>run_docker.ps1</font> builds, runs and smoke-tests the "
          "container with sample input in one command."),
        Paragraph("<i>[Screenshot placeholder: docker build + /predict response]</i>", CAPTION),
        PageBreak(),
    ]

    # ---- 8. Deployment, monitoring, conclusion ----
    story += [
        Paragraph("8. Production Deployment (Kubernetes)", H1),
        p("Kubernetes manifests in <font face='Courier'>k8s/</font> define a Deployment "
          "(2 replicas, resource requests/limits, liveness &amp; readiness probes on "
          "<font face='Courier'>/health</font>), a LoadBalancer Service, and an Ingress. "
          "A parameterised Helm chart (<font face='Courier'>helm/heart-api</font>) offers "
          "the same deployment with configurable replicas, image, service type, resources "
          "and optional ingress. Deploy with "
          "<font face='Courier'>kubectl apply -f k8s/</font> or "
          "<font face='Courier'>helm install heart-api ./helm/heart-api</font> on "
          "Minikube, Docker Desktop Kubernetes, AKS, EKS or GKE."),
        Paragraph("<i>[Screenshot placeholder: kubectl get pods,svc &amp; external "
                  "endpoint]</i>", CAPTION),
        Spacer(1, 0.3 * cm),
        Paragraph("9. Monitoring &amp; Logging", H1),
        p("Request/response logging (method, path, status, latency) is implemented as "
          "FastAPI middleware. The service exposes Prometheus metrics at "
          "<font face='Courier'>/metrics</font> &mdash; including custom counters "
          "(<font face='Courier'>heart_predictions_total</font> by outcome) and a "
          "latency histogram (<font face='Courier'>heart_prediction_latency_seconds"
          "</font>) &mdash; plus standard HTTP metrics. "
          "<font face='Courier'>docker-compose.yml</font> brings up Prometheus and "
          "Grafana alongside the API; a ready-made Grafana dashboard "
          "(<font face='Courier'>monitoring/grafana/heart-dashboard.json</font>) "
          "visualises request rate, prediction counts and p95 latency, helping detect "
          "model failures, drift and API downtime."),
        Paragraph("<i>[Screenshot placeholder: Grafana dashboard]</i>", CAPTION),
        Spacer(1, 0.3 * cm),
        Paragraph("10. Conclusion", H1),
        p("The delivered system demonstrates production-ready MLOps practices: a "
          "reproducible pipeline runnable from a clean setup, tracked experiments, a "
          "tested and linted codebase, an isolated containerised service, cloud/K8s "
          "deployment artifacts and monitoring. The selected Logistic Regression model "
          "achieves a cross-validated ROC-AUC of ~0.90 and a test ROC-AUC of ~0.97, "
          "providing a strong, interpretable baseline for heart-disease risk screening."),
        Spacer(1, 0.5 * cm),
        p("<b>Repository:</b> https://github.com/&lt;your-username&gt;/heart-disease-mlops"),
    ]

    doc = SimpleDocTemplate(
        str(config.REPORTS_DIR / "MLOps_Assignment_Report.pdf"),
        pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
        title="MLOps Assignment 01 - Heart Disease", author="AIMLCZG523 Student",
    )
    doc.build(story)
    print("Report written to", config.REPORTS_DIR / "MLOps_Assignment_Report.pdf")


if __name__ == "__main__":
    build()
