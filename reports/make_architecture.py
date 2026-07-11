"""Render a simple architecture diagram to reports/figures/architecture.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from heart_mlops import config  # noqa: E402


def box(ax, x, y, w, h, text, color):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="black", lw=1.2
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Heart Disease MLOps - End-to-End Architecture", fontsize=13, weight="bold")

    blue, green, orange, purple, gray = "#cfe2ff", "#d1e7dd", "#ffe5b4", "#e2d4f0", "#e9ecef"

    box(ax, 0.3, 5.2, 2.2, 1.0, "UCI Heart Disease\nDataset\n(download script)", blue)
    box(ax, 3.0, 5.2, 2.2, 1.0, "EDA + Preprocessing\n(Pipeline /\nColumnTransformer)", blue)
    box(ax, 5.7, 5.2, 2.4, 1.0, "Model Training\nLR / RF / XGBoost\nGridSearchCV + CV", green)
    box(ax, 8.7, 5.2, 2.2, 1.0, "MLflow Tracking\nparams / metrics /\nartifacts / models", purple)

    box(ax, 5.7, 3.6, 2.4, 1.0, "Model Registry\nmodel.joblib\n(sklearn Pipeline)", green)
    box(ax, 8.7, 3.6, 2.2, 1.0, "Unit Tests\n(pytest)", gray)
    box(ax, 3.0, 3.6, 2.2, 1.0, "GitHub Actions\nlint -> test ->\ntrain -> docker", orange)

    box(ax, 5.7, 2.0, 2.4, 1.0, "FastAPI Service\n/predict /health\n/metrics", blue)
    box(ax, 8.7, 2.0, 2.2, 1.0, "Docker Image", orange)
    box(ax, 3.0, 2.0, 2.2, 1.0, "Kubernetes\nDeployment +\nService + Helm", orange)

    box(ax, 5.7, 0.4, 2.4, 1.0, "Client\n(curl / Postman /\nSwagger UI)", gray)
    box(ax, 8.7, 0.4, 2.2, 1.0, "Prometheus +\nGrafana\n(monitoring)", purple)

    arrow(ax, 2.5, 5.7, 3.0, 5.7)
    arrow(ax, 5.2, 5.7, 5.7, 5.7)
    arrow(ax, 8.1, 5.7, 8.7, 5.7)
    arrow(ax, 6.9, 5.2, 6.9, 4.6)      # training -> registry
    arrow(ax, 6.9, 3.6, 6.9, 3.0)      # registry -> api
    arrow(ax, 5.7, 4.1, 5.2, 4.1)      # registry <- CI (train)
    arrow(ax, 8.7, 4.1, 8.1, 4.1)      # tests -> registry area
    arrow(ax, 8.1, 2.5, 8.7, 2.5)      # api -> docker
    arrow(ax, 5.7, 2.5, 5.2, 2.5)      # api -> k8s
    arrow(ax, 6.9, 2.0, 6.9, 1.4)      # api -> client
    arrow(ax, 8.1, 2.2, 8.7, 1.2)      # api -> monitoring

    fig.tight_layout()
    out = config.FIGURES_DIR / "architecture.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
