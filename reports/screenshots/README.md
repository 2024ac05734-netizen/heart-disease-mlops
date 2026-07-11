# Screenshots for the report

Place the following screenshots here and reference them in the report:

1. `mlflow_ui.png` — MLflow experiment runs with params and metrics comparison.
2. `mlflow_run_detail.png` — a single run's metrics, ROC/confusion artifacts.
3. `github_actions.png` — a green CI/CD pipeline run (lint → test → train → docker).
4. `docker_build.png` — successful `docker build` output.
5. `docker_predict.png` — `/predict` response from the running container (or Swagger UI).
6. `k8s_pods.png` — `kubectl get pods,svc` showing running pods and the service.
7. `grafana_dashboard.png` — the Heart Disease API Grafana dashboard.
8. `swagger_ui.png` — FastAPI Swagger UI at `/docs`.

To capture MLflow: run `mlflow ui` in the project root, open http://localhost:5000.
To capture Grafana: run `docker compose up`, open http://localhost:3000 (admin/admin),
import `monitoring/grafana/heart-dashboard.json`.
