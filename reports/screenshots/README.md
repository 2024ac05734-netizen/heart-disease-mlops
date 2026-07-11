# Screenshots for the report

These screenshots were captured live from the running system on this machine
(Docker Desktop engine, Kubernetes cluster, MLflow UI, Prometheus + Grafana).
They are ready to embed in the final report.

## Captured (auto-generated, live evidence)

| File | Shows | Assignment task |
| --- | --- | --- |
| `01_mlflow_runs.png` | MLflow experiment `heart-disease-classification` with all runs (LR / RF / XGBoost) | Task 3 — Experiment tracking |
| `02_mlflow_run_detail.png` | Best run `best_logistic_regression`: params + 7 metrics (accuracy 0.885, CV ROC-AUC 0.902, F1 0.881, precision 0.839, recall 0.929) | Task 3 — Experiment tracking |
| `03_swagger_deployed_api.png` | FastAPI Swagger `/docs` of the **Kubernetes-deployed** API (`/health`, `/predict`, `/predict/batch`, `/metrics`) | Task 6 / 7 |
| `04_prometheus_targets.png` | Prometheus scrape target `heart-api` — **1/1 UP** | Task 8 — Monitoring |
| `05_grafana_dashboard.png` | Grafana "Heart Disease API Monitoring" dashboard with live data (24 predictions, outcome/request-rate/latency panels) | Task 8 — Monitoring |
| `06_kubernetes_deployment.png` | `kubectl get nodes/deployment/pods/svc` — 2/2 pods Running, LoadBalancer service; live `/predict` returns "Heart Disease" 0.91 | Task 7 — Deployment |
| `07_github_actions.png` | GitHub **Actions** tab — CI/CD Pipeline run **green ✓** (lint → test → train → docker) on `2024ac05734-netizen/heart-disease-mlops` | Task 5 — CI/CD |
| `verification-log.txt` | Combined Docker + Compose + Kubernetes verification output | Tasks 6–8 |

## How these were regenerated

- MLflow UI: `mlflow ui --backend-store-uri ./mlruns --port 5000` (project root).
- Monitoring: `docker compose up -d` (Grafana http://localhost:3000 admin/admin,
  Prometheus http://localhost:9090); dashboard `monitoring/grafana/heart-dashboard.json`.
- Kubernetes: `kubectl apply -f k8s/` (Docker Desktop Kubernetes), service on localhost:80.
- Capture scripts: `reports/capture_screenshots.py`, `reports/render_terminal.py`
  and `reports/capture_actions.py` (public Actions page).
