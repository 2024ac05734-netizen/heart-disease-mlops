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
| `verification-log.txt` | Combined Docker + Compose + Kubernetes verification output | Tasks 6–8 |

## Still to capture manually

- `github_actions.png` — a green CI/CD pipeline run (lint → test → train → docker).
  **Requires the repo pushed to GitHub first** (the workflow `.github/workflows/ci-cd.yml`
  runs automatically on push). Capture from the repo's **Actions** tab.

## How these were regenerated

- MLflow UI: `mlflow ui --backend-store-uri ./mlruns --port 5000` (project root).
- Monitoring: `docker compose up -d` (Grafana http://localhost:3000 admin/admin,
  Prometheus http://localhost:9090); dashboard `monitoring/grafana/heart-dashboard.json`.
- Kubernetes: `kubectl apply -f k8s/` (Docker Desktop Kubernetes), service on localhost:80.
- Capture scripts: `reports/capture_screenshots.py` and `reports/render_terminal.py`.
