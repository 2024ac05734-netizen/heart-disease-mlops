"""Capture live UI screenshots for the assignment report using Playwright."""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(r"D:\M Tech\MLops\Assignment\heart-disease-mlops\reports\screenshots")
OUT.mkdir(parents=True, exist_ok=True)

GRAFANA_UID = "aftkgp"


def shot(page, url, path, wait=3500, selector=None, full=False):
    try:
        page.goto(url, wait_until="load", timeout=45000)
        if selector:
            try:
                page.wait_for_selector(selector, timeout=15000)
            except Exception:
                pass
        page.wait_for_timeout(wait)
        page.screenshot(path=str(OUT / path), full_page=full)
        print(f"OK   {path}  <- {url}")
    except Exception as e:
        print(f"FAIL {path}  <- {url} : {e}")


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1600, "height": 900})
    page = ctx.new_page()

    # 1. MLflow experiment / runs table
    shot(page, "http://127.0.0.1:5000/#/experiments/160464375305541669",
         "01_mlflow_runs.png", selector="table", wait=5000)

    # 2. MLflow models comparison (metrics columns visible)
    shot(page, "http://127.0.0.1:5000/#/experiments/160464375305541669",
         "02_mlflow_experiment.png", wait=4000, full=True)

    # 3. Swagger UI of the Kubernetes-deployed API (port 80)
    shot(page, "http://localhost:80/docs", "03_swagger_deployed_api.png",
         selector="#swagger-ui", wait=4000)

    # 4. Prometheus targets page (heart-api scrape target UP)
    shot(page, "http://localhost:9090/targets", "04_prometheus_targets.png",
         wait=4000)

    # 5. Grafana login then dashboard
    try:
        page.goto("http://localhost:3000/login", wait_until="load", timeout=45000)
        page.wait_for_timeout(2000)
        page.fill("input[name='user']", "admin")
        page.fill("input[name='password']", "admin")
        page.click("button[type='submit']")
        page.wait_for_timeout(4000)
    except Exception as e:
        print("grafana login note:", e)
    shot(page,
         f"http://localhost:3000/d/{GRAFANA_UID}/heart-disease-api-monitoring?from=now-30m&to=now&refresh=5s&kiosk",
         "05_grafana_dashboard.png", wait=7000)

    browser.close()
print("DONE")
