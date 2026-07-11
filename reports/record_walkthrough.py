"""
Record an end-to-end pipeline walkthrough video (Playwright).
Tours: title -> Swagger (deployed API) -> live prediction -> MLflow runs ->
MLflow run detail -> Prometheus target -> Grafana dashboard -> Kubernetes -> outro.
Output: reports/pipeline_walkthrough.webm  (converted to .mp4 separately)
"""
import html
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(r"D:\M Tech\MLops\Assignment\heart-disease-mlops")
VID_DIR = ROOT / "reports" / "_video_raw"
VID_DIR.mkdir(parents=True, exist_ok=True)
TMP = ROOT / "reports" / "_video_tmp"
TMP.mkdir(parents=True, exist_ok=True)

EXP = "160464375305541669"
RUN = "609a03a084214bd2aea5604587eb38c4"
GRAF_UID = "aftkgp"
VW, VH = 1280, 720

CARD_CSS = """
<style>
  html,body{margin:0;height:100%;font-family:'Segoe UI',Arial,sans-serif;
    background:linear-gradient(135deg,#0b1e3f 0%,#123a6b 55%,#0b6e5f 100%);color:#fff;}
  .wrap{height:100%;display:flex;flex-direction:column;justify-content:center;
    align-items:center;text-align:center;padding:0 8%;box-sizing:border-box;}
  h1{font-size:52px;margin:0 0 12px;font-weight:700;}
  h2{font-size:30px;margin:0 0 26px;font-weight:400;color:#cfe4ff;}
  ul{font-size:24px;line-height:1.7;text-align:left;color:#eaf3ff;list-style:none;padding:0;}
  li:before{content:'\\2714  ';color:#4ee39a;}
  .tag{font-size:20px;color:#9fd0ff;margin-top:30px;}
  .step{position:fixed;top:0;left:0;right:0;background:rgba(8,20,40,.92);color:#fff;
    font-family:'Segoe UI',Arial,sans-serif;padding:12px 22px;font-size:20px;z-index:99999;
    border-bottom:2px solid #2f81f7;box-shadow:0 2px 10px rgba(0,0,0,.4);}
  .step b{color:#4ee39a;}
</style>
"""

def card(title, subtitle="", items=None, tag=""):
    li = "".join(f"<li>{html.escape(i)}</li>" for i in (items or []))
    body = f"<h1>{html.escape(title)}</h1>"
    if subtitle:
        body += f"<h2>{html.escape(subtitle)}</h2>"
    if li:
        body += f"<ul>{li}</ul>"
    if tag:
        body += f"<div class='tag'>{html.escape(tag)}</div>"
    p = TMP / (title.split()[0].lower() + str(abs(hash(title)) % 9999) + ".html")
    p.write_text(f"<!doctype html><html><head><meta charset='utf-8'>{CARD_CSS}</head>"
                 f"<body><div class='wrap'>{body}</div></body></html>", encoding="utf-8")
    return p.as_uri()

def banner(page, text):
    js = "(t)=>{let d=document.createElement('div');d.className='step';d.innerHTML=t;" \
         "document.body.appendChild(d);}"
    try:
        page.add_style_tag(content=".step{position:fixed;top:0;left:0;right:0;"
            "background:rgba(8,20,40,.92);color:#fff;font-family:Segoe UI,Arial,sans-serif;"
            "padding:12px 22px;font-size:20px;z-index:99999;border-bottom:2px solid #2f81f7;}"
            ".step b{color:#4ee39a;}")
        page.evaluate(js, text)
    except Exception as e:
        print("banner note:", e)

with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": VW, "height": VH},
                              record_video_dir=str(VID_DIR),
                              record_video_size={"width": VW, "height": VH})
    page = ctx.new_page()

    # live predictions via server-side request (no CORS issues)
    low = ctx.request.post("http://localhost:80/predict", data={
        "age":37,"sex":1,"cp":2,"trestbps":130,"chol":250,"fbs":0,"restecg":1,
        "thalach":187,"exang":0,"oldpeak":3.5,"slope":0,"ca":0,"thal":2}).json()
    high = ctx.request.post("http://localhost:80/predict", data={
        "age":67,"sex":1,"cp":0,"trestbps":160,"chol":286,"fbs":0,"restecg":0,
        "thalach":108,"exang":1,"oldpeak":1.5,"slope":1,"ca":3,"thal":2}).json()

    # 0. Title
    page.goto(card("Heart Disease MLOps",
                   "End-to-End Pipeline Walkthrough",
                   ["Data + EDA  \u2022  MLflow tracking",
                    "FastAPI  \u2022  Docker  \u2022  Kubernetes",
                    "Prometheus + Grafana monitoring"],
                   "AIMLCZG523 Assignment 01"))
    page.wait_for_timeout(5000)

    # 1. Swagger of deployed API
    page.goto("http://localhost:80/docs", wait_until="load", timeout=45000)
    page.wait_for_selector("#swagger-ui", timeout=15000)
    page.wait_for_timeout(1500)
    banner(page, "Step 1 &mdash; <b>FastAPI</b> served from the <b>Kubernetes</b> deployment (Swagger /docs)")
    page.wait_for_timeout(3500)
    try:
        page.click("text=/predict", timeout=4000)
        page.wait_for_timeout(2500)
    except Exception:
        pass
    page.wait_for_timeout(1500)

    # 2. Live prediction results card
    def fmt(d):
        return json.dumps(d)
    page.goto(card("Live /predict responses",
                   "Real calls to the deployed API (LoadBalancer :80)",
                   [f"Low-risk patient  ->  {low['label']}  (conf {low['confidence']})",
                    f"High-risk patient ->  {high['label']}  (conf {high['confidence']})"],
                   "Prediction + confidence returned as JSON"))
    page.wait_for_timeout(5500)

    # 3. MLflow runs
    page.goto(f"http://127.0.0.1:5000/#/experiments/{EXP}", wait_until="load", timeout=45000)
    page.wait_for_timeout(4000)
    banner(page, "Step 2 &mdash; <b>MLflow</b> experiment tracking: LR / RF / XGBoost runs, params &amp; metrics")
    page.wait_for_timeout(5000)

    # 4. MLflow run detail
    page.goto(f"http://127.0.0.1:5000/#/experiments/{EXP}/runs/{RUN}", wait_until="load", timeout=45000)
    page.wait_for_timeout(4000)
    banner(page, "Best run &mdash; <b>ROC-AUC 0.90</b>, accuracy 0.885, logged params + metrics + artifacts")
    page.wait_for_timeout(4500)

    # 5. Prometheus target
    page.goto("http://localhost:9090/targets", wait_until="load", timeout=45000)
    page.wait_for_timeout(3500)
    banner(page, "Step 3 &mdash; <b>Prometheus</b> scraping the API /metrics endpoint (target UP)")
    page.wait_for_timeout(4000)

    # 6. Grafana login + dashboard
    try:
        page.goto("http://localhost:3000/login", wait_until="load", timeout=45000)
        page.wait_for_timeout(1800)
        page.fill("input[name='user']", "admin")
        page.fill("input[name='password']", "Grafana#2026")
        page.click("button[type='submit']")
        page.wait_for_timeout(3500)
    except Exception as e:
        print("grafana login note:", e)
    page.goto(f"http://localhost:3000/d/{GRAF_UID}/heart-disease-api-monitoring?from=now-30m&to=now&refresh=5s&kiosk",
              wait_until="load", timeout=45000)
    page.wait_for_timeout(7000)

    # 7. Kubernetes terminal card (reuse captured kubectl text)
    kube = (ROOT / "reports" / "screenshots").parent  # placeholder
    import os
    kubetxt = Path(os.environ["TEMP"], "kubectl_out.txt").read_text(encoding="utf-8")
    term = TMP / "kube.html"
    term.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "body{margin:0;background:#0c0c0c;}"
        ".hdr{background:#1f1f1f;color:#ccc;padding:10px 18px;font-family:Segoe UI;font-size:16px;}"
        ".t{font-family:Consolas,monospace;font-size:15px;line-height:1.5;color:#e6e6e6;"
        "padding:18px 24px;white-space:pre-wrap;}</style></head><body>"
        "<div class='hdr'>Kubernetes deployment &mdash; Docker Desktop cluster</div>"
        f"<div class='t'>{html.escape(kubetxt)}</div></body></html>", encoding="utf-8")
    page.goto(term.as_uri())
    page.wait_for_timeout(6500)

    # 8. Outro
    page.goto(card("Pipeline complete",
                   "All 9 tasks demonstrated live",
                   ["EDA + 3 models tracked in MLflow",
                    "CI/CD (GitHub Actions) + 14 passing tests",
                    "Docker image -> Kubernetes (2 replicas, LoadBalancer)",
                    "Prometheus + Grafana monitoring"],
                   "Heart Disease UCI  \u2022  MLOps AIMLCZG523"))
    page.wait_for_timeout(4500)

    page.wait_for_timeout(500)
    ctx.close()  # finalizes the video
    video_path = page.video.path() if page.video else None
    browser.close()

print("RAW_VIDEO:", video_path)
