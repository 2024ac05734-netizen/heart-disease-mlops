"""Render captured terminal text to a styled PNG (Kubernetes deployment evidence)."""
import html
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(r"D:\M Tech\MLops\Assignment\heart-disease-mlops\reports\screenshots")
text = Path(os.environ["TEMP"], "kubectl_out.txt").read_text(encoding="utf-8")

page_html = f"""<!doctype html><html><head><meta charset='utf-8'><style>
body{{margin:0;background:#0c0c0c;}}
.term{{font-family:'Cascadia Code','Consolas',monospace;font-size:15px;line-height:1.5;
  color:#e6e6e6;padding:24px 28px;white-space:pre-wrap;}}
.hdr{{background:#1f1f1f;color:#ccc;padding:8px 16px;font-family:Segoe UI,sans-serif;font-size:13px;
  border-bottom:1px solid #333;}}
.dot{{height:12px;width:12px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle;}}
.up{{color:#4ec94e;}} .prompt{{color:#3b8eea;}}
</style></head><body>
<div class='hdr'><span class='dot' style='background:#ff5f56'></span>
<span class='dot' style='background:#ffbd2e'></span>
<span class='dot' style='background:#27c93f'></span>
&nbsp;Windows PowerShell — Heart Disease MLOps — Kubernetes (Docker Desktop)</div>
<div class='term'>{html.escape(text)}</div>
</body></html>"""

tmp = OUT / "_kubectl.html"
tmp.write_text(page_html, encoding="utf-8")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context(viewport={"width": 1200, "height": 700}).new_page()
    pg.goto(tmp.as_uri())
    pg.wait_for_timeout(600)
    pg.screenshot(path=str(OUT / "06_kubernetes_deployment.png"), full_page=True)
    b.close()
tmp.unlink()
print("OK 06_kubernetes_deployment.png")
