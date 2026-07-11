"""Capture the public GitHub Actions page as evidence of a green CI/CD run."""

from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = "2024ac05734-netizen/heart-disease-mlops"
OUT = Path(__file__).resolve().parent / "screenshots" / "07_github_actions.png"


def main() -> None:
    url = f"https://github.com/{REPO}/actions"
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(url, wait_until="networkidle", timeout=60000)
        # Give the workflow-run list time to render its status icons.
        page.wait_for_timeout(3000)
        page.screenshot(path=str(OUT), full_page=False)
        browser.close()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
