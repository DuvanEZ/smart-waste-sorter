"""Capture screenshots of the running Streamlit app for the report and the slides.

Usage (with the app already running on --url):
    python scripts/capture_app_screenshots.py --url http://localhost:8501 --demo-dir demo_photos --out reports/figures/app
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def wait_idle(page, extra_ms: int = 800) -> None:
    """Wait until Streamlit has finished re-running the script."""
    page.wait_for_timeout(300)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]') || "
        "!document.querySelector('[data-testid=\"stStatusWidget\"]').innerText.includes('Running')",
        timeout=60_000)
    page.wait_for_timeout(extra_ms)


def fit_and_shoot(page, path: Path, min_height: int = 1000, max_height: int = 2600) -> None:
    """Streamlit scrolls inside its own container, so enlarge the viewport to the content height first."""
    height = page.evaluate(
        "() => { const c = document.querySelector('[data-testid=\"stMainBlockContainer\"]');"
        " return c ? Math.ceil(c.getBoundingClientRect().top + c.scrollHeight) : 1000; }")
    page.set_viewport_size({"width": 1440, "height": int(min(max(height + 40, min_height), max_height))})
    page.wait_for_timeout(1200)
    page.screenshot(path=str(path))
    page.set_viewport_size({"width": 1440, "height": 1000})
    trim_bottom(path)


def trim_bottom(path: Path, margin: int = 60) -> None:
    """Remove the empty white band below the last element of the main area."""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    main = arr[:, int(arr.shape[1] * 0.25):, :]            # skip the grey sidebar
    rows = np.where((main < 245).any(axis=(1, 2)))[0]
    if len(rows):
        bottom = min(arr.shape[0], int(rows[-1]) + margin)
        img.crop((0, 0, arr.shape[1], bottom)).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--demo-dir", default=str(ROOT.parent / "demo_photos_v2"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "figures" / "app"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    demo = Path(args.demo_dir)
    samples = ROOT / "sample_images"

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
        page.goto(args.url)
        page.get_by_text("Smart Waste Sorter").first.wait_for(timeout=60_000)
        wait_idle(page, 1500)

        # 1) single image, confident prediction (file upload)
        page.locator('[data-testid="stFileUploader"] input[type="file"]').first.set_input_files(
            str(samples / "plastic_02.jpg"))
        page.get_by_text("Where does it go?").first.wait_for(timeout=60_000)
        wait_idle(page, 1500)
        fit_and_shoot(page, out / "app_single.png")

        # 2) uncertain prediction (hard case: soup cartons labelled cardboard)
        page.locator('[data-testid="stFileUploader"] input[type="file"]').first.set_input_files(
            str(samples / "hard_case_01_true-trash.jpg"))
        page.get_by_text("UNCERTAIN").first.wait_for(timeout=60_000)
        wait_idle(page, 1500)
        fit_and_shoot(page, out / "app_uncertain.png")

        # 3) invalid input: corrupted file
        page.locator('[data-testid="stFileUploader"] input[type="file"]').first.set_input_files(
            str(demo / "broken_photo.jpg"))
        page.get_by_text("not a valid or readable image").first.wait_for(timeout=60_000)
        wait_idle(page, 1000)
        fit_and_shoot(page, out / "app_invalid.png", min_height=800)

        # 4) batch: classify a folder (valid + invalid files)
        page.get_by_role("tab", name="Batch classification").click()
        wait_idle(page)
        page.get_by_text("Classify a folder on this computer").click()
        wait_idle(page)
        box = page.get_by_label("Folder path", exact=True).and_(page.locator("input"))
        box.fill(str(demo))
        box.press("Enter")
        wait_idle(page)
        page.get_by_role("button", name="Classify folder").click()
        page.get_by_text("Download results as CSV").first.wait_for(timeout=60_000)
        wait_idle(page, 2500)
        fit_and_shoot(page, out / "app_batch.png")

        # 5) model performance tab
        page.get_by_role("tab", name="Model performance").click()
        wait_idle(page, 1500)
        fit_and_shoot(page, out / "app_model.png")

        # 6) help tab
        page.get_by_role("tab", name="Help").click()
        wait_idle(page, 1000)
        fit_and_shoot(page, out / "app_help.png")
        browser.close()
    print("Screenshots saved to", out)


if __name__ == "__main__":
    main()
