"""Create the two zip archives that are embedded in the report as appendices.

    python scripts/package_submission.py --results <extracted results.zip folder> --out <folder>

* smart-waste-sorter-code.zip            (Appendix A) - the complete project: application, trained
  model, sample images, analysis/training package, notebook, scripts, tests and documentation.
* smart-waste-sorter-data-and-model.zip  (Appendix B) - the data and model files produced by the
  notebook: model files, all metric tables (incl. per-image statistics and the split manifest) and the
  sample images. (The figures are already in the report.)
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ITEMS = [".streamlit", "app", "model", "sample_images", "src", "notebooks", "scripts", "tests", "docs",
              "data/README.md", "reports/report_builder", "reports/slides_builder", "reports/figures/fig00_pipeline.png",
              "reports/figures/app", "README.md", "requirements.txt", "requirements-train.txt", "run_app.bat",
              "run_app.sh", ".gitignore"]
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".venv", "node_modules", ".git", ".ipynb_checkpoints"}
EXCLUDE_SUFFIX = {".pt", ".pth", ".pyc"}
EXCLUDE_NAMES = {"package-lock.json", "app_env.json"}

DATA_README = """Smart Waste Sorter - data and model files (Appendix B)
=====================================================

model/        waste_classifier.onnx (EfficientNet-B0, float16 weights) and model_info.json
              (class names, input size, normalisation, calibration temperature, test metrics,
              environment), confusion_matrix.png
metrics/      dataset_summary.json        counts, formats, duplicates, balance
              image_stats.csv             per-image variables (size, brightness, colour, texture,
                                          MD5, perceptual hash distance, duplicate group)
              splits.csv                  train / val / test assignment of every image used
              provenance_*.csv            matches with public source data sets
              model_comparison.csv        baselines vs final model
              history_*.csv               training curves
              test_predictions.csv        probabilities for every test image
              per_class_metrics.csv, confusion_matrix.csv, most_confused_pairs.csv,
              selective_prediction.csv, robustness.csv, test_metrics.json, run_info.json
sample_images/  test images used by the application's "sample image" option

The original images are not included (13,901 JPEG files, about 121 MB). They are available under
the MIT licence from Kaggle: https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification
(Cofone, 2025). metrics/splits.csv lists exactly which files were used in each subset.
"""


def add_tree(zf: zipfile.ZipFile, path: Path, arc_root: str, base: Path) -> int:
    n = 0
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        rel = f.relative_to(base)
        if set(rel.parts) & EXCLUDE_DIRS or f.suffix in EXCLUDE_SUFFIX or f.name in EXCLUDE_NAMES:
            continue
        zf.write(f, f"{arc_root}/{rel.as_posix()}")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results)

    code_zip = out / "smart-waste-sorter-code.zip"
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        n = sum(add_tree(zf, ROOT / item, "smart-waste-sorter", ROOT) for item in CODE_ITEMS if (ROOT / item).exists())
    print(f"{code_zip.name}: {n} files, {code_zip.stat().st_size / 1e6:.1f} MB")

    data_zip = out / "smart-waste-sorter-data-and-model.zip"
    with zipfile.ZipFile(data_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        n = 0
        # The figures are not repeated here: they are all in the report itself and in the GitHub repository.
        for sub in ["model", "metrics", "sample_images"]:
            if (results / sub).exists():
                n += add_tree(zf, results / sub, "smart-waste-sorter-data-and-model", results)
        zf.writestr("smart-waste-sorter-data-and-model/README.txt", DATA_README)
    print(f"{data_zip.name}: {n + 1} files, {data_zip.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
