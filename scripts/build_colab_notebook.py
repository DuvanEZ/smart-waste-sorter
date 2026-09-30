"""Generate notebooks/Smart_Waste_Sorter_Colab.ipynb from the source package.

The notebook is self-contained: the first cells write the `waste_classifier`
package (copied from src/) to the Colab disk, so it runs without cloning the
(private) GitHub repository. Re-run this script whenever src/ changes:
    python scripts/build_colab_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "waste_classifier"
OUT = ROOT / "notebooks" / "Smart_Waste_Sorter_Colab.ipynb"
MODULES = ["__init__", "config", "utils", "data", "image_analysis", "features", "models", "train",
           "evaluate", "explain", "export", "plots", "pipeline"]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Smart Waste Sorter - recyclable waste image classification
**INFO813 Artificial Intelligence - Project (Trimester 3, 2026)**

This notebook runs the complete project pipeline, in the same order as the report:

| Notebook section | Report section |
|---|---|
| 1. Data acquisition and description | 1.1 Description and explanation |
| 2. Visualisation | 1.2 Visualisation |
| 3. Data transformation | 2.2 Data transformation |
| 4. Baseline models | 2.1 Justification of the modelling technique |
| 5. Training EfficientNet-B0 | 2.3 Implementation |
| 6. Evaluation | 2.4 Evaluation |
| 7. Export for the application | 3 Software implementation |

**How to run:** `Runtime > Change runtime type > T4 GPU`, then `Runtime > Run all`.
The first run takes about 20-30 minutes on a T4 GPU. When Google asks for permission to
access Google Drive, accept it (the dataset zip is read from Drive and the results are saved there).
At the end `results.zip` (figures, metrics, ONNX model, sample images) is saved to Drive and downloaded.
"""))

cells.append(md("## 0. Environment"))
cells.append(code("""# GPU check and extra libraries (PyTorch, scikit-learn, OpenCV and pandas are pre-installed in Colab)
!nvidia-smi -L || echo "No GPU found - use Runtime > Change runtime type > T4 GPU (the CPU also works, but slowly)"
%pip install -q timm imagehash onnx onnxruntime onnxscript kagglehub"""))

cells.append(md("""### Project source code
The next cells write the `waste_classifier` Python package (identical to `src/waste_classifier`
in the GitHub repository). Each module is documented; the analysis cells further down only call
these functions."""))
cells.append(code("!mkdir -p waste_classifier"))
for name in MODULES:
    source = (PKG / f"{name}.py").read_text(encoding="utf-8")
    cells.append(code(f"%%writefile waste_classifier/{name}.py\n{source}"))

cells.append(md("## Settings"))
cells.append(code("""import sys, os, glob, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown

sys.path.insert(0, "/content")
from waste_classifier import pipeline as pl, plots
from waste_classifier.config import CLASS_NAMES, TrainConfig, BaselineConfig, SplitConfig
from waste_classifier.utils import set_seed, get_device, environment_info

# ---- paths (change DRIVE_ZIP if the archive is somewhere else in your Google Drive) ----
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_ZIP = f"{DRIVE_ROOT}/Auckland Institud/Artificial Intelligence/archive.zip"
DRIVE_OUT = f"{DRIVE_ROOT}/Auckland Institud/Artificial Intelligence/smart_waste_sorter_outputs"
OUT_DIR = "/content/outputs"

train_cfg, base_cfg, split_cfg = TrainConfig(), BaselineConfig(), SplitConfig()
WORKERS = os.cpu_count() or 2
set_seed(train_cfg.seed)
device = get_device()
if device.type != "cuda":      # no GPU: smaller batches to stay within the machine's RAM
    train_cfg.batch_size, base_cfg.cnn_batch_size = 32, 32
ws = pl.Workspace(OUT_DIR)

def show(name, width=950):
    \"\"\"Display a saved figure inline.\"\"\"
    display(Image(filename=str(ws.fig(name)), width=width))

def sync_to_drive():
    \"\"\"Copy the outputs to Google Drive (so nothing is lost if the Colab session ends).\"\"\"
    if os.path.isdir(DRIVE_ROOT):
        shutil.copytree(OUT_DIR, DRIVE_OUT, dirs_exist_ok=True)
        print("Outputs copied to", DRIVE_OUT)

print("Device:", device)
print(json.dumps(environment_info(), indent=1))"""))

cells.append(md("""## 1. Data acquisition and description (report section 1.1)
The Kaggle archive `archive.zip` is read from Google Drive and extracted to the fast local disk
of the Colab machine. If it is not found, it is downloaded directly from Kaggle with `kagglehub`."""))
cells.append(code("""# Mount Google Drive (accept the permission pop-up). If it is declined or not answered
# within 2 minutes, the notebook continues and downloads the dataset from Kaggle instead.
try:
    from google.colab import drive
    drive.mount("/content/drive", timeout_ms=120_000)
except Exception as exc:
    print(f"Google Drive not mounted ({type(exc).__name__}) - using Kaggle instead.")
DRIVE_OK = os.path.isdir(DRIVE_ROOT)

zip_path = DRIVE_ZIP if DRIVE_OK and os.path.exists(DRIVE_ZIP) else None
if DRIVE_OK and zip_path is None:  # search the Drive for the archive (max. 3 folder levels deep)
    hits = [p for d in range(4) for p in glob.glob(DRIVE_ROOT + "/*" * d + "/archive.zip")]
    zip_path = hits[0] if hits else None
print("Archive:", zip_path or "not on Drive -> kagglehub download")
if zip_path:
    shutil.copy(zip_path, "/content/archive.zip")
    zip_path = "/content/archive.zip"
root = pl.acquire_data(zip_path=zip_path, extract_to="/content/data", use_kagglehub=True)
print("Dataset root:", root)"""))
cells.append(code("""ctx = pl.data_understanding(root, ws, split_cfg, workers=WORKERS)
s = ctx.summary
overview = pd.DataFrame({
    "Property": ["Total images", "Readable images", "Classes", "Resolution(s)", "File format(s)", "Colour mode(s)",
                 "Total size", "Median file size", "Exact duplicate files", "Images with a near-duplicate (pHash <= 4)",
                 "Near-duplicate pairs with different labels", "Largest / smallest class ratio"],
    "Value": [f"{s['n_images']:,}", f"{s['n_readable']:,}", s["n_classes"], s["resolutions"], s["formats"], s["modes"],
              f"{s['total_size_mb']:.1f} MB", f"{s['file_size_kb']['50%']:.1f} KB", s["exact_duplicate_files"],
              s["images_with_near_duplicate"]["<=4"], s["cross_class_pairs"]["<=4"], f"{s['imbalance_ratio_max_min']:.2f}"],
})
display(overview)
display(pd.Series(s["class_counts"], name="images").to_frame().assign(share=lambda d: (d.images / d.images.sum()).round(3)))
print("metadata.csv check:", json.dumps(s.get("metadata_check", {}), indent=1)[:1500])"""))
cells.append(code("""# Descriptive statistics of the numeric image variables (per class)
display(pd.read_csv(ws.metrics_dir / "image_statistics_overall.csv", index_col=0))
display(ctx.stats.groupby("class_name")[["brightness", "contrast", "saturation", "colourfulness", "sharpness",
                                          "entropy", "edge_density", "white_background", "file_size_kb"]].median().round(3))"""))

cells.append(md("""## 2. Visualisation (report section 1.2)
Every image variable is visualised (class label, resolution, format, colour mode, file size,
brightness, contrast, saturation, colourfulness, sharpness, entropy, edge density, background,
colour channels and hue), followed by plots of the relationships between variables."""))
cells.append(code("""figs = pl.visualise(ctx, ws)
for f in figs:
    display(Markdown(f"**{f.name}**"))
    display(Image(filename=str(f), width=950))"""))
cells.append(md("""### Deep-feature map (t-SNE)
Each image is converted into a 1,280-number description by an ImageNet-pre-trained EfficientNet-B0,
and t-SNE projects these descriptions to 2-D. Clusters show which classes are visually distinct
and which overlap - a preview of the difficulty of the classification problem."""))
cells.append(code("""pl.deep_embeddings(ctx, ws, device, workers=WORKERS)
pl.tsne_map(ctx, ws)
show("fig12_tsne_embeddings.png")"""))

cells.append(md("""## 3. Data transformation (report section 2.2)
* Unreadable files, exact duplicates and files with conflicting labels are removed.
* A **stratified group split** (about 70 / 15 / 15 %) keeps the class proportions in each subset and
  puts near-duplicate images in the same subset, so the test set contains no copies of training images.
* Images are resized / centre-cropped to 224 x 224, scaled to [0, 1] and normalised with the ImageNet
  mean and standard deviation; training images are also randomly augmented."""))
cells.append(code("""data = pl.split_data(ctx, ws, split_cfg)
show("fig13_split_distribution.png", 800)
show("fig14_augmentation_examples.png", 800)"""))

cells.append(md("""## 4. Baseline models (report sections 2.1 and 2.4)
To justify the choice of model, simpler alternatives are trained on exactly the same split:
classical machine learning on hand-crafted features (k-NN, logistic regression, random forest, SVM),
a linear classifier on frozen ImageNet features, and a small CNN trained from scratch."""))
cells.append(code("""baseline_results = pl.baselines(ctx, ws, device, base_cfg, run_cnn=True, workers=WORKERS)
display(baseline_results.round(4))"""))

cells.append(md("""## 5. Training the final model: EfficientNet-B0 with transfer learning (report section 2.3)
Phase 1 trains only the new 6-class output layer on top of the frozen ImageNet backbone; phase 2
fine-tunes all layers with a smaller learning rate (AdamW, cosine schedule, label smoothing,
mixed precision on the GPU). Early stopping keeps the epoch with the best validation macro-F1."""))
cells.append(code("""print(train_cfg)
model, history = pl.train_final(ctx, ws, device, train_cfg, workers=WORKERS)
display(history.round(4))
show("fig15_learning_curves.png")
sync_to_drive()"""))

cells.append(md("""## 6. Evaluation on the held-out test set (report section 2.4)
Accuracy, precision, recall, F1 (per class and macro-averaged), confusion matrix, ROC and
precision-recall curves, 95% bootstrap confidence intervals, probability calibration
(temperature scaling), selective prediction, error analysis, Grad-CAM and robustness tests."""))
cells.append(code("""metrics = pl.evaluate_final(ctx, ws, device, train_cfg, workers=WORKERS)
keys = ["accuracy", "accuracy_ci", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_f1_ci",
        "top2_accuracy", "roc_auc_ovr_macro", "cohen_kappa", "mcc", "log_loss", "ece_before_temperature", "ece",
        "temperature", "accuracy_above_threshold", "coverage_above_threshold"]
display(pd.Series({k: metrics[k] for k in keys}, name="test set").to_frame())
display(pd.read_csv(ws.metrics_dir / "per_class_metrics.csv").round(4))
display(pd.read_csv(ws.metrics_dir / "most_confused_pairs.csv").round(4))
display(pd.read_csv(ws.metrics_dir / "model_comparison.csv").round(4))
for name in ["fig16_confusion_matrix.png", "fig17_per_class_metrics.png", "fig18_roc_pr_curves.png",
             "fig19_reliability_diagram.png", "fig20_confidence_analysis.png", "fig21_misclassified_examples.png",
             "fig22_gradcam.png", "fig23_robustness.png", "fig24_model_comparison.png"]:
    display(Markdown(f"**{name}**")); show(name)
sync_to_drive()"""))

cells.append(md("""## 7. Export for the application (report section 3)
The model is exported to ONNX (weights stored in float16 to halve the file size) so the
application only needs `onnxruntime`, not PyTorch. The ONNX outputs are checked against PyTorch."""))
cells.append(code("""info = pl.export_model(ctx, ws, train_cfg)
print(json.dumps({k: info[k] for k in ["onnx_file_mb", "cpu_latency_ms", "parity", "temperature", "metrics"]}, indent=1))
print(sorted(p.name for p in ws.sample_dir.iterdir()))"""))

cells.append(md("## 8. Save and download the results"))
cells.append(code("""zip_file = pl.package_results(ctx, ws, zip_path="/content/results.zip")
sync_to_drive()
if os.path.isdir(DRIVE_ROOT):
    shutil.copy(zip_file, DRIVE_OUT + "/results.zip")
    print("results.zip saved in", DRIVE_OUT)
from google.colab import files
files.download(str(zip_file))"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells)")
