"""Run the provenance check in Colab after the main pipeline (uses /content/outputs).

    %run provenance_run.py
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/content")
from waste_classifier import plots
from waste_classifier.provenance import check_provenance

OUT = Path("/content/outputs")
stats = pd.read_pickle(OUT / "cache" / "image_stats.pkl")
result = check_provenance(stats, OUT / "metrics")
summary = pd.read_csv(OUT / "metrics" / "provenance_summary.csv", index_col=0)
print(summary)
plots.provenance_bars(summary, OUT / "figures" / "fig25_provenance.png")
with zipfile.ZipFile("/content/provenance_results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["metrics/provenance_summary.csv", "metrics/provenance_per_image.csv", "figures/fig25_provenance.png"]:
        zf.write(OUT / f, f)
print("saved /content/provenance_results.zip")
