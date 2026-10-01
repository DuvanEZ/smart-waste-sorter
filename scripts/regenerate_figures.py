"""Re-draw the figures that can be rebuilt from the saved CSV/JSON outputs (no images,
no GPU and no model needed). Useful after a change to the plotting code.

    python scripts/regenerate_figures.py <results_dir>

<results_dir> is the extracted results.zip produced by the Colab notebook
(sub-folders figures/ and metrics/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waste_classifier import evaluate as ev  # noqa: E402
from waste_classifier import plots  # noqa: E402
from waste_classifier.config import CLASS_NAMES  # noqa: E402


def main(results_dir: str) -> None:
    rd = Path(results_dir)
    m, f = rd / "metrics", rd / "figures"
    done = []

    # Class distribution (from the dataset summary).
    summary_json = m / "dataset_summary.json"
    if summary_json.exists():
        counts = pd.Series(json.loads(summary_json.read_text())["class_counts"])
        done.append(plots.class_distribution(counts, f / "fig01_class_distribution.png"))

    # Unique images vs exact copies per class (needs only the MD5 checksums).
    stats_csv = m / "image_stats.csv"
    if stats_csv.exists():
        stats = pd.read_csv(stats_csv, usecols=["class_name", "md5", "readable"])
        stats = stats[stats["readable"].fillna(False).astype(bool)]
        done.append(plots.duplicates_by_class(stats, f / "fig01b_duplicates_by_class.png"))

    # Reliability diagram and confidence analysis from the saved test predictions.
    pred_csv, test_json = m / "test_predictions.csv", m / "test_metrics.json"
    if pred_csv.exists() and test_json.exists():
        test = json.loads(test_json.read_text())
        preds = pd.read_csv(pred_csv)
        prob = preds[[f"p_{c}" for c in CLASS_NAMES]].to_numpy(dtype=np.float64)
        y = preds["label"].to_numpy()
        T = float(test["temperature"])
        # The saved probabilities are softmax(z / T). softmax(T * log p) recovers softmax(z),
        # i.e. the probabilities before temperature scaling (the additive constant cancels).
        logp = np.log(np.clip(prob, 1e-12, 1.0))
        z = T * logp
        z -= z.max(axis=1, keepdims=True)
        prob_raw = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        done.append(plots.reliability_diagram(
            ev.reliability_table(prob_raw, y), ev.reliability_table(prob, y),
            test.get("ece_before_temperature", ev.expected_calibration_error(prob_raw, y)),
            test.get("ece", ev.expected_calibration_error(prob, y)), T, f / "fig19_reliability_diagram.png"))
        selective = pd.read_csv(m / "selective_prediction.csv") if (m / "selective_prediction.csv").exists() \
            else ev.selective_prediction(prob, y)
        threshold = float(test.get("app_threshold", 0.6))
        done.append(plots.confidence_analysis(prob, y, selective, threshold, f / "fig20_confidence_analysis.png"))

    # Provenance bars (if the provenance step was run).
    prov_csv = m / "provenance_summary.csv"
    if prov_csv.exists():
        from waste_classifier.provenance import OLD_SOURCE_NAMES

        summary = pd.read_csv(prov_csv, index_col=0).rename(columns=OLD_SOURCE_NAMES)
        done.append(plots.provenance_bars(summary, f / "fig25_provenance.png"))

    for path in done:
        print("updated", path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
