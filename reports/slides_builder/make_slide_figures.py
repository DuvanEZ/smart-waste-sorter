"""Slide-sized versions of a few report figures (larger text, fewer panels).

    python make_slide_figures.py <results_dir>

Writes <results_dir>/figures/slides/*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from waste_classifier import plots  # noqa: E402,F401  (applies the shared style)
from waste_classifier.config import CLASS_COLORS, CLASS_NAMES  # noqa: E402
from waste_classifier.plots import GREY_MARK, INK, INK_2, MUTED, SURFACE  # noqa: E402

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12.5,
                     "axes.labelsize": 12.5, "legend.fontsize": 12.5})


def boxplots(stats, out):
    feats = [("saturation", "Saturation (0-1)"), ("edge_density", "Edge density (detail)"),
             ("brightness", "Brightness (0-1)"), ("colourfulness", "Colourfulness")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, (feat, label) in zip(axes.ravel(), feats):
        data = [stats.loc[stats["class_name"] == c, feat].dropna().to_numpy() for c in CLASS_NAMES]
        bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": MUTED},
                        capprops={"color": MUTED})
        for patch, c in zip(bp["boxes"], CLASS_NAMES):
            patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.7); patch.set_edgecolor(SURFACE)
        ax.set_yticks(range(1, 7), CLASS_NAMES)
        ax.invert_yaxis()
        ax.set_title(label, loc="left")
        ax.grid(axis="y", visible=False)
    fig.tight_layout(h_pad=2.2, w_pad=2.5)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def tsne(coords, out):
    fig, ax = plt.subplots(figsize=(10, 7))
    for k, c in enumerate(CLASS_NAMES):
        m = coords["label"] == k
        ax.scatter(coords.loc[m, "x"], coords.loc[m, "y"], s=10, color=CLASS_COLORS[c], alpha=0.75, linewidth=0,
                   label=c, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=3, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main(results_dir):
    rd = Path(results_dir)
    out = rd / "figures" / "slides"
    out.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(rd / "metrics" / "image_stats.csv")
    stats = stats[stats["readable"].fillna(False).astype(bool)]
    boxplots(stats, out / "slide_boxplots.png")
    tsne(pd.read_csv(rd / "metrics" / "tsne_coordinates.csv"), out / "slide_tsne.png")
    print("slide figures in", out)


if __name__ == "__main__":
    main(sys.argv[1])
