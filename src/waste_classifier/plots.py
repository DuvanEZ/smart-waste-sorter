"""All figures used in the report (consistent style: thin marks, recessive grid,
one fixed colour per class, colour-vision-deficiency-safe palette)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

from .config import CLASS_COLORS, CLASS_NAMES  # noqa: E402

# ---- design tokens --------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GREY_MARK = "#d6d5ce"
ACCENT = "#2a78d6"        # categorical slot 1 (single-series charts)
ACCENT_2 = "#eb6834"      # categorical slot 2
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe"] + BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ec7b7a", "#b52e2e"])
SPLIT_COLORS = {"train": "#184f95", "val": "#5598e7", "test": "#9ec5f4"}


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 11.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 9.5,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 9.5, "legend.frameon": False,
        "legend.fontsize": 8.5, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })


set_style()


def _save(fig, out) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _suptitle(fig, title, subtitle=None):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK, y=1.0)
    if subtitle:
        fig.text(0.01, 0.955, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)


def _no_grid_x(ax):
    ax.grid(axis="x", visible=False)


def _no_grid_y(ax):
    ax.grid(axis="y", visible=False)


# =======================================================================================
# Part 1 - data analysis
# =======================================================================================
def class_distribution(counts: pd.Series, out, title="Images per class"):
    """Column chart of the number of images in each class (categorical variable)."""
    counts = counts.reindex(CLASS_NAMES).fillna(0)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(counts.index, counts.values, width=0.55, color=ACCENT, zorder=2)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.004, f"{int(value):,}\n{value / total:.1%}",
                ha="center", va="bottom", fontsize=8.5, color=INK_2)
    mean = counts.mean()
    ax.axhline(mean, color=MUTED, lw=1, zorder=1)
    ax.text(len(counts) - 0.45, mean, f"mean {mean:,.0f}", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, counts.max() * 1.22)
    _no_grid_x(ax)
    ax.set_title(title)
    return _save(fig, out)


def sample_grid(index: pd.DataFrame, out, per_class=6, seed=42):
    """Random examples of every class (qualitative inspection)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(per_class * 1.35, len(CLASS_NAMES) * 1.42))
    for r, cls in enumerate(CLASS_NAMES):
        paths = index.loc[index["class_name"] == cls, "path"].to_numpy()
        chosen = rng.choice(paths, size=min(per_class, len(paths)), replace=False) if len(paths) else []
        for c in range(per_class):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if c < len(chosen):
                with Image.open(chosen[c]) as im:
                    ax.imshow(im.convert("RGB"))
            if c == 0:
                ax.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=10, color=INK, labelpad=8)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    _suptitle(fig, "Random sample images from each class")
    return _save(fig, out)


def file_properties(stats: pd.DataFrame, out):
    """Resolution, file format, colour mode and file size (file-level variables)."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1.1, 0.8, 0.8, 1.5]})
    # (a) resolution
    ax = axes[0]
    dims = stats.groupby(["width", "height"]).size().reset_index(name="n")
    ax.scatter(dims["width"], dims["height"], s=40 + 260 * dims["n"] / dims["n"].max(), color=ACCENT,
               edgecolor=SURFACE, linewidth=2, zorder=3)
    for _, row in dims.nlargest(3, "n").iterrows():
        ax.annotate(f"{int(row.width)}x{int(row.height)}\n{int(row.n):,} images", (row.width, row.height),
                    xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK_2)
    pad = max(20, (dims["width"].max() - dims["width"].min()) * 0.2)
    ax.set_xlim(dims["width"].min() - pad, dims["width"].max() + pad * 2.5)
    ax.set_ylim(dims["height"].min() - pad, dims["height"].max() + pad * 2.5)
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)"); ax.set_title("(a) Resolution")
    # (b) format and (c) mode
    for ax, col, title in [(axes[1], "format", "(b) File format"), (axes[2], "mode", "(c) Colour mode")]:
        vc = stats[col].fillna("unreadable").value_counts()
        ax.bar(vc.index.astype(str), vc.values, width=0.5, color=ACCENT, zorder=2)
        for i, v in enumerate(vc.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_ylim(0, vc.max() * 1.18); ax.set_title(title); _no_grid_x(ax)
    # (d) file size
    ax = axes[3]
    ax.hist(stats["file_size_kb"], bins=50, color=ACCENT, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    med = stats["file_size_kb"].median()
    ax.axvline(med, color=INK_2, lw=1)
    ax.text(med, ax.get_ylim()[1] * 0.95, f"  median {med:.1f} KB", fontsize=8, color=INK_2, va="top")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("File size (KB)"); ax.set_ylabel("Images"); ax.set_title("(d) File size")
    fig.tight_layout()
    return _save(fig, out)


FEATURE_LABELS = {
    "file_size_kb": "File size (KB)", "brightness": "Brightness (0-1)", "contrast": "Contrast (0-1)",
    "saturation": "Saturation (0-1)", "colourfulness": "Colourfulness", "sharpness": "Sharpness (log10 var. Laplacian)",
    "entropy": "Entropy (bits)", "edge_density": "Edge density", "white_background": "White-background share of border",
    "mean_r": "Mean red", "mean_g": "Mean green", "mean_b": "Mean blue",
}


SHORT_LABELS = {
    "file_size_kb": "File size", "brightness": "Brightness", "contrast": "Contrast", "saturation": "Saturation",
    "colourfulness": "Colourfulness", "sharpness": "Sharpness", "entropy": "Entropy", "edge_density": "Edge density",
    "white_background": "White background", "mean_r": "Mean red", "mean_g": "Mean green", "mean_b": "Mean blue",
}


def feature_boxplots(stats: pd.DataFrame, features, out, title="Image statistics by class"):
    """Box plots of each numeric image variable, split by class."""
    n = len(features)
    cols = 4 if n % 4 == 0 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.7 * rows), squeeze=False)
    for ax, feat in zip(axes.ravel(), features):
        data = []
        for cls in CLASS_NAMES:
            v = stats.loc[stats["class_name"] == cls, feat].dropna().to_numpy()
            data.append(np.log10(v + 1) if feat == "sharpness" else v)
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.4},
                        whiskerprops={"color": MUTED, "linewidth": 1}, capprops={"color": MUTED, "linewidth": 1})
        for patch, cls in zip(bp["boxes"], CLASS_NAMES):
            patch.set_facecolor(CLASS_COLORS[cls]); patch.set_alpha(0.55); patch.set_edgecolor(SURFACE)
        ax.set_yticks(range(1, len(CLASS_NAMES) + 1), CLASS_NAMES)
        ax.invert_yaxis()
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        _no_grid_y(ax)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    _suptitle(fig, title, "Boxes show the middle 50% of images, the line is the median, whiskers 1.5 x IQR (outliers hidden)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out)


def white_background_share(stats: pd.DataFrame, out, cut=0.6):
    """Share of images per class whose border is mostly plain white (studio-style photo)."""
    share = stats.assign(white=stats["white_background"] >= cut).groupby("class_name")["white"].mean().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(share.index, share.values, height=0.55, color=ACCENT, zorder=2)
    for i, v in enumerate(share.values):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.2, share.max() * 1.25))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(f"Images whose 10-px border is >= {cut:.0%} near-white pixels")
    _no_grid_y(ax)
    ax.set_title("Plain white (studio) backgrounds by class")
    return _save(fig, out)


def channel_histograms(stats: pd.DataFrame, out, bins=32):
    """Average red / green / blue intensity distribution of each class (small multiples)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharex=True, sharey=True)
    x = (np.arange(bins) + 0.5) * (256 / bins)
    channel_colors = {"r": "#d03b3b", "g": "#1f8f3a", "b": "#2a78d6"}
    names = {"r": "red", "g": "green", "b": "blue"}
    for ax, cls in zip(axes.ravel(), CLASS_NAMES):
        sub = stats[stats["class_name"] == cls]
        for ch in "rgb":
            cols = [f"{ch}_hist_{i:02d}" for i in range(bins)]
            ax.plot(x, sub[cols].mean().to_numpy() * 100, color=channel_colors[ch], lw=1.8, label=names[ch])
        ax.set_title(cls, fontsize=10.5)
        ax.set_xlim(0, 255)
    for ax in axes[-1]:
        ax.set_xlabel("Pixel intensity (0 = dark, 255 = bright)")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of pixels")
    axes[0, 0].legend(loc="upper left")
    _suptitle(fig, "Colour-channel intensity distributions per class",
              "Average histogram of the red, green and blue channels over all images of the class")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, out)


def hue_heatmap(stats: pd.DataFrame, out, n_bins=18):
    """Heat-map of the hue distribution (which colours dominate each class)."""
    import colorsys

    cols = [f"hue_{i:02d}" for i in range(n_bins)]
    mat = stats.groupby("class_name")[cols].mean().reindex(CLASS_NAMES).to_numpy() * 100
    fig = plt.figure(figsize=(11, 3.9))
    ax = fig.add_axes([0.1, 0.3, 0.78, 0.5])
    strip = fig.add_axes([0.1, 0.2, 0.78, 0.045])
    cax = fig.add_axes([0.9, 0.3, 0.015, 0.5])
    im = ax.imshow(mat, aspect="auto", cmap=SEQ, vmin=0)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    hues = np.array([[colorsys.hsv_to_rgb((i * 20 + 10) / 360, 0.75, 0.9) for i in range(n_bins)]])
    strip.imshow(hues, aspect="auto")
    strip.set_yticks([])
    strip.set_xticks(np.arange(-0.5, n_bins, 3), [f"{d}°" for d in range(0, 361, 60)])
    strip.set_xlabel("Hue (colour wheel angle)")
    for s in strip.spines.values():
        s.set_visible(False)
    strip.grid(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("% of pixels", color=INK_2); cb.outline.set_visible(False)
    fig.text(0.01, 0.97, "Hue distribution of coloured pixels per class", fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.9, "Only pixels with saturation and value >= 40/255 are counted (greys and whites excluded)",
             fontsize=9.5, color=INK_2, va="top")
    return _save(fig, out)


def mean_images(mean_imgs: np.ndarray, out):
    """The pixel-wise average image of every class."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(12, 2.6))
    for ax, cls, img in zip(axes, CLASS_NAMES, mean_imgs):
        ax.imshow(img); ax.set_title(cls, fontsize=10.5, loc="center")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    _suptitle(fig, "Average image of each class")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out)


def correlation_heatmap(stats: pd.DataFrame, features, out):
    """Spearman correlation between the numeric image variables."""
    corr = stats[features].corr(method="spearman")
    labels = [SHORT_LABELS.get(f, f) for f in features]
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(corr.to_numpy(), cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(features)), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    vals = corr.to_numpy()
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j and abs(vals[i, j]) >= 0.5:
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(vals[i, j]) > 0.75 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman correlation", color=INK_2); cb.outline.set_visible(False)
    ax.set_title("Correlation between image variables (|r| >= 0.5 labelled)")
    return _save(fig, out)


def scatter_by_class(stats: pd.DataFrame, x, y, out, per_class=700, seed=42, logx=False, logy=False):
    """Relationship between two variables; one panel per class highlighted over all others."""
    rng = np.random.default_rng(seed)
    sample = pd.concat([d.sample(min(per_class, len(d)), random_state=int(rng.integers(1e9)))
                        for _, d in stats.groupby("class_name")])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=True, sharey=True)
    xs, ys = sample[x], sample[y]
    if logx:
        xs = np.log10(xs + 1)
    if logy:
        ys = np.log10(ys + 1)
    for ax, cls in zip(axes.ravel(), CLASS_NAMES):
        m = (sample["class_name"] == cls).to_numpy()
        ax.scatter(xs[~m], ys[~m], s=5, color=GREY_MARK, alpha=0.6, linewidth=0, rasterized=True)
        ax.scatter(xs[m], ys[m], s=9, color=CLASS_COLORS[cls], alpha=0.75, linewidth=0, rasterized=True)
        ax.set_title(cls, fontsize=10.5)
    xl, yl = FEATURE_LABELS.get(x, x), FEATURE_LABELS.get(y, y)
    for ax in axes[-1]:
        ax.set_xlabel(("log10 " if logx else "") + xl)
    for ax in axes[:, 0]:
        ax.set_ylabel(("log10 " if logy else "") + yl)
    _suptitle(fig, f"{yl.split(' (')[0]} vs {xl.split(' (')[0].lower()} by class",
              f"Each panel highlights one class (colour) over a sample of all images (grey), {per_class} images per class")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out)


def tsne_small_multiples(coords: np.ndarray, labels: np.ndarray, out, title=None):
    """2-D t-SNE map of deep features; one panel per class highlighted over the others."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for ax, (k, cls) in zip(axes.ravel(), enumerate(CLASS_NAMES)):
        m = labels == k
        ax.scatter(coords[~m, 0], coords[~m, 1], s=4, color=GREY_MARK, alpha=0.6, linewidth=0, rasterized=True)
        ax.scatter(coords[m, 0], coords[m, 1], s=7, color=CLASS_COLORS[cls], alpha=0.85, linewidth=0, rasterized=True)
        ax.set_title(f"{cls} (n={m.sum():,})", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor(GRID)
    _suptitle(fig, title or "t-SNE map of pre-trained EfficientNet-B0 features",
              "Nearby points = images the network considers similar; each panel highlights one class")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out)


def duplicate_histogram(nn_dist: np.ndarray, out, threshold=4):
    """Distribution of the distance from each image to its most similar other image."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    counts = np.bincount(nn_dist.astype(int), minlength=33)[:33]
    colors = [ACCENT_2 if d <= threshold else ACCENT for d in range(33)]
    ax.bar(np.arange(33), counts, width=0.8, color=colors, zorder=2)
    ax.axvline(threshold + 0.5, color=INK_2, lw=1)
    n_dup = int((nn_dist <= threshold).sum())
    ax.text(threshold + 0.8, counts.max() * 0.95,
            f"<= {threshold}: near-duplicates\n{n_dup:,} images ({n_dup / len(nn_dist):.1%})",
            fontsize=8.5, color=INK_2, va="top")
    ax.set_xlabel("pHash Hamming distance to the most similar other image (0 = identical)")
    ax.set_ylabel("Images")
    _no_grid_x(ax)
    ax.set_title("Near-duplicate analysis with perceptual hashing")
    return _save(fig, out)


def image_pairs(pairs, out, title, max_pairs=6):
    """Show example pairs of images side by side (e.g. near-duplicates)."""
    from PIL import Image

    pairs = pairs[:max_pairs]
    if not pairs:
        return None
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.1 * len(pairs), 4.6), squeeze=False)
    for c, (pa, pb, caption) in enumerate(pairs):
        for r, p in enumerate([pa, pb]):
            ax = axes[r, c]
            with Image.open(p) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        axes[1, c].set_xlabel(caption, fontsize=8, color=INK_2)
    _suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out)


def split_distribution(df: pd.DataFrame, out):
    """Images per class in the training, validation and test sets."""
    table = df.groupby(["class_name", "split"]).size().unstack("split").reindex(CLASS_NAMES)[["train", "val", "test"]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.26
    x = np.arange(len(CLASS_NAMES))
    for k, split in enumerate(["train", "val", "test"]):
        bars = ax.bar(x + (k - 1) * (width + 0.02), table[split].to_numpy(), width=width,
                      color=SPLIT_COLORS[split], label=f"{split} ({table[split].sum():,})", zorder=2)
        if split == "test":
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(b.get_height())}",
                        ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("Images")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.set_ylim(0, table.to_numpy().max() * 1.2)
    _no_grid_x(ax)
    ax.set_title("Stratified group split: images per class and subset", pad=24)
    return _save(fig, out)


def augmentation_examples(paths, augment, out, n_aug=5, seed=42):
    """Original images next to randomly augmented versions (training transformation)."""
    import torch
    from PIL import Image

    torch.manual_seed(seed)
    fig, axes = plt.subplots(len(paths), n_aug + 1, figsize=(1.9 * (n_aug + 1), 2.0 * len(paths)))
    for r, p in enumerate(paths):
        with Image.open(p) as im:
            img = im.convert("RGB")
        views = [img.resize((224, 224))] + [augment(img) for _ in range(n_aug)]
        for c, v in enumerate(views):
            ax = axes[r, c]
            ax.imshow(v); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title("original" if c == 0 else f"augmented #{c}", fontsize=9, loc="center")
    _suptitle(fig, "Data augmentation applied to training images")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out)


# =======================================================================================
# Part 2 - modelling and evaluation
# =======================================================================================
def learning_curves(history: pd.DataFrame, out, title="Learning curves - EfficientNet-B0"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    ep = history["epoch"]
    switch = history.loc[history["phase"] != history["phase"].iloc[0], "epoch"]
    for ax, metric, label in [(axes[0], "loss", "Cross-entropy loss"), (axes[1], "acc", "Accuracy")]:
        ax.plot(ep, history[f"train_{metric}"], color=ACCENT, marker="o", ms=4, label="training")
        ax.plot(ep, history[f"val_{metric}"], color=ACCENT_2, marker="o", ms=4, label="validation")
        if len(switch):
            ax.axvline(switch.iloc[0] - 0.5, color=MUTED, lw=1)
            ax.text(switch.iloc[0] - 0.4, ax.get_ylim()[1], "  fine-tuning starts", fontsize=8,
                    color=MUTED, va="top")
        ax.set_xlabel("Epoch"); ax.set_title(label)
        ax.set_xticks(ep)
    axes[0].legend(loc="upper right")
    _suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out)


def model_comparison(results: pd.DataFrame, out, metric="test_macro_f1", highlight=None):
    """Horizontal bars: test macro-F1 of every model (the chosen model highlighted)."""
    df = results.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 1.3))
    colors = [ACCENT if m == highlight else GREY_MARK for m in df["model"]]
    ax.barh(df["model"], df[metric], height=0.55, color=colors, zorder=2)
    for i, v in enumerate(df[metric]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro-F1 on the test set")
    _no_grid_y(ax)
    ax.set_title("Model comparison (same train / validation / test split)")
    return _save(fig, out)


def confusion_matrix_plot(cm: np.ndarray, out, title="Confusion matrix - test set"):
    cm_norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm_norm, cmap=SEQ, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.1%}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Share of the true class (row %)", color=INK_2); cb.outline.set_visible(False)
    ax.set_title(title)
    return _save(fig, out)


def per_class_metrics(report: pd.DataFrame, out):
    """Dot plot of precision, recall and F1 for each class."""
    fig, ax = plt.subplots(figsize=(8, 3.9))
    y = np.arange(len(report))
    styles = [("precision", "#2a78d6", "o"), ("recall", "#eb6834", "s"), ("f1", "#1baf7a", "D")]
    lo = min(report[["precision", "recall", "f1"]].min().min(), 0.9) - 0.03
    for k, (col, color, marker) in enumerate(styles):
        ax.scatter(report[col], y + (k - 1) * 0.18, color=color, marker=marker, s=46,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=col if col != "f1" else "F1-score")
    for i, row in report.iterrows():
        ax.text(1.005, i, f"F1 {row.f1:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, report["class"])
    ax.invert_yaxis()
    ax.set_xlim(max(0.0, lo), 1.05)
    ax.set_xlabel("Score on the test set")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.12))
    ax.set_title("Per-class precision, recall and F1", pad=26)
    return _save(fig, out)


def roc_pr_curves(y: np.ndarray, prob: np.ndarray, out):
    """One-vs-rest ROC and precision-recall curves for every class."""
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for k, cls in enumerate(CLASS_NAMES):
        yk = (y == k).astype(int)
        if yk.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yk, prob[:, k])
        prec, rec, _ = precision_recall_curve(yk, prob[:, k])
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(yk, prob[:, k])
        for ax in axes[:2]:
            ax.plot(fpr, tpr, color=CLASS_COLORS[cls], lw=1.8, label=f"{cls} (AUC {roc_auc:.3f})")
        axes[2].plot(rec, prec, color=CLASS_COLORS[cls], lw=1.8, label=f"{cls} (AP {ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color=MUTED, lw=1)
    axes[0].set_title("ROC curves (one-vs-rest)")
    axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate")
    axes[0].legend(loc="lower right", fontsize=7.5)
    axes[1].set_xlim(0, 0.2); axes[1].set_ylim(0.8, 1.001)
    axes[1].set_title("ROC - zoom on the top-left corner")
    axes[1].set_xlabel("False-positive rate")
    axes[2].set_title("Precision-recall curves")
    axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
    axes[2].set_ylim(0.5, 1.01)
    axes[2].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, out)


def reliability_diagram(before: pd.DataFrame, after: pd.DataFrame, ece_before, ece_after, temperature, out):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, label="perfect calibration")
    for df, color, label in [(before, ACCENT_2, f"before scaling (ECE {ece_before:.3f})"),
                             (after, ACCENT, f"after T = {temperature:.2f} (ECE {ece_after:.3f})")]:
        d = df.dropna()
        ax.plot(d["confidence"], d["accuracy"], color=color, marker="o", ms=5, label=label)
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Reliability diagram (test set)")
    return _save(fig, out)


def confidence_analysis(prob: np.ndarray, y: np.ndarray, selective: pd.DataFrame, threshold, out):
    conf, correct = prob.max(1), prob.argmax(1) == y
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    bins = np.linspace(0, 1, 26)
    ax = axes[0]
    ax.hist(conf[correct], bins=bins, color=ACCENT, alpha=0.85, label=f"correct ({correct.sum():,})", zorder=2)
    ax.hist(conf[~correct], bins=bins, color=ACCENT_2, alpha=0.9, label=f"wrong ({(~correct).sum():,})", zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Confidence of the predicted class"); ax.set_ylabel("Images (log scale)")
    ax.legend(loc="upper left")
    ax.set_title("Confidence of correct vs wrong predictions")
    ax = axes[1]
    ax.plot(selective["threshold"], selective["accuracy"], color=ACCENT, marker="o", ms=3.5, label="accuracy of accepted")
    ax.plot(selective["threshold"], selective["coverage"], color=ACCENT_2, marker="s", ms=3.5, label="share accepted (coverage)")
    ax.axvline(threshold, color=INK_2, lw=1)
    row = selective.iloc[(selective["threshold"] - threshold).abs().argmin()]
    ax.text(threshold - 0.02, 0.45,
            f"app threshold {threshold:.0%}\naccuracy {row.accuracy:.1%}\ncoverage {row.coverage:.1%}",
            fontsize=8, color=INK_2, va="center", ha="right",
            bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round,pad=0.3"})
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Proportion")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.18))
    ax.set_title("Selective prediction: rejecting uncertain images")
    fig.tight_layout()
    return _save(fig, out)


def image_gallery(items, out, title, cols=6, subtitle=None):
    """items: list of (image_path, caption, is_correct) shown in a grid."""
    from PIL import Image

    if not items:
        return None
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.05 * cols, 2.35 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for ax, (path, caption, ok) in zip(axes.ravel(), items):
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_xlabel(caption, fontsize=7.8, color=INK if ok else "#b52e2e")
    _suptitle(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.93))
    return _save(fig, out)


def gradcam_grid(images: np.ndarray, cams: np.ndarray, captions, out, cols=4):
    """Pairs of (image, Grad-CAM overlay)."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(2.0 * cols * 2, 2.25 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    for k in range(n):
        r, c = divmod(k, cols)
        axes[r, 2 * c].imshow(images[k])
        axes[r, 2 * c + 1].imshow(images[k])
        axes[r, 2 * c + 1].imshow(cams[k], cmap="jet", alpha=0.45)
        axes[r, 2 * c].set_xlabel(captions[k], fontsize=7.8, color=INK_2)
        axes[r, 2 * c + 1].set_xlabel("Grad-CAM", fontsize=7.8, color=MUTED)
    _suptitle(fig, "Grad-CAM: image regions that drove the prediction",
              "Red = high influence on the predicted class, blue = low influence")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out)


def robustness_bars(df: pd.DataFrame, out):
    df = df.copy()
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(df) + 1.2))
    colors = [ACCENT if c == "original" else "#9ec5f4" for c in df["corruption"]]
    ax.barh(df["corruption"], df["accuracy"], height=0.55, color=colors, zorder=2)
    for i, v in enumerate(df["accuracy"]):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Accuracy on the test set after the corruption")
    _no_grid_y(ax)
    ax.set_title("Robustness to image corruptions")
    return _save(fig, out)
