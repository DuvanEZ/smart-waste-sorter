"""End-to-end pipeline, split into the steps of the report.

Each step can be called on its own (the Colab notebook runs them one cell at a time)
or all together with `run_all` (scripts/run_pipeline.py). Every step writes its
figures to <out>/figures and its numbers to <out>/metrics, so the report can be
rebuilt from the saved outputs.
"""
from __future__ import annotations

import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import (CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, BaselineConfig, SplitConfig,
                     TrainConfig)
from .data import (augmentation_preview, build_index, build_transforms, compare_with_metadata,
                   extract_zip, find_dataset_root, find_metadata_csv, load_rgb, make_dataset,
                   make_loader, make_splits)
from .image_analysis import (NUMERIC_FEATURES, analyse_dataset, duplicate_groups, hashes_to_uint64,
                             near_duplicates)
from .utils import environment_info, save_json, set_seed

KAGGLE_HANDLE = "zlatan599/garbage-dataset-classification"
APP_THRESHOLD = 0.60  # confidence below which the app reports "uncertain"


@dataclass
class Workspace:
    """Output folders of one pipeline run."""

    out_dir: Path

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        for d in [self.fig_dir, self.metrics_dir, self.model_dir, self.cache_dir, self.sample_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def fig_dir(self):
        return self.out_dir / "figures"

    @property
    def metrics_dir(self):
        return self.out_dir / "metrics"

    @property
    def model_dir(self):
        return self.out_dir / "model"

    @property
    def cache_dir(self):
        return self.out_dir / "cache"

    @property
    def sample_dir(self):
        return self.out_dir / "sample_images"

    def fig(self, name):
        return self.fig_dir / name


@dataclass
class Context:
    """Objects shared between the steps."""

    root: Path | None = None
    index: pd.DataFrame | None = None
    stats: pd.DataFrame | None = None
    pairs: pd.DataFrame | None = None
    mean_images: np.ndarray | None = None
    summary: dict = field(default_factory=dict)
    data: pd.DataFrame | None = None          # readable images + split column
    embeddings: np.ndarray | None = None
    baselines: pd.DataFrame | None = None
    model: object = None
    history: pd.DataFrame | None = None
    temperature: float = 1.0
    test_metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


# =======================================================================================
# Step 0 - get the data
# =======================================================================================
def acquire_data(zip_path=None, data_dir=None, extract_to="/content/data", use_kagglehub=True, log=print) -> Path:
    """Return the dataset root (the folder that contains one sub-folder per class).

    Priority: an already extracted folder -> a zip file (e.g. on Google Drive) -> kagglehub.
    """
    if data_dir and Path(data_dir).exists():
        return find_dataset_root(data_dir)
    if zip_path and Path(zip_path).exists():
        log(f"Extracting {zip_path} ...")
        return find_dataset_root(extract_zip(zip_path, extract_to))
    if use_kagglehub:
        import kagglehub

        log(f"Downloading {KAGGLE_HANDLE} with kagglehub ...")
        return find_dataset_root(kagglehub.dataset_download(KAGGLE_HANDLE))
    raise FileNotFoundError("Dataset not found: give zip_path, data_dir or enable kagglehub")


# =======================================================================================
# Step 1.1 - description of the data set (counts, formats, quality, duplicates)
# =======================================================================================
def data_understanding(root, ws: Workspace, split_cfg=SplitConfig(), workers=2, log=print) -> Context:
    t0 = time.time()
    ctx = Context(root=Path(root))
    index = build_index(root)
    other_files = index.loc[~index["is_image_file"], "rel_path"].tolist()
    index = index[index["is_image_file"]].reset_index(drop=True)
    log(f"Found {len(index):,} image files in {index['class_name'].nunique()} class folders "
        f"({len(other_files)} other files)")

    stats, mean_imgs = analyse_dataset(index, len(CLASS_NAMES), workers=workers, log=log)
    ok = stats["readable"].fillna(False).to_numpy(dtype=bool)
    log(f"Readable images: {ok.sum():,} / {len(stats):,}")

    # ---- duplicates ------------------------------------------------------------------
    stats["md5_dup_count"] = stats.groupby("md5")["md5"].transform("size")
    readable_idx = np.flatnonzero(ok)
    hashes = hashes_to_uint64(stats.loc[ok, "phash"])
    pairs_r, nn_r = near_duplicates(hashes, threshold=max(8, split_cfg.dup_threshold))
    pairs = pairs_r.assign(i=readable_idx[pairs_r["i"]], j=readable_idx[pairs_r["j"]])
    stats["nn_phash_distance"] = np.nan
    stats.loc[ok, "nn_phash_distance"] = nn_r
    close = pairs[pairs["distance"] <= split_cfg.dup_threshold]
    stats["dup_group"] = duplicate_groups(len(stats), close)
    lab = stats["label"].to_numpy()
    pairs["same_class"] = lab[pairs["i"]] == lab[pairs["j"]]

    summary = {
        "dataset_root": str(root),
        "n_images": int(len(stats)),
        "n_readable": int(ok.sum()),
        "n_classes": int(stats["class_name"].nunique()),
        "class_counts": stats["class_name"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict(),
        "formats": stats["format"].value_counts(dropna=False).to_dict(),
        "modes": stats["mode"].value_counts(dropna=False).to_dict(),
        "extensions": stats["ext"].value_counts().to_dict(),
        "resolutions": {f"{int(w)}x{int(h)}": int(n) for (w, h), n in
                        stats.groupby(["width", "height"]).size().sort_values(ascending=False).head(10).items()},
        "file_size_kb": stats["file_size_kb"].describe().round(2).to_dict(),
        "total_size_mb": float(stats["file_size_kb"].sum() / 1024),
        "other_files": other_files,
        "unreadable_files": stats.loc[~ok, ["rel_path", "error"]].to_dict("records"),
        "exact_duplicate_files": int((stats["md5_dup_count"] > 1).sum()),
        "exact_duplicate_groups": int(stats.loc[stats["md5_dup_count"] > 1, "md5"].nunique()),
        "near_duplicate_pairs": {f"<={t}": int((pairs["distance"] <= t).sum()) for t in [0, 2, 4, 6, 8]},
        "images_with_near_duplicate": {f"<={t}": int((stats["nn_phash_distance"] <= t).sum()) for t in [0, 2, 4, 6, 8]},
        "cross_class_pairs": {f"<={t}": int(((pairs["distance"] <= t) & ~pairs["same_class"]).sum()) for t in [0, 2, 4]},
        "dup_threshold": split_cfg.dup_threshold,
        "n_dup_groups_multi": int((stats["dup_group"].value_counts() > 1).sum()),
    }
    counts = pd.Series(summary["class_counts"])
    shares = counts / counts.sum()
    summary["imbalance_ratio_max_min"] = float(counts.max() / max(counts.min(), 1))
    summary["class_share_min_max"] = [float(shares.min()), float(shares.max())]
    summary["shannon_evenness"] = float(-(shares * np.log(shares)).sum() / np.log(len(shares)))
    meta = find_metadata_csv(root)
    if meta is not None:
        try:
            summary["metadata_check"] = compare_with_metadata(index, meta)
        except Exception as exc:  # metadata is only a cross-check
            summary["metadata_check"] = {"error": str(exc)}
    feat_summary = stats.groupby("class_name")[NUMERIC_FEATURES].agg(["mean", "std", "median"]).round(4)
    feat_summary.to_csv(ws.metrics_dir / "image_statistics_by_class.csv")
    stats[NUMERIC_FEATURES].describe().T.round(4).to_csv(ws.metrics_dir / "image_statistics_overall.csv")

    stats.to_pickle(ws.cache_dir / "image_stats.pkl")
    light_cols = [c for c in stats.columns if "_hist_" not in c and not c.startswith("hue_")]
    stats[light_cols].drop(columns=["path"]).to_csv(ws.metrics_dir / "image_stats.csv", index=False)
    pairs.to_csv(ws.metrics_dir / "near_duplicate_pairs.csv", index=False)
    np.save(ws.cache_dir / "mean_images.npy", mean_imgs)
    save_json(summary, ws.metrics_dir / "dataset_summary.json")

    ctx.index, ctx.stats, ctx.pairs, ctx.mean_images, ctx.summary = index, stats, pairs, mean_imgs, summary
    ctx.timings["data_understanding_s"] = time.time() - t0
    log(f"Class counts: {summary['class_counts']}")
    log(f"Exact duplicate files: {summary['exact_duplicate_files']}, near-duplicate pairs "
        f"(<= {split_cfg.dup_threshold}): {summary['near_duplicate_pairs'].get(f'<={split_cfg.dup_threshold}', 'n/a')}")
    return ctx


# =======================================================================================
# Step 1.2 - visualisation
# =======================================================================================
def visualise(ctx: Context, ws: Workspace, log=print) -> list:
    t0 = time.time()
    s = ctx.stats[ctx.stats["readable"].fillna(False).astype(bool)]
    figs = [
        plots.class_distribution(ctx.stats["class_name"].value_counts(), ws.fig("fig01_class_distribution.png")),
        plots.sample_grid(s, ws.fig("fig02_sample_images.png")),
        plots.file_properties(ctx.stats, ws.fig("fig03_file_properties.png")),
        plots.feature_boxplots(s, ["brightness", "contrast", "saturation", "colourfulness", "sharpness",
                                   "entropy", "edge_density", "file_size_kb"],
                               ws.fig("fig04_image_statistics_by_class.png")),
        plots.white_background_share(s, ws.fig("fig04b_white_background.png")),
        plots.channel_histograms(s, ws.fig("fig05_rgb_histograms.png")),
        plots.hue_heatmap(s, ws.fig("fig06_hue_heatmap.png")),
        plots.mean_images(ctx.mean_images, ws.fig("fig07_mean_images.png")),
        plots.correlation_heatmap(s, NUMERIC_FEATURES, ws.fig("fig08_correlation_heatmap.png")),
        plots.scatter_by_class(s, "brightness", "saturation", ws.fig("fig09_brightness_vs_saturation.png")),
        plots.duplicate_histogram(s["nn_phash_distance"].to_numpy(), ws.fig("fig10_near_duplicates.png"),
                                  threshold=ctx.summary["dup_threshold"]),
    ]
    # Example near-duplicate pairs (same class first, then pairs with different labels).
    st, p = ctx.stats, ctx.pairs
    examples = []
    for _, row in p[p["distance"] <= ctx.summary["dup_threshold"]].sort_values("distance").head(4).iterrows():
        a, b = st.iloc[int(row.i)], st.iloc[int(row.j)]
        examples.append((a.path, b.path, f"{a.class_name} / {b.class_name}\ndistance {int(row.distance)}"))
    for _, row in p[(~p["same_class"]) & (p["distance"] <= ctx.summary["dup_threshold"])].head(3).iterrows():
        a, b = st.iloc[int(row.i)], st.iloc[int(row.j)]
        examples.append((a.path, b.path, f"{a.class_name} / {b.class_name}\ndistance {int(row.distance)}"))
    if examples:
        figs.append(plots.image_pairs(examples, ws.fig("fig11_duplicate_examples.png"),
                                      "Examples of near-duplicate pairs (top and bottom image of each column)"))
    ctx.timings["visualise_s"] = time.time() - t0
    log(f"Saved {len([f for f in figs if f])} figures to {ws.fig_dir}")
    return [f for f in figs if f]


# =======================================================================================
# Step 1.2 (cont.) - deep features: t-SNE map of class similarity
# =======================================================================================
def deep_embeddings(ctx: Context, ws: Workspace, device, batch_size=128, workers=2, log=print) -> np.ndarray:
    """1,280-d EfficientNet-B0 (ImageNet) embedding of every readable image (cached)."""
    from .models import create_feature_extractor, extract_embeddings

    cache = ws.cache_dir / "embeddings_imagenet.npz"
    readable = ctx.stats[ctx.stats["readable"].fillna(False).astype(bool)]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if len(z["paths"]) == len(readable):
            ctx.embeddings = z["emb"]
            return ctx.embeddings
    t0 = time.time()
    ds = make_dataset(readable["path"], readable["label"], build_transforms(224, train=False))
    loader = make_loader(ds, batch_size, shuffle=False, num_workers=workers, device_type=device.type)
    emb, _ = extract_embeddings(create_feature_extractor(pretrained=True), loader, device)
    np.savez_compressed(cache, emb=emb.astype(np.float16), paths=readable["path"].to_numpy())
    ctx.embeddings = emb.astype(np.float32)
    ctx.timings["embeddings_s"] = time.time() - t0
    log(f"Embeddings: {emb.shape} in {time.time() - t0:.0f}s")
    return ctx.embeddings


def tsne_map(ctx: Context, ws: Workspace, per_class=500, seed=42, log=print):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    readable = ctx.stats[ctx.stats["readable"].fillna(False).astype(bool)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(readable["label"] == k),
                                     size=min(per_class, int((readable["label"] == k).sum())), replace=False)
                          for k in range(len(CLASS_NAMES)) if (readable["label"] == k).any()])
    x = PCA(n_components=min(50, len(idx) - 1), random_state=seed).fit_transform(ctx.embeddings[idx])
    coords = TSNE(n_components=2, perplexity=min(35, max(5, len(idx) // 10)), init="pca",
                  learning_rate="auto", random_state=seed).fit_transform(x)
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": readable["label"].to_numpy()[idx]}) \
        .to_csv(ws.metrics_dir / "tsne_coordinates.csv", index=False)
    return plots.tsne_small_multiples(coords, readable["label"].to_numpy()[idx], ws.fig("fig12_tsne_embeddings.png"))


# =======================================================================================
# Step 2.2 - data transformation: split + augmentation
# =======================================================================================
def split_data(ctx: Context, ws: Workspace, split_cfg=SplitConfig(), log=print) -> pd.DataFrame:
    data = ctx.stats[ctx.stats["readable"].fillna(False).astype(bool)].reset_index(drop=True)
    # Exact duplicates inside the same class are kept once (they add no information).
    before = len(data)
    data = data.drop_duplicates(subset=["md5", "label"]).reset_index(drop=True)
    removed_exact = before - len(data)
    # Identical files with *different* labels are ambiguous -> removed completely.
    conflict = data.groupby("md5")["label"].transform("nunique") > 1
    removed_conflicts = int(conflict.sum())
    data = data[~conflict].reset_index(drop=True)
    data["split"] = make_splits(data["label"], data["dup_group"], split_cfg.test_size, split_cfg.val_size, split_cfg.seed)
    ctx.data = data
    data[["rel_path", "class_name", "label", "split", "dup_group"]].to_csv(ws.metrics_dir / "splits.csv", index=False)
    table = data.groupby(["class_name", "split"]).size().unstack("split").reindex(CLASS_NAMES)[["train", "val", "test"]]
    table.loc["total"] = table.sum()
    table.to_csv(ws.metrics_dir / "split_counts.csv")
    ctx.summary.update(removed_exact_duplicates=removed_exact, removed_label_conflicts=removed_conflicts,
                       split_counts=table.to_dict())
    save_json(ctx.summary, ws.metrics_dir / "dataset_summary.json")
    # leakage check: no duplicate group in two splits
    leak = data.groupby("dup_group")["split"].nunique().gt(1).sum()
    log(f"Removed {removed_exact} exact duplicates and {removed_conflicts} conflicting files; "
        f"groups spanning several splits: {leak}")
    log(table.to_string())
    plots.split_distribution(data, ws.fig("fig13_split_distribution.png"))
    rng = np.random.default_rng(split_cfg.seed)
    demo = [data.loc[data["label"] == k, "path"].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            for k in [0, 1, 4] if (data["label"] == k).any()]
    plots.augmentation_examples(demo, augmentation_preview(224), ws.fig("fig14_augmentation_examples.png"))
    return data


def _subset(ctx, split):
    d = ctx.data
    return d[d["split"] == split]


def _loaders(ctx, img_size, batch_size, workers, device, train_aug=True):
    tr, va, te = (_subset(ctx, s) for s in ["train", "val", "test"])
    ds_tr = make_dataset(tr["path"], tr["label"], build_transforms(img_size, train=train_aug))
    ds_va = make_dataset(va["path"], va["label"], build_transforms(img_size, train=False))
    ds_te = make_dataset(te["path"], te["label"], build_transforms(img_size, train=False))
    return (make_loader(ds_tr, batch_size, True, workers, device.type),
            make_loader(ds_va, batch_size, False, workers, device.type),
            make_loader(ds_te, batch_size, False, workers, device.type))


# =======================================================================================
# Step 2.1 / 2.4 - baseline models for comparison
# =======================================================================================
def _scores(y, pred, prefix):
    from sklearn.metrics import accuracy_score, f1_score

    return {f"{prefix}_accuracy": accuracy_score(y, pred), f"{prefix}_macro_f1": f1_score(y, pred, average="macro")}


def baselines(ctx: Context, ws: Workspace, device, cfg=BaselineConfig(), run_cnn=True, workers=2, log=print) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    from .features import extract_features

    t0 = time.time()
    data = ctx.data
    cache = ws.cache_dir / "handcrafted_features.npy"
    if cache.exists() and np.load(cache).shape[0] == len(data):
        X = np.load(cache)
    else:
        log("Extracting hand-crafted features (colour histogram + HOG + LBP) ...")
        X = extract_features(data["path"], size=cfg.handcrafted_size, workers=workers)
        np.save(cache, X)
    y = data["label"].to_numpy()
    m = {s: (data["split"] == s).to_numpy() for s in ["train", "val", "test"]}
    log(f"Feature matrix {X.shape}")

    n_comp = min(cfg.pca_components, X.shape[1], int(m["train"].sum()) - 1)
    models = {
        "k-NN (k=5) + HOG/colour/LBP": make_pipeline(StandardScaler(), PCA(n_comp, random_state=cfg.seed),
                                                     KNeighborsClassifier(5)),
        "Logistic regression + HOG/colour/LBP": make_pipeline(StandardScaler(), PCA(n_comp, random_state=cfg.seed),
                                                              LogisticRegression(max_iter=3000, C=0.5)),
        "Random forest + HOG/colour/LBP": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=cfg.seed),
        "SVM (RBF) + HOG/colour/LBP": make_pipeline(StandardScaler(), PCA(n_comp, random_state=cfg.seed),
                                                    SVC(C=10, gamma="scale")),
    }
    rows = []
    for name, clf in models.items():
        start = time.time()
        clf.fit(X[m["train"]], y[m["train"]])
        fit_s = time.time() - start
        row = {"model": name, "family": "classical ML", "fit_seconds": fit_s}
        row.update(_scores(y[m["val"]], clf.predict(X[m["val"]]), "val"))
        row.update(_scores(y[m["test"]], clf.predict(X[m["test"]]), "test"))
        rows.append(row)
        log(f"  {name}: val acc {row['val_accuracy']:.3f}, test acc {row['test_accuracy']:.3f} ({fit_s:.0f}s)")

    # Linear probe: frozen ImageNet features + logistic regression (transfer learning without fine-tuning)
    if ctx.embeddings is not None:
        readable = ctx.stats[ctx.stats["readable"].fillna(False).astype(bool)].reset_index(drop=True)
        pos = pd.Series(np.arange(len(readable)), index=readable["path"])
        E = ctx.embeddings[pos.loc[data["path"]].to_numpy()]
        start = time.time()
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
        probe.fit(E[m["train"]], y[m["train"]])
        row = {"model": "Frozen EfficientNet-B0 + logistic regression", "family": "transfer learning (frozen)",
               "fit_seconds": time.time() - start}
        row.update(_scores(y[m["val"]], probe.predict(E[m["val"]]), "val"))
        row.update(_scores(y[m["test"]], probe.predict(E[m["test"]]), "test"))
        rows.append(row)
        log(f"  linear probe: val acc {row['val_accuracy']:.3f}, test acc {row['test_accuracy']:.3f}")

    if run_cnn:
        import torch

        from .models import SimpleCNN, count_parameters
        from .train import fit_from_scratch, predict_logits

        set_seed(cfg.seed)
        tr_loader, va_loader, te_loader = _loaders(ctx, cfg.cnn_img_size, cfg.cnn_batch_size, workers, device)
        cnn = SimpleCNN(len(CLASS_NAMES))
        log(f"Small CNN from scratch: {count_parameters(cnn)[0]:,} parameters, {cfg.cnn_epochs} epochs max")
        start = time.time()
        cnn, hist, best_ep = fit_from_scratch(cnn, tr_loader, va_loader, cfg.cnn_epochs, cfg.cnn_lr, device, log=log)
        hist.to_csv(ws.metrics_dir / "history_small_cnn.csv", index=False)
        va_logits, va_y = predict_logits(cnn, va_loader, device)
        te_logits, te_y = predict_logits(cnn, te_loader, device)
        row = {"model": "Small CNN trained from scratch", "family": "deep learning (scratch)",
               "fit_seconds": time.time() - start}
        row.update(_scores(va_y, va_logits.argmax(1), "val"))
        row.update(_scores(te_y, te_logits.argmax(1), "test"))
        rows.append(row)
        plots.learning_curves(hist, ws.fig("fig15b_learning_curves_small_cnn.png"), "Learning curves - small CNN from scratch")
        del cnn
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ctx.baselines = pd.DataFrame(rows)
    ctx.baselines.to_csv(ws.metrics_dir / "baseline_results.csv", index=False)
    ctx.timings["baselines_s"] = time.time() - t0
    return ctx.baselines


# =======================================================================================
# Step 2.3 - train the final model
# =======================================================================================
def train_final(ctx: Context, ws: Workspace, device, cfg=TrainConfig(), workers=2, log=print):
    import torch

    from .models import create_efficientnet, count_parameters
    from .train import fit_transfer_learning

    t0 = time.time()
    set_seed(cfg.seed)
    tr_loader, va_loader, _ = _loaders(ctx, cfg.img_size, cfg.batch_size, workers, device)
    model = create_efficientnet(len(CLASS_NAMES), pretrained=True, drop_rate=cfg.drop_rate,
                                drop_path_rate=cfg.drop_path_rate)
    total, _ = count_parameters(model)
    log(f"EfficientNet-B0: {total:,} parameters; train {len(tr_loader.dataset):,} / val {len(va_loader.dataset):,} images")
    model, history, best_epoch = fit_transfer_learning(model, tr_loader, va_loader, cfg, device, log=log)
    history.to_csv(ws.metrics_dir / "history_efficientnet_b0.csv", index=False)
    torch.save(model.state_dict(), ws.model_dir / "efficientnet_b0_best.pt")
    save_json({**cfg.to_dict(), "best_epoch": best_epoch, "train_seconds": time.time() - t0,
               "n_parameters": total}, ws.metrics_dir / "training_config.json")
    plots.learning_curves(history, ws.fig("fig15_learning_curves.png"))
    ctx.model, ctx.history = model, history
    ctx.timings["train_final_s"] = time.time() - t0
    return model, history


# =======================================================================================
# Step 2.4 - evaluation of the final model
# =======================================================================================
def evaluate_final(ctx: Context, ws: Workspace, device, cfg=TrainConfig(), workers=2, threshold=APP_THRESHOLD,
                   robustness_max=1200, log=print) -> dict:
    import torch

    from . import evaluate as ev
    from .explain import GradCAM, denormalise
    from .train import predict_logits

    t0 = time.time()
    model = ctx.model.to(device).eval()
    _, va_loader, te_loader = _loaders(ctx, cfg.img_size, cfg.batch_size, workers, device)
    va_logits, va_y = predict_logits(model, va_loader, device)
    te_logits, te_y = predict_logits(model, te_loader, device)
    T = ev.fit_temperature(va_logits, va_y)
    prob_raw, prob = ev.softmax(te_logits), ev.softmax(te_logits, T)
    ctx.temperature = T

    metrics = ev.classification_metrics(te_y, prob, CLASS_NAMES)
    metrics["ece_before_temperature"] = ev.expected_calibration_error(prob_raw, te_y)
    metrics["log_loss_before_temperature"] = ev.classification_metrics(te_y, prob_raw, CLASS_NAMES)["log_loss"]
    metrics["temperature"] = T
    metrics.update(ev.bootstrap_ci(te_y, prob.argmax(1)))
    val_metrics = ev.classification_metrics(va_y, ev.softmax(va_logits, T), CLASS_NAMES)
    per_class = ev.per_class_report(te_y, prob, CLASS_NAMES)
    cm = ev.confusion(te_y, prob, len(CLASS_NAMES))
    confused = ev.most_confused_pairs(cm, CLASS_NAMES)
    selective = ev.selective_prediction(prob, te_y)
    row_t = selective.iloc[(selective["threshold"] - threshold).abs().argmin()]
    metrics["app_threshold"] = threshold
    metrics["accuracy_above_threshold"] = float(row_t["accuracy"])
    metrics["coverage_above_threshold"] = float(row_t["coverage"])

    test = _subset(ctx, "test").reset_index(drop=True)
    pred_df = test[["rel_path", "class_name", "label"]].copy()
    pred_df["pred_label"] = prob.argmax(1)
    pred_df["pred_class"] = [CLASS_NAMES[k] for k in pred_df["pred_label"]]
    pred_df["confidence"] = prob.max(1)
    for k, c in enumerate(CLASS_NAMES):
        pred_df[f"p_{c}"] = prob[:, k]
    pred_df.to_csv(ws.metrics_dir / "test_predictions.csv", index=False)
    per_class.to_csv(ws.metrics_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(ws.metrics_dir / "confusion_matrix.csv")
    confused.to_csv(ws.metrics_dir / "most_confused_pairs.csv", index=False)
    selective.to_csv(ws.metrics_dir / "selective_prediction.csv", index=False)

    # ---- figures -------------------------------------------------------------------------
    plots.confusion_matrix_plot(cm, ws.fig("fig16_confusion_matrix.png"))
    plots.per_class_metrics(per_class, ws.fig("fig17_per_class_metrics.png"))
    plots.roc_pr_curves(te_y, prob, ws.fig("fig18_roc_pr_curves.png"))
    plots.reliability_diagram(ev.reliability_table(prob_raw, te_y), ev.reliability_table(prob, te_y),
                              metrics["ece_before_temperature"], metrics["ece"], T, ws.fig("fig19_reliability_diagram.png"))
    plots.confidence_analysis(prob, te_y, selective, threshold, ws.fig("fig20_confidence_analysis.png"))
    wrong = pred_df[pred_df["pred_label"] != pred_df["label"]].sort_values("confidence", ascending=False)
    items = [(test.loc[i, "path"], f"true: {r.class_name}\npred: {r.pred_class} ({r.confidence:.0%})", False)
             for i, r in wrong.head(18).iterrows()]
    plots.image_gallery(items, ws.fig("fig21_misclassified_examples.png"),
                        f"Misclassified test images ({len(wrong)} of {len(pred_df):,})",
                        subtitle="Sorted by the model's confidence in the wrong answer (most confident first)")

    # Grad-CAM on one correct image per class + the two most confident mistakes
    cam = GradCAM(model)
    rng = np.random.default_rng(cfg.seed)
    picks = []
    for k in range(len(CLASS_NAMES)):
        ok_idx = pred_df.index[(pred_df["label"] == k) & (pred_df["pred_label"] == k)]
        if len(ok_idx):
            picks.append(int(rng.choice(ok_idx)))
    picks += [int(i) for i in wrong.index[:2]]
    tf = build_transforms(cfg.img_size, train=False)
    batch = torch.stack([tf(load_rgb(test.loc[i, "path"])) for i in picks]).to(device)
    heat, pred_k = cam(batch)
    cam.remove()
    captions = [f"true {test.loc[i, 'class_name']} / pred {CLASS_NAMES[p]}" for i, p in zip(picks, pred_k)]
    plots.gradcam_grid(denormalise(batch, IMAGENET_MEAN, IMAGENET_STD), heat, captions, ws.fig("fig22_gradcam.png"))

    # Robustness to corruptions (subset of the test set)
    rob_idx = pd.concat([d.sample(min(len(d), robustness_max // len(CLASS_NAMES)), random_state=cfg.seed)
                         for _, d in test.groupby("label")]).index
    rows = []
    for name, fn in ev.corruption_transforms().items():
        preds = []
        for start in range(0, len(rob_idx), cfg.batch_size):
            chunk = rob_idx[start:start + cfg.batch_size]
            xb = torch.stack([tf(fn(load_rgb(test.loc[i, "path"]))) for i in chunk]).to(device)
            with torch.inference_mode():
                preds.append(model(xb).argmax(1).cpu().numpy())
        acc = float((np.concatenate(preds) == test.loc[rob_idx, "label"].to_numpy()).mean())
        rows.append({"corruption": name, "accuracy": acc})
    robustness = pd.DataFrame(rows)
    robustness.to_csv(ws.metrics_dir / "robustness.csv", index=False)
    plots.robustness_bars(robustness, ws.fig("fig23_robustness.png"))

    # Comparison with the baselines
    final_row = {"model": "EfficientNet-B0 fine-tuned (final)", "family": "transfer learning (fine-tuned)",
                 "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
                 "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
                 "fit_seconds": ctx.timings.get("train_final_s", np.nan)}
    comparison = pd.concat([ctx.baselines if ctx.baselines is not None else pd.DataFrame(),
                            pd.DataFrame([final_row])], ignore_index=True)
    comparison.to_csv(ws.metrics_dir / "model_comparison.csv", index=False)
    plots.model_comparison(comparison, ws.fig("fig24_model_comparison.png"), highlight=final_row["model"])

    metrics["val"] = val_metrics
    metrics["most_confused"] = confused.to_dict("records")
    save_json(metrics, ws.metrics_dir / "test_metrics.json")
    ctx.test_metrics = metrics
    ctx.timings["evaluate_s"] = time.time() - t0
    log(f"Test accuracy {metrics['accuracy']:.4f} (95% CI {metrics['accuracy_ci'][0]:.4f}-{metrics['accuracy_ci'][1]:.4f}), "
        f"macro-F1 {metrics['macro_f1']:.4f}, temperature {T:.2f}")
    log(per_class.round(4).to_string(index=False))
    return metrics


# =======================================================================================
# Step 3 - export the model for the application
# =======================================================================================
def export_model(ctx: Context, ws: Workspace, cfg=TrainConfig(), log=print) -> dict:
    import torch

    from . import evaluate as ev
    from .export import benchmark_latency, compress_weights_fp16, export_onnx, onnx_logits

    t0 = time.time()
    model = ctx.model.eval().cpu()
    fp32 = export_onnx(model, ws.cache_dir / "waste_classifier_fp32.onnx", cfg.img_size)
    final = compress_weights_fp16(fp32, ws.model_dir / "waste_classifier.onnx")
    test = _subset(ctx, "test").reset_index(drop=True)
    tf = build_transforms(cfg.img_size, train=False)
    check = test.sample(min(256, len(test)), random_state=cfg.seed)
    xb = torch.stack([tf(load_rgb(p)) for p in check["path"]])
    with torch.inference_mode():
        ref = model(xb).numpy()
    out32, out16 = onnx_logits(fp32, xb.numpy()), onnx_logits(final, xb.numpy())
    parity = {
        "max_abs_diff_fp32": float(np.abs(ref - out32).max()),
        "max_abs_diff_fp16_weights": float(np.abs(ref - out16).max()),
        "prediction_agreement_fp16_weights": float((ref.argmax(1) == out16.argmax(1)).mean()),
        "n_images_checked": int(len(check)),
    }
    latency = benchmark_latency(final, cfg.img_size)
    info = {
        "model_name": "EfficientNet-B0 (transfer learning, fine-tuned)",
        "class_names": CLASS_NAMES,
        "input_size": cfg.img_size,
        "resize": "shorter side to input_size (bicubic), then centre crop",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "temperature": float(ctx.temperature),
        "confidence_threshold": APP_THRESHOLD,
        "metrics": {k: ctx.test_metrics.get(k) for k in
                    ["accuracy", "macro_f1", "balanced_accuracy", "top2_accuracy", "roc_auc_ovr_macro", "ece",
                     "accuracy_ci", "macro_f1_ci", "n"]},
        "per_class": pd.read_csv(ws.metrics_dir / "per_class_metrics.csv").round(4).to_dict("records"),
        "onnx_file_mb": round(final.stat().st_size / 1e6, 2),
        "cpu_latency_ms": latency,
        "parity": parity,
        "trained_on": time.strftime("%Y-%m-%d"),
        "dataset": "Garbage Images Dataset (2000/class), Kaggle: zlatan599/garbage-dataset-classification",
        "environment": environment_info(),
    }
    save_json(info, ws.model_dir / "model_info.json")
    shutil.copy(ws.fig("fig16_confusion_matrix.png"), ws.model_dir / "confusion_matrix.png")

    # Sample images for the demo: two confidently-correct test images per class + hardest cases
    preds = pd.read_csv(ws.metrics_dir / "test_predictions.csv")
    preds["path"] = test["path"].to_numpy()
    for f in ws.sample_dir.glob("*"):
        f.unlink()
    for cls in CLASS_NAMES:
        ok = preds[(preds["class_name"] == cls) & (preds["pred_class"] == cls)].sort_values("confidence", ascending=False)
        for k, (_, r) in enumerate(ok.iloc[[0, len(ok) // 2]].iterrows() if len(ok) > 1 else ok.iterrows(), 1):
            shutil.copy(r.path, ws.sample_dir / f"{cls}_{k:02d}{Path(r.path).suffix.lower()}")
    hard = preds[preds["pred_class"] != preds["class_name"]].sort_values("confidence").head(2)
    for k, (_, r) in enumerate(hard.iterrows(), 1):
        shutil.copy(r.path, ws.sample_dir / f"hard_case_{k:02d}_true-{r.class_name}{Path(r.path).suffix.lower()}")
    ctx.timings["export_s"] = time.time() - t0
    log(f"ONNX model: {info['onnx_file_mb']} MB, CPU latency {latency['median_ms']:.1f} ms/image, parity {parity}")
    return info


# =======================================================================================
# Package everything for the report
# =======================================================================================
def package_results(ctx: Context, ws: Workspace, zip_path=None, log=print) -> Path:
    save_json({"timings_s": ctx.timings, "environment": environment_info()}, ws.metrics_dir / "run_info.json")
    zip_path = Path(zip_path or ws.out_dir.parent / "results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in [ws.fig_dir, ws.metrics_dir, ws.model_dir, ws.sample_dir]:
            for f in sorted(folder.rglob("*")):
                if f.is_file() and not f.name.endswith(".pt"):
                    zf.write(f, f.relative_to(ws.out_dir))
    log(f"Results packaged in {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def run_all(zip_path=None, data_dir=None, out_dir="outputs", device=None, workers=2, run_cnn=True,
            train_cfg=TrainConfig(), base_cfg=BaselineConfig(), split_cfg=SplitConfig(), log=print):
    """Run every step in order (used by scripts/run_pipeline.py)."""
    from .utils import get_device

    set_seed(train_cfg.seed)
    device = device or get_device()
    ws = Workspace(out_dir)
    root = acquire_data(zip_path, data_dir, extract_to=str(Path(out_dir) / "data"), log=log)
    ctx = data_understanding(root, ws, split_cfg, workers, log)
    visualise(ctx, ws, log)
    deep_embeddings(ctx, ws, device, workers=workers, log=log)
    tsne_map(ctx, ws, log=log)
    split_data(ctx, ws, split_cfg, log)
    baselines(ctx, ws, device, base_cfg, run_cnn=run_cnn, workers=workers, log=log)
    train_final(ctx, ws, device, train_cfg, workers, log)
    evaluate_final(ctx, ws, device, train_cfg, workers, log=log)
    export_model(ctx, ws, train_cfg, log)
    return package_results(ctx, ws, log=log)
