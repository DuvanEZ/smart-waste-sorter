"""Where did the images come from? Empirical provenance check.

The Kaggle dataset was assembled from several public garbage datasets, but the
notebook that built it is private. This module downloads candidate source datasets,
computes a perceptual hash (pHash) of each of their images and looks for matching
hashes among our images. An image whose hash is within a small Hamming distance of a
source image is attributed to that source.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import CLASS_NAMES
from .image_analysis import _popcount64, hashes_to_uint64

# Candidate sources (all public on Kaggle; cited in the report)
# Older result files used the label "Garbage Classification v2 (Suman, 2024)" for the third source.
OLD_SOURCE_NAMES = {"Garbage Classification v2 (Suman, 2024)": "Garbage Dataset v2 (Kunwar, 2026)"}
SOURCES = [
    {"name": "TrashNet (Thung & Yang, 2016)", "handle": "asdasdasasdas/garbage-classification"},
    {"name": "Garbage Classification 12 classes (Mohamed, 2021)", "handle": "mostafaabla/garbage-classification"},
    {"name": "Garbage Dataset v2 (Kunwar, 2026)", "handle": "sumn2u/garbage-classification-v2"},
]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _hash_variants(path: Path) -> list[int]:
    """pHash of the image squashed to a square and of its centre crop."""
    import imagehash

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        crop = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return [str(imagehash.phash(im.resize((256, 256)))), str(imagehash.phash(crop.resize((256, 256))))]


def _hash_chunk(paths):
    rows = []
    for p in paths:
        try:
            h1, h2 = _hash_variants(Path(p))
        except Exception:  # unreadable source file
            continue
        rows.append({"path": str(p), "source_class": Path(p).parent.name.lower(), "h_squash": h1, "h_crop": h2})
    return rows


def hash_source(root: Path, max_images: int | None = None, workers: int = 2, log=print) -> pd.DataFrame:
    """Perceptual hashes of every image of a candidate source data set (in parallel)."""
    from concurrent.futures import ProcessPoolExecutor

    files = sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if max_images:
        files = files[:max_images]
    chunks = [files[i:i + 500] for i in range(0, len(files), 500)]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for part in pool.map(_hash_chunk, chunks):
            rows += part
    log(f"    hashed {len(rows):,} images")
    return pd.DataFrame(rows)


def min_distance(query: np.ndarray, ref: np.ndarray, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """For every query hash: smallest Hamming distance to any reference hash and its index."""
    best = np.full(len(query), 64, dtype=np.int64)
    arg = np.zeros(len(query), dtype=np.int64)
    for start in range(0, len(query), block):
        d = _popcount64(np.bitwise_xor(query[start:start + block, None], ref[None, :])).astype(np.int64)
        best[start:start + block] = d.min(axis=1)
        arg[start:start + block] = d.argmin(axis=1)
    return best, arg


def check_provenance(stats: pd.DataFrame, out_dir, threshold: int = 6, workers: int = 2, log=print) -> pd.DataFrame:
    """Attribute every image of `stats` (needs 'phash' and 'class_name') to a source dataset."""
    import kagglehub

    out_dir = Path(out_dir)
    ours = stats[stats["readable"].fillna(False).astype(bool)].reset_index(drop=True)
    q = hashes_to_uint64(ours["phash"])
    result = ours[["rel_path", "class_name"]].copy()
    result["source"], result["distance"], result["source_class"] = "unidentified", 64, ""
    for src in SOURCES:
        try:
            log(f"  downloading {src['handle']} ...")
            root = Path(kagglehub.dataset_download(src["handle"]))
            ref = hash_source(root, workers=workers, log=log)
        except Exception as exc:
            log(f"  could not use {src['handle']}: {exc}")
            continue
        ref_hashes = np.concatenate([hashes_to_uint64(ref["h_squash"]), hashes_to_uint64(ref["h_crop"])])
        ref_class = np.concatenate([ref["source_class"].to_numpy(), ref["source_class"].to_numpy()])
        d, a = min_distance(q, ref_hashes)
        better = (d <= threshold) & (d < result["distance"].to_numpy())
        result.loc[better, "source"] = src["name"]
        result.loc[better, "distance"] = d[better]
        result.loc[better, "source_class"] = ref_class[a[better]]
        log(f"  {src['name']}: {len(ref):,} source images, {(d <= threshold).sum():,} of our images match")
    result.to_csv(out_dir / "provenance_per_image.csv", index=False)
    summary = result.groupby(["class_name", "source"]).size().unstack("source", fill_value=0).reindex(CLASS_NAMES)
    summary.loc["total"] = summary.sum()
    summary.to_csv(out_dir / "provenance_summary.csv")
    return result
