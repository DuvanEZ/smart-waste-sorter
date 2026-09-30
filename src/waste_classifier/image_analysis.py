"""Per-image properties used for the data analysis (Part 1 of the report).

For every image we record
  * file properties  : format, colour mode, width, height, frames, integrity, MD5
  * perceptual hash  : 64-bit pHash used to find (near-)duplicate images
  * pixel statistics : mean R/G/B, brightness, contrast, saturation, colourfulness,
                       sharpness, entropy, edge density, background whiteness,
                       hue distribution
These "variables" describe each image numerically, so they can be summarised,
visualised and compared between classes.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

HUE_BINS = 18          # 18 bins of 20 degrees each
CHANNEL_BINS = 32      # intensity histogram bins per colour channel
THUMB_SIZE = 64        # size of the thumbnails averaged into the "mean image" of a class
NUMERIC_FEATURES = [
    "file_size_kb", "mean_r", "mean_g", "mean_b", "brightness", "contrast", "saturation",
    "colourfulness", "sharpness", "entropy", "edge_density", "white_background",
]


def _phash_hex(img: Image.Image) -> str:
    """64-bit perceptual hash as a 16-character hex string (kept as text so that
    pandas never converts it to a float and loses precision)."""
    import imagehash

    return str(imagehash.phash(img))


def hashes_to_uint64(hex_hashes) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def analyse_image(path: str) -> tuple[dict, np.ndarray | None]:
    """Return (properties, 64x64 thumbnail) for one image file."""
    import cv2

    row = {"path": path}
    try:
        raw = open(path, "rb").read()
        row["md5"] = hashlib.md5(raw).hexdigest()
        with Image.open(path) as probe:          # integrity check without decoding
            row.update(format=probe.format, mode=probe.mode,
                       width=probe.size[0], height=probe.size[1],
                       n_frames=getattr(probe, "n_frames", 1))
            probe.verify()
        with Image.open(path) as img:             # full decode
            img = ImageOps.exif_transpose(img).convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
            row["phash"] = _phash_hex(img)
            thumb = np.asarray(img.resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR), dtype=np.uint8)
        row["readable"] = True
        row["error"] = ""
    except Exception as exc:  # corrupted / unreadable file
        row.update(readable=False, error=f"{type(exc).__name__}: {exc}")
        return row, None

    rgb = arr.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b                      # perceived brightness
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)                   # H: 0-179, S/V: 0-255
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    row["mean_r"], row["mean_g"], row["mean_b"] = float(r.mean()), float(g.mean()), float(b.mean())
    row["brightness"] = float(lum.mean() / 255)
    row["contrast"] = float(lum.std() / 255)
    row["saturation"] = float(hsv[..., 1].mean() / 255)
    # Colourfulness metric of Hasler & Suesstrunk (2003).
    rg, yb = r - g, 0.5 * (r + g) - b
    row["colourfulness"] = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    # Sharpness: variance of the Laplacian (low values = blurry image).
    row["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Shannon entropy of the grey-level histogram (amount of detail / texture).
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    nz = hist[hist > 0]
    row["entropy"] = float(-(nz * np.log2(nz)).sum())
    # Edge density: share of pixels on a Canny edge.
    row["edge_density"] = float((cv2.Canny(gray, 100, 200) > 0).mean())
    # Background whiteness: share of near-white, low-saturation pixels in a 10-px border.
    border = np.zeros(gray.shape, dtype=bool)
    border[:10, :] = border[-10:, :] = border[:, :10] = border[:, -10:] = True
    white = (lum > 200) & (hsv[..., 1] < 40)
    row["white_background"] = float(white[border].mean())
    # Hue histogram of clearly coloured pixels (share of all pixels in each 20-degree bin).
    coloured = (hsv[..., 1] >= 40) & (hsv[..., 2] >= 40)
    hue_counts = np.bincount((hsv[..., 0][coloured] // 10).astype(int), minlength=HUE_BINS)[:HUE_BINS]
    for i, c in enumerate(hue_counts):
        row[f"hue_{i:02d}"] = float(c / gray.size)
    # Intensity histogram of each colour channel (32 bins of 8 grey levels).
    for ch, name in enumerate("rgb"):
        counts = np.bincount((arr[..., ch] >> 3).ravel(), minlength=CHANNEL_BINS) / gray.size
        for i, c in enumerate(counts):
            row[f"{name}_hist_{i:02d}"] = float(c)
    return row, thumb


def _analyse_chunk(args):
    """Worker: analyse a chunk of images and return rows + per-class thumbnail sums."""
    paths, labels, n_classes = args
    rows, sums = [], np.zeros((n_classes, THUMB_SIZE, THUMB_SIZE, 3), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    for path, label in zip(paths, labels):
        row, thumb = analyse_image(path)
        rows.append(row)
        if thumb is not None:
            sums[label] += thumb
            counts[label] += 1
    return rows, sums, counts


def analyse_dataset(index: pd.DataFrame, n_classes: int, workers: int = 2, chunk: int = 250, log=print):
    """Analyse every image (in parallel) and compute the mean image of each class."""
    paths, labels = index["path"].tolist(), index["label"].to_numpy()
    jobs = [(paths[i:i + chunk], labels[i:i + chunk], n_classes) for i in range(0, len(paths), chunk)]
    all_rows = []
    sums = np.zeros((n_classes, THUMB_SIZE, THUMB_SIZE, 3), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_analyse_chunk, jobs)
            for k, (rows, s, c) in enumerate(results, 1):
                all_rows += rows; sums += s; counts += c
                if k % 10 == 0 or k == len(jobs):
                    log(f"  analysed {min(k * chunk, len(paths))}/{len(paths)} images")
    else:
        for k, job in enumerate(jobs, 1):
            rows, s, c = _analyse_chunk(job)
            all_rows += rows; sums += s; counts += c
    stats = pd.DataFrame(all_rows)
    stats = index.merge(stats, on="path", how="left")
    mean_images = (sums / np.maximum(counts, 1)[:, None, None, None]).astype(np.uint8)
    return stats, mean_images


# --------------------------------------------------------------------------------------
# Duplicate detection
# --------------------------------------------------------------------------------------
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):          # NumPy >= 2.0
        return np.bitwise_count(x).astype(np.uint8)
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1).astype(np.uint8)


def near_duplicates(hashes, threshold: int = 4, block: int = 512):
    """Compare every pair of 64-bit pHashes.

    Returns
      pairs   : DataFrame (i, j, distance) with distance <= threshold (i < j)
      nn_dist : for each image, the Hamming distance to its most similar other image
    """
    h = np.asarray(hashes, dtype=np.uint64)
    n = len(h)
    nn_dist = np.full(n, 64, dtype=np.uint8)
    found = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        d = _popcount64(np.bitwise_xor(h[start:stop, None], h[None, :]))
        d[np.arange(stop - start), np.arange(start, stop)] = 255      # ignore self-match
        nn_dist[start:stop] = d.min(axis=1)
        ii, jj = np.nonzero(d <= threshold)
        ii = ii + start
        keep = jj > ii
        found.append(np.stack([ii[keep], jj[keep], d[ii[keep] - start, jj[keep]]], axis=1))
    pairs = np.concatenate(found) if found else np.empty((0, 3), dtype=int)
    return pd.DataFrame(pairs, columns=["i", "j", "distance"]).astype(int), nn_dist


def duplicate_groups(n: int, pairs: pd.DataFrame) -> np.ndarray:
    """Union-find: images connected by near-duplicate pairs get the same group id."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(pairs["i"].to_numpy(), pairs["j"].to_numpy()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(a) for a in range(n)])
