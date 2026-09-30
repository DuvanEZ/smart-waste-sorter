"""Hand-crafted image features for the classical machine-learning baselines.

Classical algorithms (k-NN, logistic regression, random forest, SVM) cannot use raw
pixels effectively, so each image is converted into a fixed-length feature vector:
  * colour  : joint HSV colour histogram (8 hue x 4 saturation x 4 value = 128 bins)
  * shape   : Histogram of Oriented Gradients (HOG) on a 128x128 grey image (1,764 values)
  * texture : Local Binary Pattern (LBP) histogram (10 values)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image


def handcrafted_features(path: str, size: int = 128) -> np.ndarray:
    import cv2
    from skimage.feature import hog, local_binary_pattern

    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    colour = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    colour = colour / max(colour.sum(), 1)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    shape = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                block_norm="L2-Hys", feature_vector=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    texture = np.histogram(lbp, bins=10, range=(0, 10))[0] / lbp.size
    return np.concatenate([colour, shape, texture]).astype(np.float32)


def _chunk(args):
    paths, size = args
    return np.stack([handcrafted_features(p, size) for p in paths])


def extract_features(paths, size: int = 128, workers: int = 2, chunk: int = 250) -> np.ndarray:
    paths = list(paths)
    jobs = [(paths[i:i + chunk], size) for i in range(0, len(paths), chunk)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_chunk, jobs))
    else:
        parts = [_chunk(job) for job in jobs]
    return np.concatenate(parts)
