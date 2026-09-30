"""Create a tiny synthetic dataset with the same folder layout as the Kaggle archive.

Used only to test the pipeline end-to-end without the real data:
    python scripts/make_synthetic_dataset.py --out /tmp/synthetic --per-class 40
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
BASE = {"cardboard": (160, 120, 80), "glass": (60, 150, 90), "metal": (150, 150, 160),
        "paper": (235, 235, 225), "plastic": (80, 140, 220), "trash": (90, 70, 60)}


def make_image(cls, rng):
    bg = tuple(int(v) for v in rng.integers(200, 256, 3)) if rng.random() < 0.5 else tuple(int(v) for v in rng.integers(40, 200, 3))
    img = Image.new("RGB", (256, 256), bg)
    d = ImageDraw.Draw(img)
    colour = tuple(int(np.clip(c + rng.normal(0, 25), 0, 255)) for c in BASE[cls])
    x0, y0 = rng.integers(20, 90, 2)
    x1, y1 = x0 + rng.integers(90, 150), y0 + rng.integers(90, 150)
    shape = CLASSES.index(cls) % 3
    if shape == 0:
        d.rectangle([x0, y0, x1, y1], fill=colour)
    elif shape == 1:
        d.ellipse([x0, y0, x1, y1], fill=colour)
    else:
        d.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=colour)
    for _ in range(int(rng.integers(3, 12))):
        a, b = rng.integers(0, 256, 2)
        d.line([a, b, a + rng.integers(-40, 40), b + rng.integers(-40, 40)], fill=tuple(int(v) for v in rng.integers(0, 256, 3)), width=2)
    return img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0, 1.2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/synthetic_garbage")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    root = Path(args.out) / "Garbage_Dataset_Classification"
    if root.exists():
        shutil.rmtree(root)
    rows = []
    for cls in CLASSES:
        folder = root / "images" / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            name = f"{cls}_{i:05d}.jpg"
            make_image(cls, rng).save(folder / name, quality=90)
            rows.append((name, cls))
    # a duplicate inside a class, a duplicate across classes and a corrupted file (quality checks)
    shutil.copy(root / "images/glass/glass_00000.jpg", root / "images/glass/glass_dup.jpg")
    shutil.copy(root / "images/paper/paper_00001.jpg", root / "images/cardboard/cardboard_conflict.jpg")
    (root / "images/metal/metal_broken.jpg").write_bytes(b"not really a jpeg")
    rows += [("glass_dup.jpg", "glass"), ("cardboard_conflict.jpg", "cardboard"), ("metal_broken.jpg", "metal")]
    with open(root / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        w.writerows(rows)
    print(f"Synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
