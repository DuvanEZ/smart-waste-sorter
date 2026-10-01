"""Loading the image files, building the file index and creating the data splits."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import CLASS_NAMES, CLASS_TO_IDX, IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD


# --------------------------------------------------------------------------------------
# Locating and indexing the dataset
# --------------------------------------------------------------------------------------
def extract_zip(zip_path, dest_dir) -> Path:
    """Extract the Kaggle archive (skipped if it was already extracted)."""
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    marker = dest_dir / ".extracted"
    if marker.exists():
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    marker.write_text(zip_path.name)
    return dest_dir


def find_dataset_root(search_dir) -> Path:
    """Return the folder that contains one sub-folder per class (e.g. .../images)."""
    search_dir = Path(search_dir)
    wanted = set(CLASS_NAMES)
    candidates = [search_dir] + sorted(p for p in search_dir.rglob("*") if p.is_dir())
    for folder in candidates:
        subfolders = {p.name.lower() for p in folder.iterdir() if p.is_dir()}
        if len(subfolders & wanted) >= 4:  # tolerate a missing class, report it later
            return folder
    raise FileNotFoundError(
        f"No folder with class sub-folders {CLASS_NAMES} was found under {search_dir}"
    )


def build_index(root) -> pd.DataFrame:
    """One row per file: path, class name, integer label, extension and size on disk."""
    root = Path(root)
    rows = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name.lower()
        if class_name not in CLASS_TO_IDX:
            print(f"[warning] folder '{class_dir.name}' is not one of the expected classes - skipped")
            continue
        for file in sorted(class_dir.rglob("*")):
            if not file.is_file() or file.name.startswith("."):
                continue
            rows.append(
                {
                    "path": str(file),
                    "rel_path": file.relative_to(root).as_posix(),
                    "filename": file.name,
                    "class_name": class_name,
                    "label": CLASS_TO_IDX[class_name],
                    "ext": file.suffix.lower(),
                    "file_size_kb": file.stat().st_size / 1024,
                }
            )
    df = pd.DataFrame(rows)
    df["is_image_file"] = df["ext"].isin(IMAGE_EXTENSIONS)
    return df


def find_metadata_csv(root) -> Path | None:
    """The Kaggle archive ships a metadata.csv next to (or above) the image folders."""
    root = Path(root)
    for folder in [root, root.parent, root.parent.parent]:
        hits = sorted(folder.glob("*.csv"))
        if hits:
            return hits[0]
    return None


def compare_with_metadata(index: pd.DataFrame, metadata_path) -> dict:
    """Cross-check the folder structure against metadata.csv (labels and file names)."""
    meta = pd.read_csv(metadata_path)
    report = {"metadata_file": Path(metadata_path).name, "metadata_rows": len(meta),
              "metadata_columns": list(meta.columns)}
    # Guess which column holds the file name and which holds the label.
    file_col = next((c for c in meta.columns if meta[c].astype(str).str.contains(r"\.(?:jpe?g|png)$", case=False).mean() > 0.5), None)
    label_col = next((c for c in meta.columns if set(meta[c].astype(str).str.lower().unique()) <= set(CLASS_NAMES) | {"nan"}), None)
    report["file_column"], report["label_column"] = file_col, label_col
    if label_col is not None:
        report["metadata_counts"] = meta[label_col].astype(str).str.lower().value_counts().to_dict()
    if file_col is not None:
        meta_names = set(meta[file_col].astype(str).map(lambda s: Path(s).name))
        disk_names = set(index["filename"])
        report["files_in_metadata_not_on_disk"] = len(meta_names - disk_names)
        report["files_on_disk_not_in_metadata"] = len(disk_names - meta_names)
        if label_col is not None:
            merged = index.merge(
                meta.assign(_name=meta[file_col].astype(str).map(lambda s: Path(s).name),
                            _label=meta[label_col].astype(str).str.lower())[["_name", "_label"]],
                left_on="filename", right_on="_name", how="inner")
            report["label_agreement"] = float((merged["_label"] == merged["class_name"]).mean()) if len(merged) else None
    return report


# --------------------------------------------------------------------------------------
# Train / validation / test split
# --------------------------------------------------------------------------------------
def make_splits(labels, groups, test_size=0.15, val_size=0.15, seed=42) -> np.ndarray:
    """Stratified *group* split into train / val / test.

    * Stratified: every split keeps the class proportions of the full dataset.
    * Grouped: near-duplicate images share a group id and always land in the same
      split, so the test set never contains a copy of a training image (no leakage).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    labels, groups = np.asarray(labels), np.asarray(groups)
    n = len(labels)
    split = np.array(["train"] * n, dtype=object)

    n_test_folds = max(2, int(round(1 / test_size)))
    outer = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(np.zeros(n), labels, groups))
    split[test_idx] = "test"

    relative_val = val_size / (1 - len(test_idx) / n)
    n_val_folds = max(2, int(round(1 / relative_val)))
    inner = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=seed)
    _, val_rel = next(inner.split(np.zeros(len(trainval_idx)), labels[trainval_idx], groups[trainval_idx]))
    split[trainval_idx[val_rel]] = "val"
    return split


# --------------------------------------------------------------------------------------
# PyTorch dataset and image transformations
# --------------------------------------------------------------------------------------
def load_rgb(path) -> Image.Image:
    """Open an image, apply its EXIF orientation and convert it to 3-channel RGB."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


class RandomGaussianNoise:
    """Add camera-like Gaussian noise to a tensor in [0, 1] with probability `p`.

    The noise strength is drawn uniformly from [0, max_sigma] for every image, so the
    network learns to ignore the sensor noise of photos taken in poor light.
    """

    def __init__(self, p: float = 0.3, max_sigma: float = 0.08):
        self.p, self.max_sigma = p, max_sigma

    def __call__(self, x):
        import torch

        if torch.rand(1).item() < self.p:
            sigma = torch.rand(1).item() * self.max_sigma
            x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
        return x


def _augmentations(img_size: int):
    """Random transformations applied to training images (PIL stage)."""
    from torchvision import transforms as T

    return [
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=20, interpolation=T.InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),  # hue unchanged: colour is informative
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),  # slightly out-of-focus photos
    ]


def build_transforms(img_size: int = 224, train: bool = False):
    """Image transformations applied before the images enter the network.

    Evaluation: resize the shorter side to `img_size`, centre-crop a square,
    convert to a tensor in [0, 1] and normalise with the ImageNet statistics.
    Training: the same plus random data augmentation (crop, flips, rotation,
    brightness/contrast/saturation jitter, blur and sensor noise) so the model sees a
    slightly different version of every image in every epoch, which reduces
    over-fitting. The hue is not changed because colour carries class information
    (brown cardboard, green glass).
    """
    from torchvision import transforms as T

    bicubic = T.InterpolationMode.BICUBIC
    normalise = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([*_augmentations(img_size), T.ToTensor(), RandomGaussianNoise(p=0.3, max_sigma=0.08), normalise])
    return T.Compose([T.Resize(img_size, interpolation=bicubic), T.CenterCrop(img_size), T.ToTensor(), normalise])


def augmentation_preview(img_size: int = 224):
    """Training augmentation without normalisation (used only to plot examples)."""
    from torchvision import transforms as T

    return T.Compose([*_augmentations(img_size), T.ToTensor(), RandomGaussianNoise(p=0.3, max_sigma=0.08),
                      T.ToPILImage()])


def make_dataset(paths, labels, transform):
    """Create a torch Dataset lazily (keeps torch optional for the analysis part)."""
    import torch

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = list(paths)
            self.labels = np.asarray(labels, dtype=np.int64)
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return self.transform(load_rgb(self.paths[i])), int(self.labels[i])

    return ImageDataset(paths, labels, transform)


def make_loader(dataset, batch_size, shuffle, num_workers=2, device_type="cpu"):
    import torch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def copy_tree(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)
