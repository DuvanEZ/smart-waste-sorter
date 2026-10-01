"""Project-wide configuration.

Every value that controls the experiment lives here, so the data analysis,
the training code, the Colab notebook and the application all agree on the
class names, the image size, the normalisation constants and the random seed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# Fixed random seed -> reproducible splits, sampling and training.
SEED = 42

# The six material classes in alphabetical order.
# The integer label of a class is its position in this list (label encoding).
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# File extensions that are treated as images.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# ImageNet channel statistics. The backbone network was pre-trained on ImageNet
# images normalised with these values, so our images must be normalised the same way.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Input resolution of the network (EfficientNet-B0 native resolution).
IMG_SIZE = 224

# One fixed colour per class, used in every figure (colour follows the class).
# Categorical palette validated for colour-vision deficiency (adjacent pairs).
CLASS_COLORS = {
    "cardboard": "#2a78d6",
    "glass": "#eb6834",
    "metal": "#1baf7a",
    "paper": "#eda100",
    "plastic": "#e87ba4",
    "trash": "#008300",
}


@dataclass
class SplitConfig:
    """How the data are divided into training, validation and test sets."""

    test_size: float = 0.15      # share of images held out for the final test
    val_size: float = 0.15       # share of images used for model selection
    dup_threshold: int = 4       # pHash Hamming distance that defines a near-duplicate
    seed: int = SEED


@dataclass
class TrainConfig:
    """Hyper-parameters of the final model (EfficientNet-B0, transfer learning)."""

    model_name: str = "efficientnet_b0"
    img_size: int = IMG_SIZE
    batch_size: int = 64
    num_workers: int = 2
    # Phase 1 trains only the new classification head (backbone frozen).
    head_epochs: int = 3
    head_lr: float = 1e-3
    # Phase 2 fine-tunes the whole network with a smaller learning rate.
    finetune_epochs: int = 15
    finetune_lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    drop_rate: float = 0.3        # dropout before the classifier
    drop_path_rate: float = 0.1   # stochastic depth inside the backbone
    patience: int = 4             # early stopping patience (epochs without improvement)
    seed: int = SEED

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BaselineConfig:
    """Settings of the comparison (baseline) models."""

    handcrafted_size: int = 128   # images are resized to 128x128 for HOG/colour/LBP features
    pca_components: int = 150
    cnn_img_size: int = 128       # small CNN trained from scratch
    cnn_epochs: int = 15
    cnn_lr: float = 1e-3
    cnn_batch_size: int = 64
    seed: int = SEED
