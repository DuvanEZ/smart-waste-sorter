"""Neural-network architectures: EfficientNet-B0 (transfer learning) and a small CNN."""
from __future__ import annotations

import os

import torch
from torch import nn

# Optional local copy of the ImageNet weights (used when the Hugging Face hub is not reachable).
WEIGHTS_ENV = "EFFICIENTNET_B0_WEIGHTS"


def _pretrained_kwargs(pretrained: bool) -> dict:
    kwargs = {"pretrained": pretrained}
    weights_file = os.environ.get(WEIGHTS_ENV)
    if pretrained and weights_file:
        kwargs["pretrained_cfg_overlay"] = {"file": weights_file}
    return kwargs


def create_efficientnet(num_classes: int, pretrained: bool = True,
                        drop_rate: float = 0.3, drop_path_rate: float = 0.1) -> nn.Module:
    """EfficientNet-B0 pre-trained on ImageNet with a new `num_classes` output layer.

    The convolutional backbone (feature extractor) keeps its ImageNet weights; only the
    final fully-connected layer is replaced, because ImageNet has 1,000 classes and our
    problem has six.
    """
    import timm

    model = timm.create_model("efficientnet_b0", num_classes=num_classes, drop_rate=drop_rate,
                              drop_path_rate=drop_path_rate, **_pretrained_kwargs(pretrained))
    # Start the new output layer at zero: every class starts with the same score (loss = ln 6)
    # instead of large random logits, so the head converges quickly in phase 1.
    head = model.get_classifier()
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return model


def create_feature_extractor(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 without its classifier: returns a 1,280-d embedding per image."""
    import timm

    return timm.create_model("efficientnet_b0", num_classes=0, **_pretrained_kwargs(pretrained))


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (or unfreeze) every layer except the classification head."""
    head = model.get_classifier()
    head_params = {id(p) for p in head.parameters()}
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable
    for p in head.parameters():
        p.requires_grad = True


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class SimpleCNN(nn.Module):
    """A small VGG-style CNN trained from scratch (baseline without transfer learning).

    4 blocks of [conv 3x3 -> batch-norm -> ReLU] x 2 + max-pool, followed by global
    average pooling, dropout and a linear classifier (~1.2 M parameters).
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


@torch.inference_mode()
def extract_embeddings(model: nn.Module, loader, device) -> tuple:
    """Run a frozen feature extractor over a loader and return (embeddings, labels)."""
    import numpy as np

    model.eval().to(device)
    feats, labels = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model(x.to(device, non_blocking=True))
        feats.append(f.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)
