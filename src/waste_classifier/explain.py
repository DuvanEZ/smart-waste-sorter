"""Grad-CAM (Selvaraju et al., 2017): which image regions drove a prediction."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Class-activation heat-map from the gradients of the last convolutional layer."""

    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        layer = target_layer if target_layer is not None else model.conv_head  # EfficientNet (timm)
        self.activations = None
        self.gradients = None
        self._handle = layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:  # no gradient hooks during normal (inference) use
            output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def remove(self):
        """Detach the hook so the model behaves normally afterwards."""
        self._handle.remove()

    def __call__(self, x: torch.Tensor, class_idx=None) -> tuple[np.ndarray, np.ndarray]:
        """x: (N,3,H,W) normalised batch. Returns (heat-maps N x H x W in [0,1], predicted classes)."""
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            pred = logits.argmax(1)
            target = pred if class_idx is None else torch.as_tensor(class_idx, device=x.device)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, target.view(-1, 1)).sum().backward()
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # importance of each channel
            cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            cam = cam - cam.amin(dim=(1, 2), keepdim=True)
            cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return cam.detach().cpu().numpy(), pred.detach().cpu().numpy()


def denormalise(x: torch.Tensor, mean, std) -> np.ndarray:
    """Undo ImageNet normalisation -> (N,H,W,3) float image in [0,1] for plotting."""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    img = (x.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(0, 2, 3, 1).numpy()
