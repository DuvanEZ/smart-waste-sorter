"""Prediction engine of the application.

Loads the trained model (ONNX format, see model/) and turns an image into a
prediction. Only NumPy, Pillow and ONNX Runtime are needed - no PyTorch and no GPU.
The pre-processing below reproduces exactly the evaluation transform used during
training: resize the shorter side to 224 px (bicubic), centre-crop 224 x 224,
scale to [0, 1] and normalise with the ImageNet mean and standard deviation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"

try:  # Pillow >= 9.1
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # older Pillow
    BICUBIC = Image.BICUBIC


@dataclass
class Prediction:
    """Result for one image."""

    label: str                          # predicted class name
    confidence: float                   # probability of the predicted class (0-1)
    probabilities: dict[str, float]     # probability of every class
    top_k: list[tuple[str, float]]      # classes sorted from most to least likely
    is_confident: bool                  # confidence >= threshold
    threshold: float = 0.6
    warnings: list[str] = field(default_factory=list)


class WasteClassifier:
    """Wraps the ONNX model and its metadata (class names, normalisation, temperature)."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        info_path = model_dir / "model_info.json"
        onnx_path = model_dir / "waste_classifier.onnx"
        if not onnx_path.exists() or not info_path.exists():
            raise FileNotFoundError(
                f"Model files not found in '{model_dir}'. Expected 'waste_classifier.onnx' and 'model_info.json'."
            )
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.class_names: list[str] = self.info["class_names"]
        self.input_size: int = int(self.info.get("input_size", 224))
        self.mean = np.array(self.info.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(self.info.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.temperature: float = float(self.info.get("temperature", 1.0))
        self.default_threshold: float = float(self.info.get("confidence_threshold", 0.6))
        options = ort.SessionOptions()
        options.log_severity_level = 3  # only errors
        self.session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------ pre-processing
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """PIL image -> normalised float32 array of shape (3, H, W)."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        size = self.input_size
        w, h = image.size
        # Resize so that the shorter side equals `size` (same rule as torchvision.Resize(int)).
        if w <= h:
            new_w, new_h = size, int(size * h / w)
        else:
            new_w, new_h = int(size * w / h), size
        image = image.resize((new_w, new_h), BICUBIC)
        # Centre crop (same rounding as torchvision.CenterCrop).
        top = int(round((new_h - size) / 2.0))
        left = int(round((new_w - size) / 2.0))
        image = image.crop((left, top, left + size, top + size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return arr.transpose(2, 0, 1)

    # ------------------------------------------------------------------ inference
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.temperature          # temperature scaling -> calibrated probabilities
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """Class probabilities for a list of images, shape (N, n_classes)."""
        out = []
        for start in range(0, len(images), batch_size):
            batch = np.stack([self.preprocess(img) for img in images[start:start + batch_size]])
            logits = self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]
            out.append(self._softmax(logits))
        return np.concatenate(out) if out else np.zeros((0, len(self.class_names)), dtype=np.float32)

    def predict(self, image: Image.Image, threshold: float | None = None) -> Prediction:
        return self.predict_batch([image], threshold)[0]

    def predict_batch(self, images: list[Image.Image], threshold: float | None = None) -> list[Prediction]:
        threshold = self.default_threshold if threshold is None else float(threshold)
        probs = self.predict_proba(images)
        results = []
        for p in probs:
            order = np.argsort(-p)
            top = [(self.class_names[i], float(p[i])) for i in order]
            results.append(Prediction(
                label=top[0][0],
                confidence=top[0][1],
                probabilities={c: float(v) for c, v in zip(self.class_names, p)},
                top_k=top,
                is_confident=top[0][1] >= threshold,
                threshold=threshold,
            ))
        return results

    # ------------------------------------------------------------------ metadata
    @property
    def test_accuracy(self) -> float | None:
        return (self.info.get("metrics") or {}).get("accuracy")

    @property
    def test_macro_f1(self) -> float | None:
        return (self.info.get("metrics") or {}).get("macro_f1")
