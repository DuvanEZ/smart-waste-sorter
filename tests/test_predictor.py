"""Prediction engine: probabilities, thresholds and pre-processing parity with training."""
import numpy as np
import pytest
from PIL import Image

from predictor import WasteClassifier


def test_prediction_structure(tiny_model_dir):
    clf = WasteClassifier(tiny_model_dir)
    img = Image.new("RGB", (300, 200), (10, 200, 30))
    p = clf.predict(img, threshold=0.6)
    assert p.label in clf.class_names
    assert abs(sum(p.probabilities.values()) - 1) < 1e-5
    assert p.top_k[0][0] == p.label and p.top_k[0][1] >= p.top_k[1][1]
    assert p.is_confident == (p.confidence >= 0.6)


def test_batch_prediction(tiny_model_dir):
    clf = WasteClassifier(tiny_model_dir)
    imgs = [Image.new("RGB", (256, 256), c) for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]]
    probs = clf.predict_proba(imgs)
    assert probs.shape == (3, 6) and np.allclose(probs.sum(1), 1, atol=1e-5)


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        WasteClassifier(tmp_path)


def test_preprocessing_matches_training_transform(tiny_model_dir):
    """The app must feed the network exactly what it saw during evaluation."""
    T = pytest.importorskip("torchvision.transforms")
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (300, 420, 3), dtype=np.uint8))
    ref = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(img).numpy()
    ours = WasteClassifier(tiny_model_dir).preprocess(img)
    assert ours.shape == ref.shape == (3, 224, 224)
    assert np.abs(ours - ref).max() < 1e-4
