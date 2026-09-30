"""Shared fixtures: make the app modules importable and build a tiny ONNX model."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@pytest.fixture()
def tiny_model_dir(tmp_path):
    """A minimal ONNX classifier (global average pool + linear layer) with the same
    input/output signature as the real model, so the predictor can be tested offline."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(size=(6, 3)).astype(np.float32), "W")
    b = numpy_helper.from_array(np.zeros(6, dtype=np.float32), "B")
    nodes = [helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
             helper.make_node("Flatten", ["pooled"], ["flat"]),
             helper.make_node("Gemm", ["flat", "W", "B"], ["logits"], transB=1)]
    graph = helper.make_graph(nodes, "tiny",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 6])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, tmp_path / "waste_classifier.onnx")
    (tmp_path / "model_info.json").write_text(json.dumps({
        "class_names": CLASSES, "input_size": 224, "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225], "temperature": 1.0, "confidence_threshold": 0.6}))
    return tmp_path
