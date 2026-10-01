"""Export the trained PyTorch model to ONNX so the application does not need PyTorch."""
from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np
import torch


def export_onnx(model, out_path, img_size: int = 224, opset: int = 17) -> Path:
    """Write the model as an ONNX graph with a dynamic batch dimension."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # classic exporter: simple, well-tested graphs for CNNs
    try:
        torch.onnx.export(model, (dummy,), str(out_path), **kwargs)
    except Exception as exc:  # newer PyTorch without the classic exporter -> dynamo exporter
        print(f"[export] classic exporter failed ({type(exc).__name__}); retrying with dynamo=True")
        kwargs.pop("do_constant_folding", None)
        kwargs["dynamo"] = True
        kwargs["opset_version"] = max(opset, 18)
        torch.onnx.export(model, (dummy,), str(out_path), **kwargs)
    _merge_external_data(out_path)
    return out_path


def _merge_external_data(path: Path) -> None:
    """Make sure all weights are stored inside the .onnx file (single-file model)."""
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    onnx.save_model(model, str(path), save_as_external_data=False)
    for extra in path.parent.glob(path.name + ".data"):
        extra.unlink()


def compress_weights_fp16(src, dst, min_elements: int = 1024) -> Path:
    """Store large weight tensors as float16 (half the file size) + Cast nodes back to float32.

    ONNX Runtime folds the Cast nodes when the model is loaded, so inference still runs
    in float32; only the rounding of the stored weights changes (verified afterwards).
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph
    graph_inputs = {i.name for i in graph.input}
    casts, new_inits = [], []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT or init.name in graph_inputs:
            continue
        if int(np.prod(init.dims)) < min_elements:
            continue
        arr = numpy_helper.to_array(init).astype(np.float16)
        half_name = init.name + "__fp16"
        new_inits.append(numpy_helper.from_array(arr, half_name))
        casts.append(helper.make_node("Cast", [half_name], [init.name], to=TensorProto.FLOAT,
                                      name=init.name + "__to_fp32"))
        graph.initializer.remove(init)
    graph.initializer.extend(new_inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(dst))
    return Path(dst)


def onnx_logits(onnx_path, batch: np.ndarray, chunk: int = 32) -> np.ndarray:
    """Logits of an ONNX model for a batch of images, computed in small chunks to limit memory."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False      # return memory after each run (Colab has ~12 GB RAM)
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = [session.run(None, {name: batch[i:i + chunk].astype(np.float32)})[0] for i in range(0, len(batch), chunk)]
    del session
    return np.concatenate(out)


def benchmark_latency(onnx_path, img_size: int = 224, runs: int = 30) -> dict:
    """Median CPU latency for one image (what the lecturer will experience)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(3):
        session.run(None, {name: x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {name: x})
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}
