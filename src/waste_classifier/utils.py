"""Small helpers shared by the whole project (seeding, device, JSON, timing)."""
from __future__ import annotations

import json
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Make Python, NumPy and PyTorch random generators reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is not needed for the data-analysis part
        pass


def get_device():
    """Return the fastest available PyTorch device: CUDA GPU, Apple MPS or CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_builtin(obj):
    """Convert NumPy / pandas scalars and arrays into JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_builtin(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and obj != obj:  # NaN -> null
        return None
    return obj


def save_json(data, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_builtin(data), indent=2), encoding="utf-8")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def timer(label: str, log=print):
    """Context manager that logs how long a block of code took."""
    start = time.perf_counter()
    yield
    log(f"[time] {label}: {time.perf_counter() - start:.1f} s")


def environment_info() -> dict:
    """Versions of the main libraries (reported in the notebook and the report)."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["numpy", "pandas", "sklearn", "skimage", "cv2", "PIL", "matplotlib",
                 "torch", "torchvision", "timm", "onnx", "onnxruntime", "imagehash"]:
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception:  # library not installed in this environment
            info[name] = "not installed"
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info
