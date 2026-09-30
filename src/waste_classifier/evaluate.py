"""Performance evaluation: metrics, confidence intervals, calibration and robustness."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average gap between confidence and accuracy, weighted by bin size."""
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def reliability_table(prob, y, n_bins: int = 10) -> pd.DataFrame:
    conf, pred = prob.max(1), prob.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"bin_low": lo, "bin_high": hi, "count": int(m.sum()),
                     "confidence": float(conf[m].mean()) if m.any() else np.nan,
                     "accuracy": float((pred[m] == y[m]).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T that minimises the
    validation negative log-likelihood. It changes confidence, never the predicted class."""
    from scipy.optimize import minimize_scalar

    def nll(log_t):
        p = softmax(logits, np.exp(log_t))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()

    res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def classification_metrics(y: np.ndarray, prob: np.ndarray, class_names) -> dict:
    """Headline metrics for a multi-class classifier."""
    pred = prob.argmax(1)
    labels = np.arange(len(class_names))
    top2 = np.argsort(-prob, axis=1)[:, :2]
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y, pred, labels=labels,
                                                                   average="macro", zero_division=0)
    metrics = {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_f1": f1_score(y, pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "top2_accuracy": float((top2 == y[:, None]).any(1).mean()),
        "log_loss": log_loss(y, np.clip(prob, 1e-12, 1), labels=labels),
        "ece": expected_calibration_error(prob, y),
    }
    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError:  # a class is missing (only happens on tiny test runs)
        metrics["roc_auc_ovr_macro"] = float("nan")
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in metrics.items()}


def per_class_report(y, prob, class_names) -> pd.DataFrame:
    pred = prob.argmax(1)
    labels = np.arange(len(class_names))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f, "support": s})
    aucs = []
    for k in labels:
        try:
            aucs.append(roc_auc_score((y == k).astype(int), prob[:, k]))
        except ValueError:
            aucs.append(np.nan)
    df["roc_auc"] = aucs
    return df


def bootstrap_ci(y, pred, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """95% bootstrap confidence intervals for accuracy and macro-F1 on the test set."""
    rng = np.random.default_rng(seed)
    n = len(y)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y[idx], pred[idx]))
        f1s.append(f1_score(y[idx], pred[idx], average="macro"))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"accuracy_ci": np.percentile(accs, q).tolist(), "macro_f1_ci": np.percentile(f1s, q).tolist()}


def confusion(y, prob, n_classes) -> np.ndarray:
    return confusion_matrix(y, prob.argmax(1), labels=np.arange(n_classes))


def most_confused_pairs(cm: np.ndarray, class_names, k: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                rows.append({"true": class_names[i], "predicted": class_names[j], "count": int(cm[i, j]),
                             "share_of_true_class": cm[i, j] / cm[i].sum()})
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(k).reset_index(drop=True)


def selective_prediction(prob, y, thresholds=None) -> pd.DataFrame:
    """Accuracy vs coverage when predictions below a confidence threshold are rejected."""
    thresholds = np.round(np.arange(0.0, 1.0, 0.05), 2) if thresholds is None else thresholds
    conf, correct = prob.max(1), prob.argmax(1) == y
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({"threshold": float(t), "coverage": float(keep.mean()),
                     "accuracy": float(correct[keep].mean()) if keep.any() else np.nan,
                     "rejected": int((~keep).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Robustness to realistic image corruptions (test set)
# --------------------------------------------------------------------------------------
def corruption_transforms():
    """Simple corruptions that mimic phone photos taken in less ideal conditions."""
    from PIL import Image, ImageEnhance, ImageFilter

    def jpeg(img, quality=20):
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def noise(img, sigma=20, seed=0):
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + np.random.default_rng(seed).normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return {
        "original": lambda im: im,
        "blur": lambda im: im.filter(ImageFilter.GaussianBlur(radius=2)),
        "darker (-40%)": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
        "brighter (+40%)": lambda im: ImageEnhance.Brightness(im).enhance(1.4),
        "rotated 90 deg": lambda im: im.rotate(90, expand=True),
        "JPEG quality 20": jpeg,
        "Gaussian noise": noise,
        "greyscale": lambda im: im.convert("L").convert("RGB"),
    }
