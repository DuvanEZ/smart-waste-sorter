"""Training loops: two-phase transfer learning with early stopping."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from .models import count_parameters, set_backbone_trainable


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None):
    """One pass over `loader`. Trains when an optimizer is given, otherwise evaluates.

    Returns (mean loss, accuracy, logits, labels); logits/labels only when evaluating.
    """
    training = optimizer is not None
    model.train(training)
    use_amp = device.type == "cuda"
    total_loss, correct, seen = 0.0, 0, 0
    all_logits, all_labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()          # learning rate is updated every batch
            else:
                all_logits.append(logits.float().cpu())
                all_labels.append(y.cpu())
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    logits = torch.cat(all_logits).numpy() if all_logits else None
    labels = torch.cat(all_labels).numpy() if all_labels else None
    return total_loss / seen, correct / seen, logits, labels


def _make_scaler(device):
    if device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler()


def _cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    """Linear warm-up followed by cosine decay (per batch)."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _train_phase(model, phase, epochs, lr, train_loader, val_loader, criterion, device, cfg,
                 history, best, log, patience=None):
    """Train for up to `epochs` epochs, keeping the weights with the best validation macro-F1."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    steps = epochs * len(train_loader)
    scheduler = _cosine_with_warmup(optimizer, steps, warmup_steps=max(1, len(train_loader) // 2))
    scaler = _make_scaler(device)
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler)
        va_loss, va_acc, va_logits, va_labels = run_epoch(model, val_loader, criterion, device)
        va_f1 = f1_score(va_labels, va_logits.argmax(1), average="macro")
        row = {"epoch": len(history) + 1, "phase": phase, "lr": lr_now, "train_loss": tr_loss,
               "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
               "seconds": time.time() - start}
        history.append(row)
        improved = va_f1 > best["f1"] + 1e-4
        if improved:
            best.update(f1=va_f1, epoch=row["epoch"], state=copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        log(f"  [{phase}] epoch {row['epoch']:>2}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.3f} acc {va_acc:.3f} macro-F1 {va_f1:.3f}"
            f"{'  *best*' if improved else ''} ({row['seconds']:.0f}s)")
        if patience is not None and epochs_without_improvement >= patience:
            log(f"  early stopping: no improvement for {patience} epochs")
            break


def fit_transfer_learning(model, train_loader, val_loader, cfg, device, log=print):
    """Two-phase fine-tuning of a pre-trained network.

    Phase 1 ("head"): the ImageNet backbone is frozen and only the new classifier is
    trained, so the random output layer does not destroy the pre-trained features.
    Phase 2 ("fine-tune"): all layers are unfrozen and trained with a smaller learning
    rate, adapting the features to waste images. Early stopping keeps the weights of the
    epoch with the best validation macro-F1.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    history, best = [], {"f1": -1.0, "epoch": 0, "state": None}

    set_backbone_trainable(model, False)
    total, trainable = count_parameters(model)
    log(f"Phase 1 - frozen backbone: {trainable:,} of {total:,} parameters trainable")
    _train_phase(model, "head", cfg.head_epochs, cfg.head_lr, train_loader, val_loader,
                 criterion, device, cfg, history, best, log)

    set_backbone_trainable(model, True)
    total, trainable = count_parameters(model)
    log(f"Phase 2 - fine-tuning: {trainable:,} of {total:,} parameters trainable")
    _train_phase(model, "fine-tune", cfg.finetune_epochs, cfg.finetune_lr, train_loader, val_loader,
                 criterion, device, cfg, history, best, log, patience=cfg.patience)

    model.load_state_dict(best["state"])
    log(f"Best validation macro-F1 {best['f1']:.4f} at epoch {best['epoch']}")
    return model, pd.DataFrame(history), best["epoch"]


def fit_from_scratch(model, train_loader, val_loader, epochs, lr, device, weight_decay=1e-4,
                     patience=5, log=print):
    """Single-phase training used for the small CNN baseline."""
    class _Cfg:  # minimal config object for _train_phase
        pass

    cfg = _Cfg()
    cfg.weight_decay = weight_decay
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    history, best = [], {"f1": -1.0, "epoch": 0, "state": None}
    _train_phase(model, "scratch", epochs, lr, train_loader, val_loader, criterion, device, cfg,
                 history, best, log, patience=patience)
    model.load_state_dict(best["state"])
    return model, pd.DataFrame(history), best["epoch"]


@torch.inference_mode()
def predict_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    _, _, logits, labels = run_epoch(model, loader, criterion, device)
    return logits, labels
