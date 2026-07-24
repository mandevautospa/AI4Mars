"""
src/train_utils.py
==================
Reusable training utilities for the AI4Mars segmentation pipeline.

Tensor shape conventions used throughout:
    - Input images  : [B, 3, H, W]  float32
    - Output logits : [B, num_classes, H, W]  float32  (raw, un-softmaxed)
    - Target masks  : [B, H, W]  int64 (long)

Loss function:
    torch.nn.CrossEntropyLoss(ignore_index=255)

    CrossEntropyLoss expects logits (NOT softmax probabilities) and target
    class IDs as integers.  The ignore_index=255 argument tells it to skip
    pixels labelled 255 (unlabeled / out-of-scope regions in AI4Mars masks).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU).

    Returns
    -------
    torch.device
        The device object to pass to ``.to(device)``.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model and optimizer state to disk.

    Parameters
    ----------
    model : nn.Module
        The model whose weights we want to save.
    optimizer : torch.optim.Optimizer
        The optimizer whose state we also save (allows resuming training).
    epoch : int
        Current epoch number (stored as metadata in the checkpoint).
    path : Path
        Destination file path — should end in ``.pth``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if metadata is not None:
        payload["metadata"] = metadata

    torch.save(payload, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
    expected_metadata: Optional[Dict[str, Any]] = None,
    require_metadata_match: bool = True,
) -> int:
    """Load model and optimizer weights from a checkpoint file.

    Parameters
    ----------
    model : nn.Module
        Model to load weights into (must have the same architecture).
    optimizer : torch.optim.Optimizer
        Optimizer to restore state into.
    path : Path
        Path to the ``.pth`` checkpoint file.
    device : torch.device
        Device to map the tensors to when loading.

    Returns
    -------
    int
        The epoch at which the checkpoint was saved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if expected_metadata is not None:
        _validate_checkpoint_metadata(
            checkpoint=checkpoint,
            expected_metadata=expected_metadata,
            require_metadata_match=require_metadata_match,
            checkpoint_path=path,
        )

    epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {path}  (epoch {epoch})")
    return epoch


def _validate_checkpoint_metadata(
    checkpoint: Dict[str, Any],
    expected_metadata: Dict[str, Any],
    require_metadata_match: bool,
    checkpoint_path: Path,
) -> None:
    """Validate selected metadata keys against expected values."""
    checkpoint_metadata = checkpoint.get("metadata")
    if checkpoint_metadata is None:
        message = (
            "Checkpoint metadata not found. Cannot verify split provenance for "
            f"{checkpoint_path}."
        )
        if require_metadata_match:
            raise RuntimeError(message)
        print(f"WARNING: {message}")
        return

    mismatches = []
    for key, expected_value in expected_metadata.items():
        actual_value = checkpoint_metadata.get(key)
        if actual_value != expected_value:
            mismatches.append((key, expected_value, actual_value))

    if mismatches:
        mismatch_text = "; ".join(
            f"{key}: expected={expected!r}, actual={actual!r}"
            for key, expected, actual in mismatches
        )
        message = (
            "Checkpoint metadata does not match expected evaluation metadata: "
            f"{mismatch_text}"
        )
        if require_metadata_match:
            raise RuntimeError(message)
        print(f"WARNING: {message}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
    scheduler: Optional[Any] = None,
) -> float:
    """Run one full training epoch.

    Parameters
    ----------
    model : nn.Module
        Segmentation model.  Expected input ``[B, 3, H, W]``, output
        ``[B, num_classes, H, W]``.
    dataloader : DataLoader
        Training data loader yielding ``(images, masks)`` batches.
    optimizer : torch.optim.Optimizer
        Optimiser (e.g. ``torch.optim.Adam``).
    loss_fn : nn.Module
        Loss function (e.g. ``CrossEntropyLoss(ignore_index=255)``).
    device : torch.device
        Device to run computations on.
    max_grad_norm : float
        Maximum L2 norm for gradient clipping.  Clipping stabilises training
        and prevents NaN losses caused by exploding gradients.  Set to ``0``
        to disable.  Default ``1.0``.
    scheduler : optional
        A ``torch.optim.lr_scheduler`` instance whose ``.step()`` method
        is called **per batch** (e.g. ``OneCycleLR``) *or* ``None`` to
        skip.  For epoch-level schedulers (e.g. ``ReduceLROnPlateau``,
        ``CosineAnnealingLR``) call ``.step()`` outside this function.

    Returns
    -------
    float
        Mean training loss over all *valid* (non-NaN) batches in this epoch.
    """
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)  # [B, 3, H, W]
        masks = masks.to(device)    # [B, H, W]

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)      # [B, num_classes, H, W]

        # Compute loss
        loss = loss_fn(logits, masks)

        # Guard against NaN/Inf loss — skip the update so that one bad batch
        # does not corrupt model weights or the gradient state.
        if not torch.isfinite(loss):
            print(
                f"  WARNING: non-finite loss (NaN/Inf) at batch "
                f"{batch_idx + 1} — skipping update."
            )
            continue

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients / NaN loss.
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Per-batch scheduler step (e.g. OneCycleLR)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        valid_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}  loss={loss.item():.4f}")

    if valid_batches == 0:
        return float("nan")
    return total_loss / valid_batches


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int = 4,
    ignore_index: int = 255,
    return_per_class_iou: bool = False,
) -> dict:
    """Evaluate the model on a validation or test DataLoader.

    Parameters
    ----------
    model : nn.Module
        Segmentation model (same architecture as used during training).
    dataloader : DataLoader
        Validation / test data loader.
    loss_fn : nn.Module
        Loss function used to compute validation loss.
    device : torch.device
        Device to run computations on.
    num_classes : int
        Number of semantic classes (not counting the ignore class).
    ignore_index : int
        Pixels with this label are excluded from metric computation.
    return_per_class_iou : bool
        If True, also compute and return per-class IoU in key
        ``"per_class_iou"``.

    Returns
    -------
    dict
        Keys: ``"val_loss"`` (float), ``"pixel_acc"`` (float),
        ``"mean_iou"`` (float), ``"finite_loss_batches"`` (int), and
        ``"skipped_all_ignore_loss_batches"`` (int).
    """
    model.eval()
    total_loss = 0.0
    finite_loss_batches = 0
    skipped_all_ignore_loss_batches = 0
    total_correct = 0
    total_valid = 0
    class_intersections = torch.zeros(num_classes, dtype=torch.long)
    class_unions = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)              # [B, num_classes, H, W]
            valid = masks != ignore_index
            if valid.any():
                loss = loss_fn(logits, masks)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Evaluation loss is non-finite for a batch containing valid target pixels."
                    )
                total_loss += loss.item()
                finite_loss_batches += 1
            else:
                skipped_all_ignore_loss_batches += 1

            # Convert logits to predicted class IDs
            preds = logits.argmax(dim=1)        # [B, H, W]

            total_correct += ((preds == masks) & valid).sum().item()
            total_valid += valid.sum().item()

            for class_idx in range(num_classes):
                pred_c = (preds == class_idx) & valid
                true_c = (masks == class_idx) & valid
                class_intersections[class_idx] += (pred_c & true_c).sum().cpu()
                class_unions[class_idx] += (pred_c | true_c).sum().cpu()

    acc = total_correct / max(total_valid, 1)

    per_class_iou = []
    for class_idx in range(num_classes):
        union = int(class_unions[class_idx].item())
        if union == 0:
            per_class_iou.append(None)
        else:
            intersection = int(class_intersections[class_idx].item())
            per_class_iou.append(intersection / union)

    valid_scores = [score for score in per_class_iou if score is not None]
    miou = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0

    if finite_loss_batches == 0:
        raise RuntimeError("Evaluation split contains no batches with valid target pixels.")

    results = {
        "val_loss": total_loss / finite_loss_batches,
        "pixel_acc": acc,
        "mean_iou": miou,
        "finite_loss_batches": finite_loss_batches,
        "skipped_all_ignore_loss_batches": skipped_all_ignore_loss_batches,
    }

    if return_per_class_iou:
        results["per_class_iou"] = per_class_iou

    return results
