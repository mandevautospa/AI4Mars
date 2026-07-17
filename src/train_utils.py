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

from src.metrics import mean_iou, pixel_accuracy
from src.metrics import intersection_over_union


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
    require_metadata_match: bool = False,
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

    Returns
    -------
    float
        Mean training loss over all batches in this epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)  # [B, 3, H, W]
        masks = masks.to(device)    # [B, H, W]

        # Forward pass
        logits = model(images)      # [B, num_classes, H, W]

        # Compute loss
        loss = loss_fn(logits, masks)

        # Backward pass and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}  loss={loss.item():.4f}")

    return total_loss / len(dataloader)


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
        ``"mean_iou"`` (float).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)              # [B, num_classes, H, W]
            loss = loss_fn(logits, masks)
            total_loss += loss.item()

            # Convert logits to predicted class IDs
            preds = logits.argmax(dim=1)        # [B, H, W]
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    # Concatenate all batches along the batch dimension
    all_preds = torch.cat(all_preds, dim=0)     # [N, H, W]
    all_targets = torch.cat(all_targets, dim=0) # [N, H, W]

    acc = pixel_accuracy(all_preds, all_targets, ignore_index=ignore_index)
    miou = mean_iou(all_preds, all_targets, num_classes=num_classes, ignore_index=ignore_index)

    results = {
        "val_loss": total_loss / len(dataloader),
        "pixel_acc": acc,
        "mean_iou": miou,
    }

    if return_per_class_iou:
        results["per_class_iou"] = intersection_over_union(
            all_preds,
            all_targets,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    return results
