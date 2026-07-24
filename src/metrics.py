"""
src/metrics.py
==============
Segmentation evaluation metrics for the AI4Mars project.

Why not just use accuracy?
--------------------------
Pixel accuracy = (correct pixels) / (total pixels).

This sounds good but is *misleading* when one class dominates the scene.
For example, if 90% of Mars images are "soil", a model that always predicts
"soil" achieves 90% pixel accuracy while being completely useless for
detecting rocks or sand.

Intersection over Union (IoU) is the standard metric for segmentation:
    IoU(class c) = TP_c / (TP_c + FP_c + FN_c)

where:
    TP = pixels correctly predicted as class c
    FP = pixels wrongly predicted as class c
    FN = pixels of class c missed by the model

Mean IoU (mIoU) averages IoU across all valid classes.
"""

import torch


def pixel_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
) -> float:
    """Compute overall pixel accuracy, ignoring *ignore_index* pixels.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted class map, shape ``[B, H, W]``, dtype long.
        Each value is a class ID (already argmax-ed from logits).
    targets : torch.Tensor
        Ground-truth class map, shape ``[B, H, W]``, dtype long.
    ignore_index : int
        Pixels where ``target == ignore_index`` are excluded from the count.

    Returns
    -------
    float
        Fraction of correctly classified non-ignored pixels in ``[0, 1]``.

    Notes
    -----
    Use as a *sanity check*, not as the primary metric — see module docstring
    for why accuracy alone is not sufficient for segmentation.
    """
    valid = targets != ignore_index          # boolean mask of valid pixels
    correct = (preds == targets) & valid     # correctly classified valid pixels
    return correct.sum().item() / valid.sum().item()


def segmentation_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Count ground-truth/prediction class pairs for valid segmentation pixels.

    Rows correspond to ground-truth classes and columns correspond to predicted
    classes. Pixels whose target is *ignore_index* are excluded.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"preds and targets must have the same shape; got {tuple(preds.shape)} and {tuple(targets.shape)}."
        )

    valid = targets != ignore_index
    valid_targets = targets[valid].to(torch.long)
    valid_preds = preds[valid].to(torch.long)

    if valid_targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long)

    if (
        valid_targets.min().item() < 0
        or valid_targets.max().item() >= num_classes
        or valid_preds.min().item() < 0
        or valid_preds.max().item() >= num_classes
    ):
        raise ValueError(f"Valid predictions and targets must be in [0, {num_classes - 1}].")

    flattened_pairs = valid_targets * num_classes + valid_preds
    return torch.bincount(
        flattened_pairs,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def intersection_over_union(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> list:
    """Compute per-class Intersection over Union.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted class map, shape ``[B, H, W]``, dtype long.
    targets : torch.Tensor
        Ground-truth class map, shape ``[B, H, W]``, dtype long.
    num_classes : int
        Number of foreground classes (not counting *ignore_index*).
    ignore_index : int
        Pixels with this target value are excluded from all calculations.

    Returns
    -------
    list[float | None]
        Length *num_classes*.  Each element is the IoU for that class,
        or ``None`` if the class does not appear in either predictions or
        ground truth (i.e. there is no support to compute IoU for it).
    """
    ious = []

    # Create a validity mask once — reuse for all classes.
    valid = targets != ignore_index

    for c in range(num_classes):
        # Only consider pixels that are actually valid (not ignored).
        pred_c = (preds == c) & valid
        true_c = (targets == c) & valid

        intersection = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()

        if union == 0:
            # Class not present in this batch — exclude from mean calculation.
            ious.append(None)
        else:
            ious.append(intersection / union)

    return ious


def mean_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> float:
    """Compute mean IoU (mIoU) across all valid classes.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted class map, shape ``[B, H, W]``, dtype long.
    targets : torch.Tensor
        Ground-truth class map, shape ``[B, H, W]``, dtype long.
    num_classes : int
        Number of foreground classes.
    ignore_index : int
        Pixels with this target value are excluded.

    Returns
    -------
    float
        Mean IoU across classes that appear in *targets* (or *preds*).
        Returns 0.0 if no valid class is found.
    """
    ious = intersection_over_union(preds, targets, num_classes, ignore_index)
    valid_ious = [iou for iou in ious if iou is not None]
    if not valid_ious:
        return 0.0
    return sum(valid_ious) / len(valid_ious)
