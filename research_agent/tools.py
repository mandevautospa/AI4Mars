from __future__ import annotations

import json
import math
from pathlib import Path

from agents import function_tool


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_PARTS = {
    ".git",
    ".research_agent",
    ".venv",
    "__pycache__",
    "models",
    "raw",
}


def _safe_path(relative_path: str) -> Path:
    candidate = (REPO_ROOT / relative_path).resolve()
    if candidate != REPO_ROOT and REPO_ROOT not in candidate.parents:
        raise ValueError("Path must remain inside the AI4Mars repository.")
    relative_parts = candidate.relative_to(REPO_ROOT).parts
    if any(part in EXCLUDED_PARTS or part.startswith(".env") for part in relative_parts):
        raise ValueError("That path is excluded from agent access.")
    return candidate


def _project_files(subdirectory: str = ".") -> list[str]:
    base = _safe_path(subdirectory)
    if not base.exists() or not base.is_dir():
        raise ValueError("Subdirectory does not exist or is not a directory.")

    files: list[str] = []
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in EXCLUDED_PARTS or part.startswith(".env") for part in relative.parts):
            continue
        if path.suffix.lower() in ALLOWED_SUFFIXES or path.name == ".gitignore":
            files.append(relative.as_posix())
        if len(files) >= 250:
            break
    return sorted(files)


@function_tool
def list_project_files(subdirectory: str = ".") -> str:
    """List readable AI4Mars project files under a repository-relative directory."""
    return json.dumps(_project_files(subdirectory), indent=2)


@function_tool
def read_project_file(path: str, max_characters: int = 16000) -> str:
    """Read a text-based project file using a repository-relative path.

    Use this to ground claims about the implementation. Secret files, model files,
    raw data, virtual environments, and paths outside the repository are blocked.
    """
    candidate = _safe_path(path)
    if not candidate.is_file():
        raise ValueError("File does not exist.")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES and candidate.name != ".gitignore":
        raise ValueError("Unsupported file type.")
    limit = min(max(max_characters, 500), 30000)
    text = candidate.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + f"\n\n[truncated after {limit} characters]"
    return text


def _notebook_digest(path: str, max_cells: int = 16) -> dict[str, object]:
    candidate = _safe_path(path)
    if candidate.suffix.lower() != ".ipynb" or not candidate.is_file():
        raise ValueError("Provide a repository-relative .ipynb path.")

    notebook = json.loads(candidate.read_text(encoding="utf-8"))
    cells: list[dict[str, object]] = []
    for index, cell in enumerate(notebook.get("cells", [])[: min(max(max_cells, 1), 40)]):
        source = "".join(cell.get("source", []))[:5000]
        outputs: list[str] = []
        for output in cell.get("outputs", []):
            if "text" in output:
                outputs.append("".join(output["text"])[:2000])
            data = output.get("data", {})
            if "text/plain" in data:
                outputs.append("".join(data["text/plain"])[:2000])
        cells.append(
            {
                "index": index,
                "cell_type": cell.get("cell_type", "unknown"),
                "source": source,
                "text_outputs": outputs,
            }
        )
    return {
        "path": path,
        "total_cells": len(notebook.get("cells", [])),
        "included_cells": len(cells),
        "cells": cells,
    }


@function_tool
def inspect_notebook(path: str, max_cells: int = 16) -> str:
    """Extract readable source and text outputs from an AI4Mars notebook."""
    return json.dumps(_notebook_digest(path, max_cells), indent=2)


def _segmentation_metrics(
    confusion_matrix: list[list[int]], class_names: list[str]
) -> dict[str, object]:
    size = len(confusion_matrix)
    if size == 0 or any(len(row) != size for row in confusion_matrix):
        raise ValueError("Confusion matrix must be a non-empty square matrix.")
    if len(class_names) != size:
        raise ValueError("class_names length must match the confusion matrix size.")
    if any(value < 0 for row in confusion_matrix for value in row):
        raise ValueError("Confusion matrix counts cannot be negative.")

    total = sum(sum(row) for row in confusion_matrix)
    true_positive_total = sum(confusion_matrix[i][i] for i in range(size))
    per_class: dict[str, dict[str, float | int | None]] = {}
    ious: list[float] = []
    f1s: list[float] = []

    for i, name in enumerate(class_names):
        tp = confusion_matrix[i][i]
        fn = sum(confusion_matrix[i]) - tp
        fp = sum(confusion_matrix[row][i] for row in range(size)) - tp
        support = tp + fn
        union = tp + fp + fn
        iou = tp / union if union else None
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / support if support else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        if iou is not None:
            ious.append(iou)
        if f1 is not None:
            f1s.append(f1)
        per_class[name] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1_dice": f1,
            "iou": iou,
        }

    return {
        "pixel_accuracy": true_positive_total / total if total else None,
        "mean_iou_present_classes": sum(ious) / len(ious) if ious else None,
        "macro_f1_present_classes": sum(f1s) / len(f1s) if f1s else None,
        "per_class": per_class,
    }


@function_tool
def compute_segmentation_metrics(
    confusion_matrix: list[list[int]], class_names: list[str]
) -> str:
    """Compute pixel accuracy, per-class IoU, precision, recall, and Dice/F1.

    Rows must represent ground-truth classes and columns predicted classes.
    """
    return json.dumps(_segmentation_metrics(confusion_matrix, class_names), indent=2)


@function_tool
def estimate_tensor_memory(
    shape: list[int], dtype: str = "float32", simultaneous_copies: float = 1.0
) -> str:
    """Estimate raw tensor memory for a shape, dtype, and number of simultaneous copies.

    This is a lower-bound tensor estimate, not a full training-memory prediction.
    """
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("All tensor dimensions must be positive integers.")
    if simultaneous_copies <= 0:
        raise ValueError("simultaneous_copies must be positive.")
    bytes_per_element = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "float16": 2,
        "bfloat16": 2,
        "int16": 2,
        "float32": 4,
        "int32": 4,
        "float64": 8,
        "int64": 8,
    }
    normalized_dtype = dtype.lower()
    if normalized_dtype not in bytes_per_element:
        raise ValueError(f"Unsupported dtype: {dtype}")
    elements = math.prod(shape)
    total_bytes = elements * bytes_per_element[normalized_dtype] * simultaneous_copies
    return json.dumps(
        {
            "shape": shape,
            "dtype": normalized_dtype,
            "elements": elements,
            "simultaneous_copies": simultaneous_copies,
            "bytes": total_bytes,
            "mib": total_bytes / 1024**2,
            "gib": total_bytes / 1024**3,
            "warning": (
                "This excludes parameters, gradients, optimizer states, retained "
                "activations, CUDA context, allocator fragmentation, and other tensors."
            ),
        },
        indent=2,
    )
