"""
src/visualize.py
================
Matplotlib helpers for inspecting AI4Mars image/mask pairs.

Class mapping
-------------
The values below are the *assumed* AI4Mars class IDs.
Always verify them against the official dataset documentation:
  https://zenodo.org/record/4033453

    0   -> soil
    1   -> bedrock
    2   -> sand
    3   -> big_rock
    255 -> ignore / unlabeled
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# Dictionary mapping class ID → human-readable name.
# Verify these IDs against the official AI4Mars documentation before training!
CLASS_NAMES = {
    0: "soil",
    1: "bedrock",
    2: "sand",
    3: "big_rock",
    255: "ignore",
}

# ---------------------------------------------------------------------------
# Comments: American English spelling throughout Python source files
# ---------------------------------------------------------------------------

# Assign a fixed color to each class for consistent overlays.
# Colours are (R, G, B) tuples in the [0, 255] range.
CLASS_COLORS = {
    0: (210, 180, 140),   # soil      — tan
    1: (128, 128, 128),   # bedrock   — grey
    2: (194, 178, 128),   # sand      — sandy yellow
    3: (139, 69,  19),    # big_rock  — saddle brown
    255: (0,   0,   0),   # ignore    — black
}


# ---------------------------------------------------------------------------
# Low-level display helpers
# ---------------------------------------------------------------------------

def show_image(image: np.ndarray, title: str = "Image", ax=None) -> None:
    """Display a numpy image array with matplotlib.

    Parameters
    ----------
    image : np.ndarray
        Shape ``[H, W, 3]`` (uint8 or float32 in [0,1]).
    title : str
        Axes title.
    ax : matplotlib Axes, optional
        If provided, draw into this axes; otherwise create a new figure.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    if standalone:
        plt.tight_layout()
        plt.show()


def show_mask(mask: np.ndarray, title: str = "Mask", ax=None) -> None:
    """Display a class-ID mask with one color per class.

    Parameters
    ----------
    mask : np.ndarray
        Shape ``[H, W]``, integer class IDs.
    title : str
        Axes title.
    ax : matplotlib Axes, optional
        If provided, draw into this axes; otherwise create a new figure.
    """
    # Convert integer class IDs to an RGB image for display.
    rgb = _mask_to_rgb(mask)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    # Add a legend showing which color corresponds to which class.
    patches = [
        mpatches.Patch(
            color=[c / 255 for c in CLASS_COLORS.get(cid, (200, 200, 200))],
            label=f"{cid}: {name}",
        )
        for cid, name in CLASS_NAMES.items()
        if cid in np.unique(mask)
    ]
    if patches:
        ax.legend(handles=patches, loc="lower right", fontsize=8)

    if standalone:
        plt.tight_layout()
        plt.show()


def show_image_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
    title: str = "Overlay",
    ax=None,
) -> None:
    """Show the mask overlaid on the image with transparency.

    Parameters
    ----------
    image : np.ndarray
        Shape ``[H, W, 3]``, uint8 or float32 in [0, 1].
    mask : np.ndarray
        Shape ``[H, W]``, integer class IDs.
    alpha : float
        Transparency of the mask layer (0 = invisible, 1 = fully opaque).
    title : str
        Axes title.
    ax : matplotlib Axes, optional
        If provided, draw into this axes; otherwise create a new figure.
    """
    # Normalise image to float [0, 1] if needed.
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    mask_rgb = _mask_to_rgb(mask).astype(np.float32) / 255.0

    # Blend: overlay = (1 - alpha) * image + alpha * mask
    overlay = (1.0 - alpha) * img_float + alpha * mask_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    if standalone:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# High-level sample inspector
# ---------------------------------------------------------------------------

def show_sample(image_path, mask_path) -> None:
    """Load and display a single image/mask pair with full diagnostics.

    Prints:
      - Image path and shape
      - Mask path and shape
      - Unique class IDs found in the mask
      - Pixel count per class

    Shows three panels side-by-side:
      1. Original image
      2. Class mask (colored)
      3. Image + mask overlay

    Parameters
    ----------
    image_path : path-like
        Path to the rover photograph.
    mask_path : path-like
        Path to the corresponding class-ID mask.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    # Load
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path), dtype=np.int64)

    # Diagnostics
    print(f"Image  : {image_path.name}  shape={image.shape}  dtype={image.dtype}")
    print(f"Mask   : {mask_path.name}  shape={mask.shape}  dtype={mask.dtype}")
    unique_ids = np.unique(mask)
    print(f"Unique class IDs in mask: {unique_ids.tolist()}")
    print_mask_distribution(mask)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    show_image(image, title="Image", ax=axes[0])
    show_mask(mask, title="Mask", ax=axes[1])
    show_image_mask_overlay(image, mask, title="Overlay", ax=axes[2])
    plt.suptitle(image_path.stem, fontsize=11)
    plt.tight_layout()
    plt.show()


def print_mask_distribution(mask: np.ndarray) -> None:
    """Print how many pixels belong to each class.

    Parameters
    ----------
    mask : np.ndarray
        Shape ``[H, W]``, integer class IDs.
    """
    total = mask.size
    print(f"\nMask class distribution (total {total} pixels):")
    for cid in sorted(np.unique(mask)):
        count = int((mask == cid).sum())
        pct = 100.0 * count / total
        name = CLASS_NAMES.get(int(cid), "unknown")
        print(f"  class {cid:3d} ({name:>10s}): {count:>8d} px  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a class-ID mask to an RGB image using :data:`CLASS_COLORS`.

    Parameters
    ----------
    mask : np.ndarray
        Shape ``[H, W]``, integer class IDs.

    Returns
    -------
    np.ndarray
        Shape ``[H, W, 3]``, dtype uint8.
    """
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        rgb[mask == cid] = color
    return rgb
