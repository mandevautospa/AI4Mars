"""
src/visualize.py
================
Matplotlib helpers for inspecting AI4Mars image/mask pairs.

Class mapping
-------------
AI4Mars uses TWO different pixel-value label schemes (see
``data/raw/ai4mars/ai4mars-dataset-merged-0.6/label_keys.json``):

``NAV`` (most images -- MSL/MER and M2020 non-geology):
    0   -> soil
    1   -> bedrock
    2   -> sand
    3   -> big_rock
    255 -> ignore / unlabeled

Some MSL MastCam training masks (``msl/mcam/labels/train/*_15033_merged.png``)
use a legacy ignore value ``4`` on disk. Those masks are normalized to ``255``
before display and downstream use.

``M2020_GEO`` (only ``m2020/labels/M2020_GEO/...``, Perseverance geology
labels): bedrock subtypes (0-6), float rock subtypes (10-17), sand subtypes
(20-22), pebbles (30), vein (40), hill/peak (50), 255 -> ignore.

Functions below accept an optional ``scheme`` argument (``"NAV"`` or
``"M2020_GEO"``); ``show_sample`` auto-detects it from the mask path.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from src.dataset import normalize_ai4mars_mask


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

# Dictionary mapping class ID → human-readable name (NAV scheme).
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

# Assign a fixed color to each class for consistent overlays (NAV scheme).
# Colours are (R, G, B) tuples in the [0, 255] range.
CLASS_COLORS = {
    0: (210, 180, 140),   # soil      — tan
    1: (128, 128, 128),   # bedrock   — grey
    2: (194, 178, 128),   # sand      — sandy yellow
    3: (139, 69,  19),    # big_rock  — saddle brown
    255: (0,   0,   0),   # ignore    — black
}

# Class ID → human-readable name (M2020_GEO scheme -- Perseverance geology).
M2020_GEO_CLASS_NAMES = {
    0: "bedrock_massive",
    1: "bedrock_layered_angled",
    2: "bedrock_layered_flat",
    3: "bedrock_layered_unsure",
    4: "bedrock_conglomerate",
    5: "bedrock_holey",
    6: "bedrock_unsure",
    10: "float_rock_massive",
    11: "float_rock_layered_angled",
    12: "float_rock_layered_flat",
    13: "float_rock_layered_unsure",
    14: "float_rock_conglomerate",
    15: "float_rock_holey",
    16: "float_rock_mixed",
    17: "float_rock_unsure",
    20: "sand_dune",
    21: "sand_ripples",
    22: "sand_sand",
    30: "pebbles",
    40: "vein",
    50: "hill_peak",
    255: "ignore",
}

# Class ID → color (M2020_GEO scheme). Grouped by base type (bedrock=grey,
# float rock=brown, sand=yellow tones), shaded by subtype.
M2020_GEO_CLASS_COLORS = {
    0: (100, 100, 100),
    1: (120, 120, 120),
    2: (140, 140, 140),
    3: (160, 160, 160),
    4: (110, 110, 130),
    5: (90,  90,  90),
    6: (170, 170, 170),
    10: (139, 90,  43),
    11: (155, 103, 60),
    12: (170, 118, 78),
    13: (185, 133, 96),
    14: (120, 78,  35),
    15: (100, 65,  30),
    16: (145, 95,  50),
    17: (200, 150, 110),
    20: (222, 202, 130),
    21: (210, 190, 110),
    22: (194, 178, 128),
    30: (180, 160, 120),
    40: (100, 200, 200),
    50: (60,  120, 60),
    255: (0,   0,   0),
}

# Registries used to look up the right dict given a scheme name.
CLASS_NAME_SCHEMES = {"NAV": CLASS_NAMES, "M2020_GEO": M2020_GEO_CLASS_NAMES}
CLASS_COLOR_SCHEMES = {"NAV": CLASS_COLORS, "M2020_GEO": M2020_GEO_CLASS_COLORS}


def detect_label_scheme(mask_path) -> str:
    """Infer the AI4Mars label scheme ("NAV" or "M2020_GEO") from a mask path.

    Parameters
    ----------
    mask_path : path-like
        Path to a mask file.

    Returns
    -------
    str
        ``"M2020_GEO"`` if the path contains that folder name, else ``"NAV"``.
    """
    return "M2020_GEO" if "M2020_GEO" in str(mask_path) else "NAV"


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


def show_mask(mask: np.ndarray, title: str = "Mask", ax=None, scheme: str = "NAV") -> None:
    """Display a class-ID mask with one color per class.

    Parameters
    ----------
    mask : np.ndarray
        Shape ``[H, W]``, integer class IDs.
    title : str
        Axes title.
    ax : matplotlib Axes, optional
        If provided, draw into this axes; otherwise create a new figure.
    scheme : str
        Label scheme to use for names/colors: ``"NAV"`` or ``"M2020_GEO"``.
    """
    class_names = CLASS_NAME_SCHEMES[scheme]
    class_colors = CLASS_COLOR_SCHEMES[scheme]

    # Convert integer class IDs to an RGB image for display.
    rgb = _mask_to_rgb(mask, scheme=scheme)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    # Add a legend showing which color corresponds to which class.
    patches = [
        mpatches.Patch(
            color=[c / 255 for c in class_colors.get(cid, (200, 200, 200))],
            label=f"{cid}: {name}",
        )
        for cid, name in class_names.items()
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
    scheme: str = "NAV",
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
    scheme : str
        Label scheme to use for colors: ``"NAV"`` or ``"M2020_GEO"``.
    """
    # Normalise image to float [0, 1] if needed.
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    # Some discovered pairs can have mismatched dimensions; for visual overlay
    # only, resize mask to image size with nearest-neighbour to preserve IDs.
    if mask.shape != image.shape[:2]:
        target_size = (image.shape[1], image.shape[0])
        mask = np.array(
            Image.fromarray(mask.astype(np.int32)).resize(target_size, resample=Image.NEAREST),
            dtype=np.int64,
        )

    mask_rgb = _mask_to_rgb(mask, scheme=scheme).astype(np.float32) / 255.0

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
    mask = normalize_ai4mars_mask(mask, mask_path)

    # Auto-detect which of the two AI4Mars label schemes this mask uses.
    scheme = detect_label_scheme(mask_path)

    # Diagnostics
    print(f"Image  : {image_path.name}  shape={image.shape}  dtype={image.dtype}")
    print(f"Mask   : {mask_path.name}  shape={mask.shape}  dtype={mask.dtype}")
    print(f"Label scheme: {scheme}")
    if mask.shape != image.shape[:2]:
        print(
            "WARNING: image/mask spatial shapes differ; "
            "resizing mask to image size for display overlay."
        )
    unique_ids = np.unique(mask)
    print(f"Unique class IDs in mask: {unique_ids.tolist()}")
    print_mask_distribution(mask, scheme=scheme)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    show_image(image, title="Image", ax=axes[0])
    show_mask(mask, title="Mask", ax=axes[1], scheme=scheme)
    show_image_mask_overlay(image, mask, title="Overlay", ax=axes[2], scheme=scheme)
    plt.suptitle(image_path.stem, fontsize=11)
    plt.tight_layout()
    plt.show()


def print_mask_distribution(mask: np.ndarray, scheme: str = "NAV") -> None:
    """Print how many pixels belong to each class.

    Parameters
    ----------
    mask : np.ndarray
        Shape ``[H, W]``, integer class IDs.
    scheme : str
        Label scheme to use for names: ``"NAV"`` or ``"M2020_GEO"``.
    """
    class_names = CLASS_NAME_SCHEMES[scheme]
    total = mask.size
    print(f"\nMask class distribution (total {total} pixels):")
    for cid in sorted(np.unique(mask)):
        count = int((mask == cid).sum())
        pct = 100.0 * count / total
        name = class_names.get(int(cid), "unknown")
        print(f"  class {cid:3d} ({name:>10s}): {count:>8d} px  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _mask_to_rgb(mask: np.ndarray, scheme: str = "NAV") -> np.ndarray:
    """Convert a class-ID mask to an RGB image using the given label scheme.

    Parameters
    ----------
    mask : np.ndarray
        Shape ``[H, W]``, integer class IDs.
    scheme : str
        Label scheme to use for colors: ``"NAV"`` or ``"M2020_GEO"``.

    Returns
    -------
    np.ndarray
        Shape ``[H, W, 3]``, dtype uint8.
    """
    class_colors = CLASS_COLOR_SCHEMES[scheme]
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in class_colors.items():
        rgb[mask == cid] = color
    return rgb
