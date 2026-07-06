"""
src/dataset.py
==============
PyTorch Dataset class for the AI4Mars terrain segmentation task.

AI4Mars label IDs (verify against official documentation!):
    0  -> soil
    1  -> bedrock
    2  -> sand
    3  -> big_rock
    255 -> ignore / unlabeled

The dataset expects image/mask pairs where:
  - Images are colour rover photographs (JPG or PNG).
  - Masks are single-channel PNG files where each pixel value is a class ID.

Usage
-----
    from src.dataset import AI4MarsDataset, build_pairs_by_stem
    from src.data_paths import RAW_DATA_DIR

    image_files = find_image_files(RAW_DATA_DIR)
    mask_files  = find_mask_files(RAW_DATA_DIR)
    pairs       = build_pairs_by_stem(image_files, mask_files)
    dataset     = AI4MarsDataset(pairs)
    image, mask = dataset[0]  # image: [3, H, W] float32, mask: [H, W] int64
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class AI4MarsDataset(Dataset):
    """PyTorch Dataset for AI4Mars image/mask pairs.

    Parameters
    ----------
    pairs : list of (image_path, mask_path)
        Each element is a 2-tuple of path-like objects pointing to an image
        and its corresponding mask.
    image_size : tuple of (width, height)
        Target spatial resolution for both image and mask.
        PIL uses (width, height) ordering — note this is the *opposite* of
        numpy's (rows, cols) / (height, width) convention.
    transform : callable, optional
        An optional additional transform applied to the ``(image, mask)``
        pair *after* the standard tensor conversion.  Leave as ``None`` for
        the baseline.
    """

    def __init__(
        self,
        pairs: List[Tuple],
        image_size: Tuple[int, int] = (256, 256),
        transform=None,
    ):
        # Ensure all paths are proper Path objects so that we can call
        # .exists(), .stem, etc. safely later.
        self.pairs = [(Path(img), Path(mask)) for img, mask in pairs]
        self.image_size = image_size  # (width, height)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]

        # ------------------------------------------------------------------
        # Load raw image and mask from disk
        # ------------------------------------------------------------------
        # convert("RGB") ensures we always get 3 channels regardless of
        # whether the original file is grayscale, RGBA, palette-mode, etc.
        image = Image.open(image_path).convert("RGB")

        # Masks store integer class IDs — do NOT convert to RGB or the pixel
        # values will be changed.  Open as-is ("L" or "P" mode is fine).
        mask = Image.open(mask_path)

        # ------------------------------------------------------------------
        # Resize
        # ------------------------------------------------------------------
        # Standard bilinear / bicubic resampling is fine for colour images.
        image = image.resize(self.image_size)

        # NEAREST-NEIGHBOUR is *required* for masks because the pixel values
        # are discrete class IDs (0, 1, 2, 3, 255).  Interpolating them would
        # create invalid intermediate values such as 1.7 or 127.
        mask = mask.resize(self.image_size, resample=Image.NEAREST)

        # ------------------------------------------------------------------
        # Convert to numpy arrays
        # ------------------------------------------------------------------
        # Image: H x W x 3, float32, range [0.0, 1.0]
        image = np.array(image, dtype=np.float32) / 255.0

        # Mask: H x W, int64, values are class IDs
        mask = np.array(mask, dtype=np.int64)

        # ------------------------------------------------------------------
        # Convert to PyTorch tensors
        # ------------------------------------------------------------------
        # permute(2, 0, 1) converts H x W x C  →  C x H x W
        # which is the standard PyTorch channel-first format expected by
        # Conv2d layers.
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, H, W]
        mask = torch.from_numpy(mask).long()               # [H, W]

        # ------------------------------------------------------------------
        # Optional extra transforms
        # ------------------------------------------------------------------
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def find_image_files(root: Path) -> List[Path]:
    """Recursively find all image files under *root*.

    Searches for common image extensions used by rover cameras.

    Parameters
    ----------
    root : Path
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted list of image file paths.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = [
        p for p in root.rglob("*")
        if p.suffix.lower() in image_extensions and p.is_file()
    ]
    return sorted(files)


def find_mask_files(root: Path) -> List[Path]:
    """Recursively find all mask files under *root*.

    AI4Mars masks are PNG files.  If the dataset uses a dedicated ``masks/``
    subdirectory, you can pass that path directly instead of the full root.

    Parameters
    ----------
    root : Path
        Directory to search.

    Returns
    -------
    list[Path]
        Sorted list of mask file paths.
    """
    files = [p for p in root.rglob("*.png") if p.is_file()]
    return sorted(files)


def build_pairs_by_stem(
    image_files: List[Path],
    mask_files: List[Path],
) -> List[Tuple[Path, Path]]:
    """Pair image and mask files by matching their filename stems.

    A "stem" is the filename without its extension.  For example:
        ``2p129641989eth0361p2600r8m1.jpg``  →  stem ``2p129641989eth0361p2600r8m1``
        ``2p129641989eth0361p2600r8m1.png``  →  stem ``2p129641989eth0361p2600r8m1``

    .. important::
        This is a **conservative default** that assumes image and mask share
        the same stem.  You **must** inspect the actual AI4Mars directory
        layout after extracting the archive and adjust the pairing logic if
        the naming convention differs.  Run ``01_dataset_inspection.ipynb``
        to check the real structure before training.

    Parameters
    ----------
    image_files : list[Path]
        Candidate image paths (from :func:`find_image_files`).
    mask_files : list[Path]
        Candidate mask paths (from :func:`find_mask_files`).

    Returns
    -------
    list[tuple[Path, Path]]
        Matched (image, mask) pairs.  Only stems present in *both* lists are
        included; unmatched files are silently skipped.
    """
    # Build a stem → path lookup for masks for O(1) lookups
    mask_by_stem = {p.stem: p for p in mask_files}

    pairs: List[Tuple[Path, Path]] = []
    for img_path in image_files:
        mask_path = mask_by_stem.get(img_path.stem)
        if mask_path is not None:
            pairs.append((img_path, mask_path))

    return pairs
