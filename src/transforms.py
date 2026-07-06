"""
src/transforms.py
=================
Helper functions for image and mask preprocessing.

Key rule for segmentation preprocessing:
  - Images contain *colour information* — normal (bilinear) resizing is fine
    because intermediate interpolated pixel values are valid colours.
  - Masks contain discrete *class IDs* (integers like 0, 1, 2, 3, 255).
    Interpolating these produces meaningless fractional values such as 1.7.
    You MUST use NEAREST-NEIGHBOUR interpolation for masks.

These helpers are kept separate from the Dataset class so they can be
reused in data-inspection notebooks or custom collate functions.
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image


def resize_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    image_size: Tuple[int, int] = (256, 256),
) -> Tuple[Image.Image, Image.Image]:
    """Resize an image and its corresponding mask to *image_size*.

    Parameters
    ----------
    image : PIL.Image.Image
        The input rover photograph (any PIL mode).
    mask : PIL.Image.Image
        The corresponding class-ID mask (single-channel PNG).
    image_size : tuple of (width, height)
        Target size.  Note: PIL uses (width, height), not (height, width).

    Returns
    -------
    (resized_image, resized_mask) : tuple of PIL.Image.Image
        Both resized to *image_size*.
    """
    # Standard resizing is fine for the photograph — bilinear interpolation
    # produces smooth colour gradients.
    resized_image = image.resize(image_size)

    # Nearest-neighbour MUST be used for the mask so that class IDs are never
    # blended together.
    resized_mask = mask.resize(image_size, resample=Image.NEAREST)

    return resized_image, resized_mask


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a float32 PyTorch tensor.

    Steps:
        1. Cast pixels to float32 and scale to [0.0, 1.0].
        2. Permute dimensions from H x W x C  →  C x H x W.

    Parameters
    ----------
    image : PIL.Image.Image
        An RGB image (mode "RGB").

    Returns
    -------
    torch.Tensor
        Shape ``[3, H, W]``, dtype ``float32``, values in ``[0.0, 1.0]``.
    """
    arr = np.array(image, dtype=np.float32) / 255.0  # H x W x 3
    return torch.from_numpy(arr).permute(2, 0, 1)    # 3 x H x W


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """Convert a PIL class-ID mask to a long (int64) PyTorch tensor.

    The mask pixel values are class IDs — we keep them as integers so that
    they can be passed directly to ``torch.nn.CrossEntropyLoss``, which
    expects target tensors of dtype ``long``.

    Parameters
    ----------
    mask : PIL.Image.Image
        A single-channel mask where each pixel is a class ID.

    Returns
    -------
    torch.Tensor
        Shape ``[H, W]``, dtype ``int64`` (long).
    """
    arr = np.array(mask, dtype=np.int64)  # H x W
    return torch.from_numpy(arr).long()   # [H, W]
