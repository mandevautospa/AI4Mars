"""
src/transforms.py
=================
Helper functions for image and mask preprocessing, plus joint augmentation
transforms for training.

Key rule for segmentation preprocessing:
  - Images contain *color information* — normal (bilinear) resizing is fine
    because intermediate interpolated pixel values are valid colors.
  - Masks contain discrete *class IDs* (integers like 0, 1, 2, 3, 255).
    Interpolating these produces meaningless fractional values such as 1.7.
    You MUST use NEAREST-NEIGHBOUR interpolation for masks.

These helpers are kept separate from the Dataset class so they can be
reused in data-inspection notebooks or custom collate functions.

Joint augmentation
------------------
``SegmentationAugment`` applies spatially consistent random transforms to
both the image tensor *and* the mask tensor in the same call.  Spatial ops
(flip, rotate) are applied identically to both; photometric ops (brightness,
contrast, saturation) are applied only to the image.

Usage example::

    from src.transforms import SegmentationAugment

    augment = SegmentationAugment(
        p_hflip=0.5,
        p_vflip=0.5,
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
    )
    # In AI4MarsDataset, pass as the ``transform`` argument:
    dataset = AI4MarsDataset(pairs, image_size=(256, 256), transform=augment)
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
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
    # Standard BILINEAR resampling (PIL default) is fine for the photograph —
    # it produces smooth color gradients between valid pixel values.
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


# ---------------------------------------------------------------------------
# Joint augmentation for training
# ---------------------------------------------------------------------------

class SegmentationAugment:
    """Randomly augment an image/mask pair in a spatially consistent way.

    Spatial transforms (horizontal flip, vertical flip) are applied identically
    to the image tensor *and* the mask tensor so that pixel-to-class
    correspondence is preserved.  Photometric transforms (brightness, contrast,
    saturation) are applied **only** to the image because the mask stores
    integer class IDs that must not be changed.

    All transforms operate on ``torch.Tensor`` inputs that come directly out of
    ``AI4MarsDataset.__getitem__``:

    * ``image`` : ``[3, H, W]`` float32 in ``[0.0, 1.0]``
    * ``mask``  : ``[H, W]`` int64

    Parameters
    ----------
    p_hflip : float
        Probability of a random horizontal flip.  Default ``0.5``.
    p_vflip : float
        Probability of a random vertical flip.  Default ``0.0`` (rarely
        helpful for rover imagery but included for completeness).
    brightness : float
        Brightness jitter range ``[1 - b, 1 + b]``.  ``0`` disables.
    contrast : float
        Contrast jitter range ``[1 - c, 1 + c]``.  ``0`` disables.
    saturation : float
        Saturation jitter range ``[1 - s, 1 + s]``.  ``0`` disables.
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
    ) -> None:
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations.

        Parameters
        ----------
        image : torch.Tensor
            Shape ``[3, H, W]``, float32, values in ``[0, 1]``.
        mask : torch.Tensor
            Shape ``[H, W]``, int64.

        Returns
        -------
        (image, mask) : Tuple[torch.Tensor, torch.Tensor]
            Augmented pair with the same shapes as the inputs.
        """
        # --- Spatial transforms (applied to both image and mask) ---
        if self.p_hflip > 0.0 and random.random() < self.p_hflip:
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        if self.p_vflip > 0.0 and random.random() < self.p_vflip:
            image = TF.vflip(image)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # --- Photometric transforms (image only) ---
        if self.brightness > 0.0:
            factor = random.uniform(
                max(0.0, 1.0 - self.brightness), 1.0 + self.brightness
            )
            image = TF.adjust_brightness(image, factor)

        if self.contrast > 0.0:
            factor = random.uniform(
                max(0.0, 1.0 - self.contrast), 1.0 + self.contrast
            )
            image = TF.adjust_contrast(image, factor)

        if self.saturation > 0.0:
            factor = random.uniform(
                max(0.0, 1.0 - self.saturation), 1.0 + self.saturation
            )
            image = TF.adjust_saturation(image, factor)

        # Clamp image values back to [0, 1] to avoid out-of-range pixels after
        # photometric jitter.
        image = image.clamp(0.0, 1.0)

        return image, mask
