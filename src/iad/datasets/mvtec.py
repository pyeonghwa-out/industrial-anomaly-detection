"""MVTec AD dataset loader.

Directory layout expected on disk (the official MVTec AD structure)::

    <root>/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*_mask.png

The loader exposes a standard ``torch.utils.data.Dataset`` returning, for each
item, a dict with the transformed image, a binary image label (0=good,
1=defect), and a binary segmentation mask resized to the model input size
(all-zeros for good images, which have no ground-truth file).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# The 15 MVTec AD categories, grouped as in the paper.
MVTEC_TEXTURES: List[str] = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
]
MVTEC_OBJECTS: List[str] = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]
MVTEC_CATEGORIES: List[str] = MVTEC_OBJECTS + MVTEC_TEXTURES

# ImageNet normalisation — required because we use frozen ImageNet backbones.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def build_transforms(
    resize: int = 256, crop: int = 224
) -> "tuple[transforms.Compose, transforms.Compose]":
    """Return (image_transform, mask_transform).

    Images are resized then centre-cropped (standard MVTec protocol). The mask
    uses nearest-neighbour interpolation so it stays binary.
    """
    image_tf = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    mask_tf = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
        ]
    )
    return image_tf, mask_tf


class MVTecDataset(Dataset):
