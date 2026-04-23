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
    """A single MVTec AD category, either the ``train`` or ``test`` split."""

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str = "train",
        resize: int = 256,
        crop: int = 224,
    ) -> None:
        if category not in MVTEC_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. Expected one of {MVTEC_CATEGORIES}."
            )
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")

        self.root = Path(root)
        self.category = category
        self.split = split
        self.crop = crop
        self.image_tf, self.mask_tf = build_transforms(resize, crop)

        self.image_paths: List[Path] = []
        self.mask_paths: List[Optional[Path]] = []
        self.labels: List[int] = []
        self.defect_types: List[str] = []
        self._index()

    # ------------------------------------------------------------------ #
    def _index(self) -> None:
        split_dir = self.root / self.category / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Could not find '{split_dir}'. Did you run scripts/download_data.py "
                f"and pass the correct --root?"
            )
        gt_dir = self.root / self.category / "ground_truth"

        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
