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
