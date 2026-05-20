"""Run a fitted model over a test split and compute all metrics + timing."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .auroc import compute_image_auroc, compute_pixel_auroc
from .pro import compute_pro


@torch.no_grad()
def evaluate(
    model,
    test_loader: DataLoader,
    compute_pro_metric: bool = True,
) -> Dict[str, float]:
    """Evaluate ``model`` on ``test_loader``.

    Returns a dict with image AUROC, pixel AUROC, PRO and average inference time
    per image (milliseconds). The model must already be fitted.
    """
    all_scores, all_labels = [], []
    all_maps, all_masks = [], []

    total_images = 0
