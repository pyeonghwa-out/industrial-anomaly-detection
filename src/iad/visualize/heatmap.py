"""Heatmap rendering: the visual "wow factor" for the README.

Produces the canonical 4-panel figure used throughout industrial AD papers:
original | ground-truth mask | predicted heatmap | overlay.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation -> HxWx3 uint8-ready float in [0, 1]."""
    img = image.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def _normalize_map(amap: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
    vmin = float(amap.min()) if vmin is None else vmin
    vmax = float(amap.max()) if vmax is None else vmax
    if vmax - vmin < 1e-12:
        return np.zeros_like(amap)
    return np.clip((amap - vmin) / (vmax - vmin), 0.0, 1.0)


def overlay_heatmap(
    image: torch.Tensor,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Blend a jet-coloured anomaly map over the original image."""
    import matplotlib.cm as cm

    base = denormalize(image)
    norm = _normalize_map(np.asarray(anomaly_map), vmin, vmax)
    colored = cm.jet(norm)[..., :3]
    return np.clip((1 - alpha) * base + alpha * colored, 0.0, 1.0)


def save_comparison_grid(
