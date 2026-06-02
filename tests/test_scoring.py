"""End-to-end scoring tests on synthetic data (no pretrained weights needed)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from iad.datasets.mvtec import MVTecDataset
from iad.metrics import evaluate
from iad.models.padim import PaDiM
from iad.models.patchcore import PatchCore
from iad.utils.gaussian import gaussian_smooth
from iad.utils.sampling import KCenterGreedy


def test_gaussian_smooth_preserves_shape():
    x = torch.randn(2, 16, 16)
    out = gaussian_smooth(x, sigma=2.0)
    assert out.shape == x.shape


def test_kcenter_returns_requested_count():
    feats = torch.randn(200, 32)
    idx = KCenterGreedy(projection_dim=8, seed=0).sample(feats, ratio=0.1)
    assert idx.numel() == 20
    assert idx.unique().numel() == 20  # no duplicates


def _loaders(root):
    train = DataLoader(
        MVTecDataset(root, "bottle", "train", resize=64, crop=64), batch_size=3
    )
    test = DataLoader(
        MVTecDataset(root, "bottle", "test", resize=64, crop=64), batch_size=3
    )
    return train, test


