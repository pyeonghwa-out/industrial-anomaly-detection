"""Tests for AUROC and PRO metrics."""
from __future__ import annotations

import numpy as np

from iad.metrics import compute_image_auroc, compute_pixel_auroc, compute_pro


def test_image_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert compute_image_auroc(scores, labels) == 1.0


def test_image_auroc_random_is_half():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert abs(compute_image_auroc(scores, labels) - 0.5) < 1e-9


def test_pixel_auroc_perfect():
    maps = np.zeros((2, 8, 8))
    masks = np.zeros((2, 8, 8))
