"""Tests for the MVTec dataset loader."""
from __future__ import annotations

import torch

from iad.datasets.mvtec import MVTEC_CATEGORIES, MVTecDataset


def test_categories_count():
    assert len(MVTEC_CATEGORIES) == 15
    assert "bottle" in MVTEC_CATEGORIES


def test_train_split_is_all_good(synthetic_root):
    ds = MVTecDataset(synthetic_root, "bottle", "train", resize=64, crop=64)
    assert len(ds) == 6
    assert all(lbl == 0 for lbl in ds.labels)


def test_test_split_labels_and_masks(synthetic_root):
    ds = MVTecDataset(synthetic_root, "bottle", "test", resize=64, crop=64)
