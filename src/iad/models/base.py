"""Common interface shared by all anomaly models.

Every model implements the same two-phase contract:

* ``fit(train_loader)``   — build the reference (memory bank, Gaussian stats, or
  trained weights) from *defect-free* images only.
* ``predict(images)``     — return per-image scores and pixel-level anomaly maps.

This uniform API is what lets ``run_benchmark.py`` swap methods with one line.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class AnomalyPrediction:
    """Output of a model on a batch of images.

    Attributes:
