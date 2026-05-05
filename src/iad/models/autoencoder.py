"""Convolutional autoencoder reconstruction baseline.

Unlike PaDiM and PatchCore, this baseline *is* trained — but only on defect-free
images, and it is deliberately small so it trains in a few minutes on a free
Colab T4. It exists to answer a reviewer's natural question: "how much does a
frozen-backbone memory method actually buy you over a classic reconstruction
autoencoder?"

The model learns to reconstruct good images. At test time, regions it cannot
reconstruct well (defects it never saw) produce a large per-pixel reconstruction
error, which becomes the anomaly map.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.gaussian import gaussian_smooth
from .base import AnomalyModel, AnomalyPrediction


class _Encoder(nn.Module):
    def __init__(self, latent: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),     # 112
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),    # 56
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),  # 28
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 14
            nn.Conv2d(256, latent, 4, 2, 1),                                  # 7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, latent: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvAutoencoder(AnomalyModel):
    name = "autoencoder"

    def __init__(
