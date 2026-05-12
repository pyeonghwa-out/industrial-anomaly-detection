"""PaDiM — Patch Distribution Modeling.

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
Anomaly Detection and Localization" (ICPR 2021).

Idea
----
1. Extract multi-layer features from a frozen backbone and concatenate them
   (after upsampling to a common spatial grid). Each spatial position now owns
   an embedding vector that fuses semantic levels.
2. Randomly keep a subset of the embedding channels — the paper shows random
   selection works as well as PCA and is far cheaper.
3. Model every spatial position ``(i, j)`` with a multivariate Gaussian
   ``N(mu_ij, Sigma_ij)`` estimated from the defect-free training set.
4. At test time the anomaly score of a position is the Mahalanobis distance to
   its Gaussian. Far-from-normal patches light up.

No gradient training happens — only mean/covariance estimation.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.gaussian import gaussian_smooth
from .base import AnomalyModel, AnomalyPrediction
from .feature_extractor import FeatureExtractor

# Default number of randomly-kept channels per backbone (from the paper /
# the widely-used anomalib defaults).
_DEFAULT_N_DIMS = {
    "resnet18": 100,
    "resnet50": 550,
    "wide_resnet50_2": 550,
}


class PaDiM(AnomalyModel):
    name = "padim"

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] | None = None,
        n_dims: int | None = None,
        image_size: int = 224,
        sigma: float = 4.0,
        epsilon: float = 0.01,
        device: str = "cuda",
        seed: int = 42,
        pretrained: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        layers = layers or ["layer1", "layer2", "layer3"]
        self.extractor = FeatureExtractor(backbone, layers, pretrained=pretrained).to(self.device)
        self.layers = layers
        self.image_size = image_size
        self.sigma = sigma
        self.epsilon = epsilon
        self.seed = seed

        total_channels = sum(self.extractor.channels.values())
        n_dims = n_dims or _DEFAULT_N_DIMS.get(backbone, 550)
        n_dims = min(n_dims, total_channels)
        generator = torch.Generator().manual_seed(seed)
        self.dim_idx = torch.randperm(total_channels, generator=generator)[:n_dims]

