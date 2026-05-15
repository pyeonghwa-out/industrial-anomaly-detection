"""PatchCore — coreset memory bank + nearest-neighbour scoring.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection"
(CVPR 2022).

Pipeline
--------
1. Extract mid-level features (``layer2``, ``layer3``) from a frozen backbone.
2. **Locally aware patch features**: average-pool each location with its 3x3
   neighbourhood so each patch embeds local context, then resize layers to a
   common grid and concatenate.
3. Collect every patch from the defect-free training set into one big bank and
   keep a **coreset** (greedy k-center subsample) — typically 1-10% of points.
4. At test time, each patch's anomaly score is its Euclidean distance to the
   nearest coreset member. The image score is the maximum patch distance,
   *reweighted* to down-weight points that sit in sparse regions of the bank.

Like PaDiM, nothing is trained — the backbone is frozen and the "memory" is
just stored feature vectors.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..utils.gaussian import gaussian_smooth
from ..utils.sampling import KCenterGreedy
from ._nn import NearestNeighbors
from .base import AnomalyModel, AnomalyPrediction
from .feature_extractor import FeatureExtractor


class PatchCore(AnomalyModel):
    name = "patchcore"

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] | None = None,
        coreset_ratio: float = 0.1,
        num_neighbors: int = 9,
        image_size: int = 224,
        sigma: float = 4.0,
        projection_dim: int = 128,
        device: str = "cuda",
        seed: int = 42,
        pretrained: bool = True,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        layers = layers or ["layer2", "layer3"]
        self.extractor = FeatureExtractor(backbone, layers, pretrained=pretrained).to(self.device)
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.num_neighbors = num_neighbors
        self.image_size = image_size
        self.sigma = sigma
        self.seed = seed
        self._sampler = KCenterGreedy(projection_dim=projection_dim, seed=seed)

        self.memory_bank: torch.Tensor | None = None
        self._nn: NearestNeighbors | None = None
        self.feat_hw: int | None = None

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return locally-aware patch features of shape ``(B, H, W, C)``."""
        feats = self.extractor(images.to(self.device))
        ref_hw = feats[self.layers[0]].shape[-2:]
        pooled = []
        for name in self.layers:
            f = feats[name]
            # 3x3 local neighbourhood aggregation (stride 1 keeps resolution).
            f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
            f = F.interpolate(f, size=ref_hw, mode="bilinear", align_corners=False)
            pooled.append(f)
        embedding = torch.cat(pooled, dim=1)  # (B, C, H, W)
