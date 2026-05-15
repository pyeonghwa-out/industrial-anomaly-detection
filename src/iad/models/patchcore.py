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
