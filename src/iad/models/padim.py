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
