"""Greedy k-center coreset subsampling for PatchCore.

PatchCore stores a *coreset* of the training patch features rather than the
full set. The coreset is chosen by greedy furthest-point (k-center) sampling so
that the retained points cover the feature manifold as evenly as possible.

To make the O(N x M) distance computation affordable, the original paper first
projects features to a lower dimension with a sparse Gaussian random matrix
(Johnson-Lindenstrauss). We reproduce that here.
"""
from __future__ import annotations

import torch
from tqdm.auto import tqdm


class KCenterGreedy:
    """Greedy k-center (furthest-point) coreset selection.

    Args:
        projection_dim: Target dimensionality of the random projection used only
            to *measure* distances. ``None`` disables projection.
        seed: RNG seed for the projection and the first centre — keeps coresets
            reproducible across runs.
    """

