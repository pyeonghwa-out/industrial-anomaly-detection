"""PRO — Per-Region Overlap score.

PRO is the localization metric MVTec recommends because pixel-AUROC is biased
towards large defects (a few huge regions dominate the pixel count). PRO instead
gives *every connected defect region equal weight*, regardless of size.

Definition (Bergmann et al.):
  * Sweep an anomaly-map threshold.
  * At each threshold, for every connected ground-truth region compute the
    fraction of that region that is correctly flagged; average these fractions
    -> PRO at this threshold.
  * Also compute the false-positive rate over defect-free pixels.
  * Plot PRO vs FPR and integrate up to a maximum FPR (default 0.3), then
    normalise by that limit. Higher is better; 1.0 is perfect.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label as cc_label
from sklearn.metrics import auc


def compute_pro(
    maps: np.ndarray,
    masks: np.ndarray,
    num_thresholds: int = 200,
    fpr_limit: float = 0.3,
) -> float:
    """Compute the normalised area under the PRO curve up to ``fpr_limit``."""
    maps = np.asarray(maps, dtype=np.float32)
    masks = (np.asarray(masks) > 0.5).astype(np.uint8)

    # Pre-label connected components once per image.
    regions = []  # list of (image_index, region_mask)
    for i in range(masks.shape[0]):
        if masks[i].sum() == 0:
            continue
