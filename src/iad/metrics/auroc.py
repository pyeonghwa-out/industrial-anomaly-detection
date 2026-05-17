"""AUROC metrics for detection (image-level) and localization (pixel-level)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level detection AUROC.

    Args:
        scores: ``(N,)`` anomaly score per image (higher = more anomalous).
        labels: ``(N,)`` binary labels (0=good, 1=defect).
    """
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
