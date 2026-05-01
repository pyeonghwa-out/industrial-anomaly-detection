"""Frozen ImageNet backbone with multi-layer feature hooks.

Both PatchCore and PaDiM treat a pretrained CNN as a *fixed* feature encoder —
no gradients, no fine-tuning. We register forward hooks on the requested ResNet
blocks (``layer1``..``layer3``) and return their activations.

Supported backbones: ``resnet18``, ``resnet50``, ``wide_resnet50_2``.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models

_BACKBONES = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "wide_resnet50_2": (models.wide_resnet50_2, "Wide_ResNet50_2_Weights"),
}


class FeatureExtractor(nn.Module):
    """Extract intermediate feature maps from a frozen ResNet-family backbone.

    Args:
        backbone: One of ``resnet18``, ``resnet50``, ``wide_resnet50_2``.
        layers: Block names to tap, e.g. ``["layer2", "layer3"]``.
        pretrained: Load ImageNet weights (always True in practice; exposed for
            fast unit tests that want random weights).
    """
