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

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Choose from {list(_BACKBONES)}."
            )
        self.backbone_name = backbone
        self.layers = layers or ["layer2", "layer3"]

        ctor, weights_enum_name = _BACKBONES[backbone]
        weights = None
        if pretrained:
            weights = getattr(models, weights_enum_name).IMAGENET1K_V1
        self.backbone = ctor(weights=weights)

        # Freeze: eval mode + no grad. This is what makes the methods
        # "training-free" — the backbone is never updated.
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        self._features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in self.layers:
