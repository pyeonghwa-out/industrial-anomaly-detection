"""High-level helpers that turn a config into a fitted model and metrics.

Keeps the CLI scripts and notebooks tiny: they just load a YAML and call
``run_experiment``.
"""
from __future__ import annotations

import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from .datasets.mvtec import MVTecDataset
from .metrics import evaluate
from .models.autoencoder import ConvAutoencoder
from .models.padim import PaDiM
from .models.patchcore import PatchCore
from .utils.config import Config
from .utils.seed import set_seed

_METHODS = {"patchcore": PatchCore, "padim": PaDiM, "autoencoder": ConvAutoencoder}


def build_model(config: Config, device: str):
    """Instantiate a model from a config's ``method`` and ``model`` sections."""
    method = config["method"]
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_METHODS)}.")
    params = dict(config.get("model", {}))
    params["device"] = device
    params["seed"] = config.get("seed", 42)
    params["image_size"] = config.get("data", {}).get("crop", 224)
    return _METHODS[method](**params)


def make_loaders(config: Config, category: str):
