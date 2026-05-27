"""Generate the heatmap gallery used in the README.

Fits a method on one category and renders the 4-panel
(input | GT | heatmap | overlay) figure for a few defect images.

Example::

    python scripts/generate_figures.py --config configs/patchcore.yaml \
        --category bottle --num-images 4 --out results/figures/bottle_patchcore.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from iad.datasets.mvtec import MVTecDataset
from iad.runner import build_model
from iad.utils.config import load_config
from iad.utils.seed import set_seed
from iad.visualize import save_comparison_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Render anomaly heatmap grids.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--root", help="Override dataset root.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

