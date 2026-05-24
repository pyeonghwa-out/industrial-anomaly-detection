"""Plot a benchmark results CSV as a per-category AUROC bar chart.

Reads the CSV written by ``run_benchmark.py`` and renders a grouped bar chart of
image-level and pixel-level AUROC per category (the MEAN row is highlighted).
Needs only pandas + matplotlib — no GPU, no model, no dataset.

Example::

    python scripts/plot_results.py --csv results/runs/patchcore_results.csv \
        --out results/figures/patchcore_results.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Bar chart of benchmark AUROC results.")
    parser.add_argument("--csv", required=True, help="Path to <method>_results.csv.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument("--title", default=None, help="Optional chart title.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    method = str(df.get("method", pd.Series(["model"])).iloc[0])
    out = Path(args.out or Path(args.csv).with_suffix("").name + "_results.png")
    if args.out is None:
        out = Path("results/figures") / out.name

    cats = df["category"].tolist()
