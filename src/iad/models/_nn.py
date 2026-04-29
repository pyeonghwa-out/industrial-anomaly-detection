"""Nearest-neighbour search backend for PatchCore.

Uses FAISS when available (fast, used on Colab/Kaggle Linux) and falls back to a
chunked ``torch.cdist`` implementation otherwise (e.g. on macOS where
``faiss-cpu`` wheels are flaky). The two paths return identical results.
"""
from __future__ import annotations

import torch

try:  # pragma: no cover - exercised implicitly by environment
    import faiss

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    _HAS_FAISS = False


class NearestNeighbors:
    """Exact L2 nearest-neighbour index over a fixed memory bank."""

    def __init__(self, bank: torch.Tensor, chunk: int = 4096) -> None:
        self.chunk = chunk
        self.dim = bank.shape[1]
        if _HAS_FAISS:
            self._index = faiss.IndexFlatL2(self.dim)
            self._index.add(bank.detach().cpu().numpy().astype("float32"))
