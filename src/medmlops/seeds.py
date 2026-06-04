"""Centralized seed and CPU determinism recipe.

Honest limit: completely reproducible results are not guaranteed across PyTorch
releases, commits, platforms, or between CPU and GPU. The defensible claim for
this project is bit-reproducibility within the pinned Docker image on a given
CPU architecture. See docs/adr/0004-cpu-determinism-limits.md.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_deterministic(seed: int = 42) -> None:
    """Set the Phase 1 deterministic execution defaults."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
