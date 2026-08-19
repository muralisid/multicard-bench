"""Determinism helpers. Every experiment fixes its seeds through here."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed every source of randomness the pipeline touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def rng(seed: int) -> np.random.Generator:
    """A local generator, preferred over global state for data construction."""
    return np.random.default_rng(seed)
