"""Reproducibility helpers for global and environment-level seeding."""

import random
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def seed_env(env, seed: int) -> None:
    """Seed a Gym/Gymnasium environment if supported."""
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
