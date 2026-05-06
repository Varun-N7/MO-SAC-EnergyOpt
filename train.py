"""Unified training entrypoint.

Usage:
  python train.py --algorithm ppo --seed 42
  python train.py --algorithm sac --seed 42
"""

import argparse
import importlib.util
import os
import sys

# Expose train/ directory as package path when imported as `train`.
__path__ = [os.path.join(os.path.dirname(__file__), "train")]

import config
from utils.logger import get_logger


def _load_callable(module_path, func_name):
    spec = importlib.util.spec_from_file_location(func_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)


def main():
    parser = argparse.ArgumentParser(description="Train MO-RL-PeakShaving model.")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    parser.add_argument("--use-real-data", action="store_true")
    args = parser.parse_args()

    if args.algorithm == "ppo":
        train_ppo = _load_callable(
            os.path.join(os.path.dirname(__file__), "train", "train_ppo.py"),
            "train_ppo",
        )
        train_ppo(seed=args.seed)
    else:
        run_sac = _load_callable(
            os.path.join(os.path.dirname(__file__), "experiments", "compare_algorithms.py"),
            "_train_sac",
        )
        logger = get_logger("train.sac")
        run_sac(
            timesteps=args.timesteps,
            use_real_data=args.use_real_data,
            dataset_path=config.REAL_DATASET_PATH if args.use_real_data else None,
            logger=logger,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
