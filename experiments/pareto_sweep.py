"""
Pareto sweep for reward-weight trade-offs.

Runs PPO training across a grid of (lambda_cost, lambda_peak, lambda_comfort),
evaluates each policy, and saves objective metrics to JSON.
"""

import itertools
import json
import os
import sys
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path for imports FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from env.smart_grid_env import SmartGridEnv
from evaluation.evaluate import evaluate_policy
from utils.logger import ensure_project_dirs, get_logger
from seeds import set_global_seed


def _run_single_setting(setting_id, reward_weights, timesteps, n_eval_episodes, use_real_data=False, seed=42):
    """Train and evaluate one weight setting."""
    logger = get_logger("experiments.pareto")
    monitor_dir = config.RESULTS_LOGS_DIR / "pareto" / f"setting_{setting_id:03d}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "reward_weights": reward_weights,
        "use_real_data": use_real_data,
        "dataset_path": config.REAL_DATASET_PATH if use_real_data else None,
    }

    train_env = SmartGridEnv(**env_kwargs)
    train_env = Monitor(train_env, str(monitor_dir))
    train_env = DummyVecEnv([lambda: train_env])

    eval_env = SmartGridEnv(**env_kwargs)
    eval_env = DummyVecEnv([lambda: eval_env])

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        verbose=0,
        tensorboard_log=str(config.TENSORBOARD_DIR / "pareto"),
        seed=seed,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)

    model_path = config.MODELS_DIR / f"ppo_pareto_{setting_id:03d}.zip"
    model.save(str(model_path))

    metrics, _ = evaluate_policy(
        eval_env, model, n_episodes=n_eval_episodes, policy_name=f"pareto_{setting_id:03d}"
    )

    logger.info(
        "Setting %03d done | weights(cost=%.2f, peak=%.2f, comfort=%.2f) | "
        "cost=%.3f, peak=%.3f, comfort=%.3f, reward=%.3f",
        setting_id,
        reward_weights["cost"],
        reward_weights["peak"],
        reward_weights["comfort"],
        metrics["mean_total_cost"],
        metrics["mean_peak_demand"],
        metrics["mean_min_comfort"],
        metrics["mean_reward"],
    )

    return {
        "setting_id": setting_id,
        "weights": {
            "lambda_cost": float(reward_weights["cost"]),
            "lambda_peak": float(reward_weights["peak"]),
            "lambda_comfort": float(reward_weights["comfort"]),
        },
        "metrics": {
            "mean_total_cost": float(metrics["mean_total_cost"]),
            "mean_peak_demand": float(metrics["mean_peak_demand"]),
            "mean_min_comfort": float(metrics["mean_min_comfort"]),
            "mean_reward": float(metrics["mean_reward"]),
            "std_reward": float(metrics["std_reward"]),
        },
    }


def run_pareto_sweep(
    weight_grid=None,
    timesteps=None,
    n_eval_episodes=None,
    use_real_data=False,
    seed=42,
):
    """Run full weight-grid sweep and save Pareto dataset."""
    logger = get_logger("experiments.pareto")
    ensure_project_dirs()
    set_global_seed(seed)

    weight_grid = weight_grid or config.PARETO_WEIGHT_GRID
    timesteps = timesteps or config.PARETO_TIMESTEPS
    n_eval_episodes = n_eval_episodes or config.PARETO_EVAL_EPISODES

    combinations = list(itertools.product(weight_grid, weight_grid, weight_grid))
    if len(combinations) < 1:
        raise ValueError("Pareto sweep requires at least one weight combination.")

    logger.info("Starting Pareto sweep with %d combinations", len(combinations))

    results = []
    for idx, (lambda_cost, lambda_peak, lambda_comfort) in enumerate(combinations, start=1):
        reward_weights = {
            "cost": float(lambda_cost),
            "peak": float(lambda_peak),
            "comfort": float(lambda_comfort),
            "battery": 0.0,
        }
        result = _run_single_setting(
            setting_id=idx,
            reward_weights=reward_weights,
            timesteps=timesteps,
            n_eval_episodes=n_eval_episodes,
            use_real_data=use_real_data,
            seed=seed,
        )
        results.append(result)

    output = {
        "weight_grid": [float(v) for v in weight_grid],
        "timesteps": timesteps,
        "n_eval_episodes": n_eval_episodes,
        "use_real_data": use_real_data,
        "num_combinations": len(combinations),
        "results": results,
    }

    output_path = config.RESULTS_DIR / config.PARETO_DATA_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Pareto data saved to: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pareto weight sweep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training timesteps per setting (default: config.PARETO_TIMESTEPS).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Evaluation episodes per setting (default: config.PARETO_EVAL_EPISODES).",
    )
    parser.add_argument(
        "--weight-grid",
        type=str,
        default=None,
        help="Comma-separated weight values, e.g. 0.5,1.0,1.5",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use configured real-world dataset instead of synthetic data.",
    )
    args = parser.parse_args()
    weight_grid = None
    if args.weight_grid:
        weight_grid = [float(v.strip()) for v in args.weight_grid.split(",") if v.strip()]
    run_pareto_sweep(
        weight_grid=weight_grid,
        timesteps=args.timesteps,
        n_eval_episodes=args.n_eval_episodes,
        use_real_data=args.use_real_data,
        seed=args.seed,
    )
