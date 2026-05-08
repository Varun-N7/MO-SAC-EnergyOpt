"""
Ablation study on reward-objective weights.

Runs four settings:
1) Peak-only objective
2) Cost-only objective
3) Comfort-only objective
4) Multi-objective (peak + cost + comfort)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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


def _load_monitor_rewards(monitor_path):
    """Load episode rewards from Stable-Baselines monitor.csv."""
    if not os.path.exists(monitor_path):
        return []
    try:
        df = pd.read_csv(monitor_path, comment="#")
        if "r" not in df.columns:
            return []
        return df["r"].tolist()
    except Exception:
        return []


def _train_and_evaluate(setting_name, reward_weights, timesteps, n_eval_episodes, use_real_data, dataset_path, logger, seed):
    """Train PPO for one ablation setting and evaluate."""
    ablation_root = config.RESULTS_LOGS_DIR / "ablation" / setting_name
    ablation_root.mkdir(parents=True, exist_ok=True)
    monitor_path = str(ablation_root / config.MONITOR_FILENAME)

    env_kwargs = {
        "reward_weights": reward_weights,
        "use_real_data": use_real_data,
        "dataset_path": dataset_path if use_real_data else None,
    }

    train_env = SmartGridEnv(**env_kwargs)
    train_env = Monitor(train_env, monitor_path)
    train_env = DummyVecEnv([lambda: train_env])

    eval_env = SmartGridEnv(**env_kwargs)
    eval_env = DummyVecEnv([lambda: eval_env])

    logger.info("Training setting=%s, timesteps=%s", setting_name, timesteps)
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
        tensorboard_log=str(config.TENSORBOARD_DIR / "ablation"),
        seed=seed,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)

    model_path = config.MODELS_DIR / f"ppo_ablation_{setting_name}.zip"
    model.save(str(model_path))

    metrics, _ = evaluate_policy(eval_env, model, n_episodes=n_eval_episodes, policy_name=setting_name)
    rewards = _load_monitor_rewards(monitor_path)

    summary_row = {
        "setting": setting_name,
        "peak_weight": reward_weights["peak"],
        "cost_weight": reward_weights["cost"],
        "comfort_weight": reward_weights["comfort"],
        "battery_weight": reward_weights["battery"],
        "mean_reward": metrics["mean_reward"],
        "std_reward": metrics["std_reward"],
        "mean_peak_demand_kW": metrics["mean_peak_demand"],
        "mean_total_cost_usd": metrics["mean_total_cost"],
        "mean_min_comfort": metrics["mean_min_comfort"],
    }

    curve_rows = [
        {
            "setting": setting_name,
            "episode": idx + 1,
            "episode_reward": reward,
        }
        for idx, reward in enumerate(rewards)
    ]
    return summary_row, curve_rows


def run_ablation_study(timesteps=None, n_eval_episodes=None, use_real_data=False, seed=42):
    logger = get_logger("experiments.ablation")
    ensure_project_dirs()
    set_global_seed(seed)

    timesteps = timesteps or config.TOTAL_TIMESTEPS
    n_eval_episodes = n_eval_episodes or config.N_EVAL_EPISODES
    dataset_path = config.REAL_DATASET_PATH if use_real_data else None

    if use_real_data and not os.path.exists(config.REAL_DATASET_PATH):
        logger.error("Real dataset missing at %s", config.REAL_DATASET_PATH)
        return

    settings = [
        ("peak_only", {"peak": 1.0, "cost": 0.0, "comfort": 0.0, "battery": 0.0}),
        ("cost_only", {"peak": 0.0, "cost": 1.0, "comfort": 0.0, "battery": 0.0}),
        ("comfort_only", {"peak": 0.0, "cost": 0.0, "comfort": 1.0, "battery": 0.0}),
        ("multi_objective", {"peak": 1.0, "cost": 1.0, "comfort": 1.0, "battery": 0.0}),
    ]

    summary_rows = []
    curve_rows = []

    for setting_name, weights in settings:
        row, curves = _train_and_evaluate(
            setting_name=setting_name,
            reward_weights=weights,
            timesteps=timesteps,
            n_eval_episodes=n_eval_episodes,
            use_real_data=use_real_data,
            dataset_path=dataset_path,
            logger=logger,
            seed=seed,
        )
        summary_rows.append(row)
        curve_rows.extend(curves)

    results_path = config.RESULTS_DIR / "ablation_results.csv"
    curves_path = config.RESULTS_DIR / "ablation_training_curves.csv"

    pd.DataFrame(summary_rows).to_csv(results_path, index=False)
    pd.DataFrame(curve_rows).to_csv(curves_path, index=False)

    logger.info("Ablation summary saved to: %s", results_path)
    logger.info("Ablation training curves saved to: %s", curves_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reward-weight ablation study.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training timesteps per setting (default: config.TOTAL_TIMESTEPS).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Evaluation episodes per setting (default: config.N_EVAL_EPISODES).",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use configured real-world dataset instead of synthetic data.",
    )
    args = parser.parse_args()
    run_ablation_study(
        timesteps=args.timesteps,
        n_eval_episodes=args.n_eval_episodes,
        use_real_data=args.use_real_data,
        seed=args.seed,
    )
