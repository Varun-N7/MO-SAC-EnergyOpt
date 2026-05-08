"""
Algorithm comparison experiment: PPO vs SAC.

Trains both algorithms under matched settings, evaluates on the same episode count,
and saves summary + training curves.
"""

import os
import sys
import time
import argparse

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path for imports FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.sac_agent import make_sac_env, build_sac_model
from env.smart_grid_env import SmartGridEnv
from evaluation.evaluate import evaluate_policy
from utils.logger import ensure_project_dirs, get_logger
from seeds import set_global_seed


def _load_monitor_rewards(monitor_path):
    if not os.path.exists(monitor_path):
        return []
    try:
        df = pd.read_csv(monitor_path, comment="#")
        if "r" not in df.columns:
            return []
        return df["r"].tolist()
    except Exception:
        return []


def _train_ppo(timesteps, n_eval_episodes, use_real_data, dataset_path, logger, seed):
    monitor_path = str(config.RESULTS_LOGS_DIR / "monitor_ppo_compare.csv")
    train_env = SmartGridEnv(use_real_data=use_real_data, dataset_path=dataset_path)
    train_env = Monitor(train_env, monitor_path)
    train_env = DummyVecEnv([lambda: train_env])

    eval_env = SmartGridEnv(use_real_data=use_real_data, dataset_path=dataset_path)
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
        tensorboard_log=str(config.TENSORBOARD_DIR / "algorithm_comparison"),
        seed=seed,
    )

    logger.info("Training PPO for %s timesteps", timesteps)
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    elapsed = time.perf_counter() - start

    model.save(str(config.MODELS_DIR / "ppo_algorithm_compare.zip"))
    metrics, _ = evaluate_policy(eval_env, model, n_episodes=n_eval_episodes, policy_name="PPO")
    rewards = _load_monitor_rewards(monitor_path)
    return metrics, rewards, elapsed


def _train_sac(timesteps, n_eval_episodes, use_real_data, dataset_path, logger, seed):
    monitor_path = str(config.RESULTS_LOGS_DIR / "monitor_sac_compare.csv")
    train_env = make_sac_env(
        use_real_data=use_real_data,
        dataset_path=dataset_path,
        monitor_path=monitor_path,
    )
    eval_env = make_sac_env(
        use_real_data=use_real_data,
        dataset_path=dataset_path,
        monitor_path=None,
    )

    model = build_sac_model(train_env, seed=seed)
    logger.info("Training SAC for %s timesteps", timesteps)
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=True)
    elapsed = time.perf_counter() - start

    model.save(str(config.MODELS_DIR / "sac_algorithm_compare.zip"))
    metrics, _ = evaluate_policy(eval_env, model, n_episodes=n_eval_episodes, policy_name="SAC")
    rewards = _load_monitor_rewards(monitor_path)
    return metrics, rewards, elapsed


def run_algorithm_comparison(timesteps=None, n_eval_episodes=None, use_real_data=False, seed=42):
    logger = get_logger("experiments.compare_algorithms")
    ensure_project_dirs()
    set_global_seed(seed)

    timesteps = timesteps or config.TOTAL_TIMESTEPS
    n_eval_episodes = n_eval_episodes or config.N_EVAL_EPISODES
    dataset_path = config.REAL_DATASET_PATH if use_real_data else None

    if use_real_data and not os.path.exists(config.REAL_DATASET_PATH):
        logger.error("Real dataset missing at %s", config.REAL_DATASET_PATH)
        return

    ppo_metrics, ppo_rewards, ppo_time = _train_ppo(
        timesteps, n_eval_episodes, use_real_data, dataset_path, logger, seed
    )
    sac_metrics, sac_rewards, sac_time = _train_sac(
        timesteps, n_eval_episodes, use_real_data, dataset_path, logger, seed
    )

    summary_df = pd.DataFrame(
        [
            {
                "algorithm": "PPO",
                "mean_reward": ppo_metrics["mean_reward"],
                "mean_cost": ppo_metrics["mean_total_cost"],
                "mean_peak_demand": ppo_metrics["mean_peak_demand"],
                "mean_comfort": ppo_metrics["mean_min_comfort"],
                "training_time_sec": ppo_time,
            },
            {
                "algorithm": "SAC",
                "mean_reward": sac_metrics["mean_reward"],
                "mean_cost": sac_metrics["mean_total_cost"],
                "mean_peak_demand": sac_metrics["mean_peak_demand"],
                "mean_comfort": sac_metrics["mean_min_comfort"],
                "training_time_sec": sac_time,
            },
        ]
    )
    summary_path = config.RESULTS_DIR / "algorithm_comparison.csv"
    summary_df.to_csv(summary_path, index=False)

    curve_rows = []
    for idx, reward in enumerate(ppo_rewards):
        curve_rows.append({"algorithm": "PPO", "episode": idx + 1, "episode_reward": reward})
    for idx, reward in enumerate(sac_rewards):
        curve_rows.append({"algorithm": "SAC", "episode": idx + 1, "episode_reward": reward})
    curves_df = pd.DataFrame(curve_rows)
    curves_path = config.RESULTS_DIR / "algorithm_training_curves.csv"
    curves_df.to_csv(curves_path, index=False)

    logger.info("Algorithm comparison saved to: %s", summary_path)
    logger.info("Training curves saved to: %s", curves_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PPO vs SAC.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Training timesteps per algorithm (default: config.TOTAL_TIMESTEPS).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Evaluation episodes per algorithm (default: config.N_EVAL_EPISODES).",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use configured real-world dataset instead of synthetic data.",
    )
    args = parser.parse_args()
    run_algorithm_comparison(
        timesteps=args.timesteps,
        n_eval_episodes=args.n_eval_episodes,
        use_real_data=args.use_real_data,
        seed=args.seed,
    )
