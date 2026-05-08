"""
Evaluate RL policy with real-world dataset
"""

import os
import sys

# Add parent directory to path for imports FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import config

from env.smart_grid_env import SmartGridEnv
from evaluation.baseline_controller import BaselineController
from evaluation.evaluate import evaluate_policy
from utils.logger import ensure_project_dirs, get_logger, log_metrics


def evaluate_all_real_data():
    """Evaluate both RL policy and baseline controller with real-world data"""
    logger = get_logger("evaluation.real_data")
    ensure_project_dirs()
    
    # Check if dataset exists
    dataset_path = config.REAL_DATASET_PATH
    if not os.path.exists(dataset_path):
        logger.error("Real-world dataset not found: %s", dataset_path)
        logger.error("Please run first: python data/download_dataset.py")
        return

    logger.info("Evaluating policies with real-world data")
    
    # Create environments with real data
    rl_env = SmartGridEnv(use_real_data=True, dataset_path=dataset_path)
    rl_env = DummyVecEnv([lambda: rl_env])
    
    baseline_env = SmartGridEnv(use_real_data=True, dataset_path=dataset_path)
    baseline_env = DummyVecEnv([lambda: baseline_env])
    
    # Load RL model
    model_path = str(config.MODELS_DIR / config.BEST_MODEL_REAL_DATA_FILENAME)
    if not os.path.exists(model_path):
        model_path = str(config.MODELS_DIR / config.FINAL_MODEL_REAL_DATA_FILENAME)
    
    if not os.path.exists(model_path):
        logger.warning("Real-data model file not found at %s", model_path)
        logger.warning("Please train the model first using train/train_ppo_real_data.py")
        return
    
    logger.info("Loading RL model from: %s", model_path)
    rl_model = PPO.load(model_path)
    
    # Create baseline controller
    baseline_unwrapped = baseline_env.envs[0].unwrapped if hasattr(baseline_env, 'envs') else baseline_env.unwrapped
    baseline_controller = BaselineController(baseline_unwrapped)
    
    # Evaluate RL policy
    logger.info("Evaluating RL policy (real-world data)...")
    rl_metrics, rl_traces = evaluate_policy(
        rl_env, rl_model, n_episodes=config.N_EVAL_EPISODES, policy_name="RL_PPO_RealData"
    )
    
    log_metrics(
        logger,
        "RL policy metrics (real-world data)",
        {
            "Mean Reward": f"{rl_metrics['mean_reward']:.2f} ± {rl_metrics['std_reward']:.2f}",
            "Mean Peak Demand (kW)": f"{rl_metrics['mean_peak_demand']:.2f} ± {rl_metrics['std_peak_demand']:.2f}",
            "Mean Total Cost (INR)": f"{rl_metrics['mean_total_cost'] * config.USD_TO_INR:.2f} ± {rl_metrics['std_total_cost'] * config.USD_TO_INR:.2f}",
            "Mean Min Comfort": f"{rl_metrics['mean_min_comfort']:.3f} ± {rl_metrics['std_min_comfort']:.3f}",
            "Final Battery Health": f"{rl_metrics['mean_final_battery_health']:.4f} ± {rl_metrics['std_final_battery_health']:.4f}",
            "Final Cycle Count": f"{rl_metrics['mean_final_cycle_count']:.4f} ± {rl_metrics['std_final_cycle_count']:.4f}",
        },
    )
    
    # Evaluate baseline
    logger.info("Evaluating baseline controller (real-world data)...")
    baseline_metrics, baseline_traces = evaluate_policy(
        baseline_env, baseline_controller, n_episodes=config.N_EVAL_EPISODES, policy_name="Baseline_RealData"
    )
    
    log_metrics(
        logger,
        "Baseline metrics (real-world data)",
        {
            "Mean Reward": f"{baseline_metrics['mean_reward']:.2f} ± {baseline_metrics['std_reward']:.2f}",
            "Mean Peak Demand (kW)": f"{baseline_metrics['mean_peak_demand']:.2f} ± {baseline_metrics['std_peak_demand']:.2f}",
            "Mean Total Cost (INR)": f"{baseline_metrics['mean_total_cost'] * config.USD_TO_INR:.2f} ± {baseline_metrics['std_total_cost'] * config.USD_TO_INR:.2f}",
            "Mean Min Comfort": f"{baseline_metrics['mean_min_comfort']:.3f} ± {baseline_metrics['std_min_comfort']:.3f}",
            "Final Battery Health": f"{baseline_metrics['mean_final_battery_health']:.4f} ± {baseline_metrics['std_final_battery_health']:.4f}",
            "Final Cycle Count": f"{baseline_metrics['mean_final_cycle_count']:.4f} ± {baseline_metrics['std_final_cycle_count']:.4f}",
        },
    )
    
    # Save summary
    summary_data = {
        'Policy': [rl_metrics['policy_name'], baseline_metrics['policy_name']],
        'Mean_Reward': [rl_metrics['mean_reward'], baseline_metrics['mean_reward']],
        'Std_Reward': [rl_metrics['std_reward'], baseline_metrics['std_reward']],
        'Mean_Peak_Demand_kW': [rl_metrics['mean_peak_demand'], baseline_metrics['mean_peak_demand']],
        'Std_Peak_Demand_kW': [rl_metrics['std_peak_demand'], baseline_metrics['std_peak_demand']],
        'Mean_Total_Cost': [rl_metrics['mean_total_cost'], baseline_metrics['mean_total_cost']],
        'Std_Total_Cost': [rl_metrics['std_total_cost'], baseline_metrics['std_total_cost']],
        'Mean_Min_Comfort': [rl_metrics['mean_min_comfort'], baseline_metrics['mean_min_comfort']],
        'Std_Min_Comfort': [rl_metrics['std_min_comfort'], baseline_metrics['std_min_comfort']],
        'Mean_Final_Battery_Health': [rl_metrics['mean_final_battery_health'], baseline_metrics['mean_final_battery_health']],
        'Std_Final_Battery_Health': [rl_metrics['std_final_battery_health'], baseline_metrics['std_final_battery_health']],
        'Mean_Final_Cycle_Count': [rl_metrics['mean_final_cycle_count'], baseline_metrics['mean_final_cycle_count']],
        'Std_Final_Cycle_Count': [rl_metrics['std_final_cycle_count'], baseline_metrics['std_final_cycle_count']]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = str(config.RESULTS_LOGS_DIR / config.SUMMARY_REAL_DATA_FILENAME)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary saved to: %s", summary_path)
    
    # Save RL episode trace
    if rl_traces and len(rl_traces[0]) > 0:
        trace_df = pd.DataFrame(rl_traces[0])
        trace_path = str(config.RESULTS_LOGS_DIR / config.EPISODE_TRACE_REAL_DATA_FILENAME)
        trace_df.to_csv(trace_path, index=False)
        logger.info("RL episode trace saved to: %s", trace_path)
    
    # Generate plots
    logger.info("Generating plots...")
    from evaluation.plot_results import plot_results
    plot_results(rl_traces[0] if rl_traces else None)
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    evaluate_all_real_data()
