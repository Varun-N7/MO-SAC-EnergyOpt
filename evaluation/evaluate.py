"""
Evaluate RL policy and baseline controller
Compare performance and save results
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
from utils.logger import ensure_project_dirs, get_logger, log_metrics


def evaluate_policy(env, policy, n_episodes=10, policy_name="policy"):
    """
    Evaluate a policy over multiple episodes
    
    Args:
        env: Environment instance
        policy: Policy to evaluate (RL model or baseline controller)
        n_episodes: Number of episodes to run
        policy_name: Name of policy for logging
        
    Returns:
        metrics: Dictionary with evaluation metrics
        episode_traces: List of episode traces
    """
    all_episode_rewards = []
    all_peak_demands = []
    all_total_costs = []
    all_min_comforts = []
    all_final_battery_healths = []
    all_final_cycle_counts = []
    episode_traces = []
    
    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        
        # Handle vectorized format
        if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
            obs = obs[0]
        if isinstance(info, list) and len(info) > 0:
            info = info[0]
        
        done = False
        episode_reward = 0.0
        episode_trace = []
        peak_demand = 0.0
        total_cost = 0.0
        min_comfort = 1.0
        final_battery_health = 1.0
        final_cycle_count = 0.0
        
        while not done:
            # Get action
            # Extract observation if vectorized
            obs_for_policy = obs
            if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                obs_for_policy = obs[0]
            
            if hasattr(policy, 'predict'):
                # Check if it's RL model (returns tuple) or baseline (returns single value)
                try:
                    pred_result = policy.predict(obs_for_policy, deterministic=True)
                    if isinstance(pred_result, tuple):
                        action, _ = pred_result
                    else:
                        action = pred_result
                except TypeError:
                    # Baseline controller doesn't take deterministic parameter
                    action = policy.predict(obs_for_policy)
            else:
                # Baseline controller
                action = policy.predict(obs_for_policy)
            
            # Step environment
            # For vectorized env, action needs to be 1D array
            if isinstance(action, (int, np.integer)):
                action_array = np.array([action], dtype=np.int32)
            elif isinstance(action, np.ndarray):
                if action.ndim == 0:
                    action_array = np.array([action], dtype=np.int32)
                else:
                    action_array = action.astype(np.int32)
            else:
                action_array = np.array([action], dtype=np.int32)
            
            step_result = env.step(action_array)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                obs, reward, done = step_result[:3]
                truncated = False
                info = step_result[3] if len(step_result) > 3 else {}
            
            # Extract from vectorized format
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            
            # Track metrics
            episode_reward += reward
            grid_import = info.get('grid_import', 0.0)
            peak_demand = max(peak_demand, grid_import)
            total_cost += grid_import * info.get('price', 0.0)
            min_comfort = min(min_comfort, info.get('comfort', 1.0))
            final_battery_health = info.get('battery_health', final_battery_health)
            final_cycle_count = info.get('cycle_count', final_cycle_count)
            
            # Store trace (use previous hour's info)
            unwrapped_env = env.envs[0].unwrapped if hasattr(env, 'envs') else env.unwrapped
            if hasattr(unwrapped_env, 'episode_history') and len(unwrapped_env.episode_history) > 0:
                last_state = unwrapped_env.episode_history[-1]
                episode_trace.append({
                    'hour': last_state['hour'],
                    'pv': last_state['pv'],
                    'base_load': last_state['base_load'],
                    'grid_import': last_state['grid_import'],
                    'soc': last_state['soc'],
                    'comfort': last_state['comfort'],
                    'battery_health': last_state.get('battery_health', np.nan),
                    'cycle_count': last_state.get('cycle_count', np.nan),
                    'action': last_state['action'],
                    'reward': last_state['reward']
                })
        
        all_episode_rewards.append(episode_reward)
        all_peak_demands.append(peak_demand)
        all_total_costs.append(total_cost)
        all_min_comforts.append(min_comfort)
        all_final_battery_healths.append(final_battery_health)
        all_final_cycle_counts.append(final_cycle_count)
        
        if episode == 0:  # Save first episode trace
            episode_traces.append(episode_trace)
    
    metrics = {
        'mean_reward': np.mean(all_episode_rewards),
        'std_reward': np.std(all_episode_rewards),
        'mean_peak_demand': np.mean(all_peak_demands),
        'std_peak_demand': np.std(all_peak_demands),
        'mean_total_cost': np.mean(all_total_costs),
        'std_total_cost': np.std(all_total_costs),
        'mean_min_comfort': np.mean(all_min_comforts),
        'std_min_comfort': np.std(all_min_comforts),
        'mean_final_battery_health': np.mean(all_final_battery_healths),
        'std_final_battery_health': np.std(all_final_battery_healths),
        'mean_final_cycle_count': np.mean(all_final_cycle_counts),
        'std_final_cycle_count': np.std(all_final_cycle_counts),
        'policy_name': policy_name
    }
    
    return metrics, episode_traces


def evaluate_all():
    """Evaluate both RL policy and baseline controller"""
    logger = get_logger("evaluation.synthetic")
    ensure_project_dirs()
    logger.info("Evaluating policies")
    
    # Create environments
    rl_env = SmartGridEnv()
    rl_env = DummyVecEnv([lambda: rl_env])
    
    baseline_env = SmartGridEnv()
    baseline_env = DummyVecEnv([lambda: baseline_env])
    
    # Load RL model
    model_path = str(config.MODELS_DIR / config.BEST_MODEL_FILENAME)
    if not os.path.exists(model_path):
        model_path = str(config.MODELS_DIR / config.FINAL_MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        logger.warning("Model file not found at %s", model_path)
        logger.warning("Please train the model first using train/train_ppo.py")
        return
    
    logger.info("Loading RL model from: %s", model_path)
    rl_model = PPO.load(model_path)
    
    # Create baseline controller
    baseline_unwrapped = baseline_env.envs[0].unwrapped if hasattr(baseline_env, 'envs') else baseline_env.unwrapped
    baseline_controller = BaselineController(baseline_unwrapped)
    
    # Evaluate RL policy
    logger.info("Evaluating RL policy...")
    rl_metrics, rl_traces = evaluate_policy(
        rl_env, rl_model, n_episodes=config.N_EVAL_EPISODES, policy_name="RL_PPO"
    )
    
    log_metrics(
        logger,
        "RL policy metrics",
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
    logger.info("Evaluating baseline controller...")
    baseline_metrics, baseline_traces = evaluate_policy(
        baseline_env, baseline_controller, n_episodes=config.N_EVAL_EPISODES, policy_name="Baseline"
    )
    
    log_metrics(
        logger,
        "Baseline metrics",
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
    summary_path = str(config.RESULTS_LOGS_DIR / config.SUMMARY_FILENAME)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary saved to: %s", summary_path)
    
    # Save RL episode trace
    if rl_traces and len(rl_traces[0]) > 0:
        trace_df = pd.DataFrame(rl_traces[0])
        trace_path = str(config.RESULTS_LOGS_DIR / config.EPISODE_TRACE_FILENAME)
        trace_df.to_csv(trace_path, index=False)
        logger.info("RL episode trace saved to: %s", trace_path)
    
    # Generate plots
    logger.info("Generating plots...")
    from evaluation.plot_results import plot_results
    plot_results(rl_traces[0] if rl_traces else None)
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    evaluate_all()

