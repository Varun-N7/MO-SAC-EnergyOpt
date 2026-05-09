"""
Quick demo: Train for a short time and show evaluation metrics
This demonstrates RL metrics (not accuracy) in action
"""

import os
import sys
sys.path.insert(0, '.')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env.smart_grid_env import SmartGridEnv
from evaluation.baseline_controller import BaselineController
from evaluation.evaluate import evaluate_policy
import config
from utils.logger import ensure_project_dirs

print("=" * 70)
print("QUICK DEMO: RL Metrics in Action")
print("=" * 70)
print("\nThis demo shows:")
print("  - No dataset needed (environment generates data)")
print("  - Metrics: Reward, Peak Demand, Cost, Comfort (not accuracy)")
print("  - Comparison: RL agent vs Baseline controller")
print("\n" + "=" * 70)

# Create directories
ensure_project_dirs()

# Create environment
print("\n[1/3] Creating environment...")
env = SmartGridEnv()
env = Monitor(env, str(config.RESULTS_LOGS_DIR))
env = DummyVecEnv([lambda: env])

# Quick training (config.DEMO_TIMESTEPS timesteps for demo)
print(f"\n[2/3] Training PPO agent ({config.DEMO_TIMESTEPS} timesteps for quick demo)...")
print("      (Full training uses 50,000 timesteps)")
print("-" * 70)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=config.LEARNING_RATE,
    n_steps=config.N_STEPS,
    batch_size=config.BATCH_SIZE,
    n_epochs=config.N_EPOCHS,
    gamma=config.GAMMA,
    verbose=0  # Less verbose for demo
)

# Train for quick demo
model.learn(total_timesteps=config.DEMO_TIMESTEPS, progress_bar=True)

# Save demo model
model.save(str(config.MODELS_DIR / config.DEMO_MODEL_FILENAME))
print(f"\n[OK] Training complete! Model saved.")

# Evaluation
print("\n[3/3] Evaluating policies...")
print("=" * 70)

# Create evaluation environments
rl_env = SmartGridEnv()
rl_env = DummyVecEnv([lambda: rl_env])

baseline_env = SmartGridEnv()
baseline_env = DummyVecEnv([lambda: baseline_env])

# Evaluate RL
print("\n--- RL Agent (Trained) ---")
# Need to handle vectorized env reset properly
import numpy as np
rl_metrics, _ = evaluate_policy(
    rl_env, model, n_episodes=5, policy_name="RL_PPO_Demo"
)

print(f"\nPerformance Metrics:")
print(f"  Mean Reward:        {rl_metrics['mean_reward']:8.2f} ± {rl_metrics['std_reward']:.2f}")
print(f"  Mean Peak Demand:   {rl_metrics['mean_peak_demand']:8.2f} ± {rl_metrics['std_peak_demand']:.2f} kW")
print(f"  Mean Total Cost:    ₹{rl_metrics['mean_total_cost'] * config.USD_TO_INR:7.2f} ± ₹{rl_metrics['std_total_cost'] * config.USD_TO_INR:.2f} INR")
print(f"  Mean Min Comfort:   {rl_metrics['mean_min_comfort']:8.3f} ± {rl_metrics['std_min_comfort']:.3f}")

# Evaluate Baseline
print("\n--- Baseline Controller (Rule-based) ---")
baseline_unwrapped = baseline_env.envs[0].unwrapped
baseline_controller = BaselineController(baseline_unwrapped)
baseline_metrics, _ = evaluate_policy(
    baseline_env, baseline_controller, n_episodes=5, policy_name="Baseline"
)

print(f"\nPerformance Metrics:")
print(f"  Mean Reward:        {baseline_metrics['mean_reward']:8.2f} ± {baseline_metrics['std_reward']:.2f}")
print(f"  Mean Peak Demand:   {baseline_metrics['mean_peak_demand']:8.2f} ± {baseline_metrics['std_peak_demand']:.2f} kW")
print(f"  Mean Total Cost:    ₹{baseline_metrics['mean_total_cost'] * config.USD_TO_INR:7.2f} ± ₹{baseline_metrics['std_total_cost'] * config.USD_TO_INR:.2f} INR")
print(f"  Mean Min Comfort:   {baseline_metrics['mean_min_comfort']:8.3f} ± {baseline_metrics['std_min_comfort']:.3f}")

# Comparison
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

reward_diff = rl_metrics['mean_reward'] - baseline_metrics['mean_reward']
peak_diff = baseline_metrics['mean_peak_demand'] - rl_metrics['mean_peak_demand']
cost_diff = baseline_metrics['mean_total_cost'] - rl_metrics['mean_total_cost']
comfort_diff = rl_metrics['mean_min_comfort'] - baseline_metrics['mean_min_comfort']

print(f"\nRL Agent vs Baseline:")
print(f"  Reward:      {reward_diff:+.2f} (RL is {'better' if reward_diff > 0 else 'worse'})")
print(f"  Peak Demand: {peak_diff:+.2f} kW (RL is {'better' if peak_diff > 0 else 'worse'})")
print(f"  Cost:        ₹{cost_diff * config.USD_TO_INR:+.2f} INR (RL is {'better' if cost_diff < 0 else 'worse'})")
print(f"  Comfort:     {comfort_diff:+.3f} (RL is {'better' if comfort_diff > 0 else 'worse'})")

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("=" * 70)
print("1. No dataset needed - environment generates data dynamically")
print("2. No accuracy metric - we measure performance (reward, peak, cost, comfort)")
print("3. Agent learns by trial and error, not from labeled examples")
print("4. Metrics show how well agent achieves objectives")
print("=" * 70)
print(f"\nNote: This was a quick demo ({config.DEMO_TIMESTEPS} timesteps).")
print("      Full training (50,000 timesteps) will show better performance!")
print("=" * 70)

