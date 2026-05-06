"""
SAC agent utilities with a continuous-to-discrete action wrapper
for compatibility with the existing SmartGridEnv.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from env.smart_grid_env import SmartGridEnv


class ContinuousActionWrapper(gym.ActionWrapper):
    """
    Wrap a discrete-action environment with a 1D continuous action in [-1, 1].

    Mapping:
      [-1.0, -0.6) -> 0
      [-0.6, -0.2) -> 1
      [-0.2,  0.2) -> 2
      [ 0.2,  0.6) -> 3
      [ 0.6,  1.0] -> 4
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        a = float(np.clip(action_arr[0], -1.0, 1.0))
        if a < -0.6:
            return 0
        if a < -0.2:
            return 1
        if a < 0.2:
            return 2
        if a < 0.6:
            return 3
        return 4


def make_sac_env(use_real_data=False, dataset_path=None, reward_weights=None, monitor_path=None):
    """Create SAC-compatible vectorized env."""
    base_env = SmartGridEnv(
        use_real_data=use_real_data,
        dataset_path=dataset_path,
        reward_weights=reward_weights,
    )
    wrapped = ContinuousActionWrapper(base_env)
    if monitor_path is not None:
        wrapped = Monitor(wrapped, monitor_path)
    return DummyVecEnv([lambda: wrapped])


def build_sac_model(env, seed=42):
    """Create SAC model with config-aligned hyperparameters."""
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        gamma=config.GAMMA,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        tensorboard_log=str(config.TENSORBOARD_DIR / "algorithm_comparison"),
        seed=seed,
    )
