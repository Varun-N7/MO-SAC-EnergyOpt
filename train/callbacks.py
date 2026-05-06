"""
Callbacks for PPO training
Includes callback to save best model based on evaluation
"""

import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import config
from utils.logger import get_logger


class SaveBestModelCallback(BaseCallback):
    """
    Callback to save the best model based on evaluation performance
    Evaluates the model periodically and saves if it's the best so far
    """
    
    def __init__(self, eval_env, best_model_path, eval_freq=config.BEST_MODEL_EVAL_FREQ, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_model_path = best_model_path
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.n_eval_episodes = config.BEST_MODEL_N_EVAL_EPISODES
        self._logger = get_logger("train.callbacks")
        
    def _on_step(self) -> bool:
        """Called at each step"""
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the current model
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=False
            )
            
            if self.verbose > 0:
                self._logger.info(
                    "Eval at step %s: mean_reward=%.2f +/- %.2f",
                    self.n_calls,
                    mean_reward,
                    std_reward,
                )
            
            # Save if this is the best model so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    self._logger.info("New best model! Saving to %s", self.best_model_path)
                self.model.save(self.best_model_path)
        
        return True

