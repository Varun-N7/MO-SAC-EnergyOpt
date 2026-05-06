"""
Train PPO agent for Smart Grid peak load shaving
"""

import os
import sys
import argparse

# Add parent directory to path for imports FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import config

from env.smart_grid_env import SmartGridEnv
from train.callbacks import SaveBestModelCallback
from utils.logger import ensure_project_dirs, get_logger
from seeds import set_global_seed


def train_ppo(seed=42):
    """Train PPO agent on Smart Grid environment"""
    logger = get_logger("train.ppo")
    ensure_project_dirs()
    set_global_seed(seed)
    
    # Create environment
    env = SmartGridEnv()
    
    # Wrap with Monitor for logging
    log_dir = str(config.RESULTS_LOGS_DIR)
    env = Monitor(env, log_dir)
    
    # Wrap with DummyVecEnv for vectorization
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = SmartGridEnv()
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        verbose=1,
        tensorboard_log=str(config.TENSORBOARD_DIR),
        seed=seed,
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_SAVE_FREQ,
        save_path=str(config.MODELS_DIR),
        name_prefix="ppo_checkpoint"
    )
    
    best_model_callback = SaveBestModelCallback(
        eval_env=eval_env,
        best_model_path=str(config.MODELS_DIR / config.BEST_MODEL_FILENAME),
        eval_freq=config.BEST_MODEL_EVAL_FREQ,
        verbose=1
    )
    
    # Train the model
    logger.info("Starting PPO training...")
    logger.info("Total timesteps: %s", config.TOTAL_TIMESTEPS)
    
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, best_model_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = str(config.MODELS_DIR / config.FINAL_MODEL_FILENAME)
    model.save(final_model_path)
    logger.info("Training completed")
    logger.info("Final model saved to: %s", final_model_path)
    logger.info("Best model saved to: %s", str(config.MODELS_DIR / config.BEST_MODEL_FILENAME))
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    train_ppo(seed=args.seed)

