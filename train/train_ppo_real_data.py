"""
Train PPO agent with real-world dataset
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


def train_ppo_real_data(seed=42):
    """Train PPO agent on Smart Grid environment with real-world data"""
    logger = get_logger("train.ppo_real_data")
    ensure_project_dirs()
    set_global_seed(seed)
    
    # Check if dataset exists
    dataset_path = config.REAL_DATASET_PATH
    if not os.path.exists(dataset_path):
        logger.error("Real-world dataset not found: %s", dataset_path)
        logger.error("Please run first: python data/download_dataset.py")
        logger.error("Or set USE_REAL_DATA = False in config.py to use synthetic data")
        return
    
    # Create environment with real data
    logger.info("Using real-world dataset for training...")
    env = SmartGridEnv(use_real_data=True, dataset_path=dataset_path)
    
    # Wrap with Monitor for logging
    log_dir = str(config.RESULTS_LOGS_DIR)
    env = Monitor(env, log_dir)
    
    # Wrap with DummyVecEnv for vectorization
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = SmartGridEnv(use_real_data=True, dataset_path=dataset_path)
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
        name_prefix="ppo_real_data_checkpoint"
    )
    
    best_model_callback = SaveBestModelCallback(
        eval_env=eval_env,
        best_model_path=str(config.MODELS_DIR / config.BEST_MODEL_REAL_DATA_FILENAME),
        eval_freq=config.BEST_MODEL_EVAL_FREQ,
        verbose=1
    )
    
    # Train the model
    logger.info("Starting PPO training with real-world data...")
    logger.info("Total timesteps: %s", config.TOTAL_TIMESTEPS)
    logger.info("Dataset: %s", dataset_path)
    
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, best_model_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = str(config.MODELS_DIR / config.FINAL_MODEL_REAL_DATA_FILENAME)
    model.save(final_model_path)
    logger.info("Training completed")
    logger.info("Final model saved to: %s", final_model_path)
    logger.info("Best model saved to: %s", str(config.MODELS_DIR / config.BEST_MODEL_REAL_DATA_FILENAME))
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on real data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    train_ppo_real_data(seed=args.seed)
