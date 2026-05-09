"""
Main script to run training and evaluation in sequence
"""

import os
import sys
from utils.logger import ensure_project_dirs, get_logger

def main():
    logger = get_logger("run_all")
    logger.info("MO-RL-PeakShaving: Training and Evaluation Pipeline")
    
    # Ensure directories exist
    ensure_project_dirs()
    
    # Step 1: Training
    logger.info("[1/2] Starting PPO training...")
    try:
        from train.train_ppo import train_ppo
        train_ppo()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.exception("Training failed: %s", e)
        sys.exit(1)
    
    # Step 2: Evaluation
    logger.info("[2/2] Starting evaluation...")
    try:
        from evaluation.evaluate import evaluate_all
        evaluate_all()
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        sys.exit(1)
    
    logger.info("Pipeline completed! Check results/ for outputs.")

if __name__ == "__main__":
    main()

