"""
Configuration file for MO-RL-PeakShaving project
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REAL_DATA_DIR = DATA_DIR / "real_world"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
TENSORBOARD_DIR = RESULTS_LOGS_DIR / "tensorboard"

# File names and paths
REAL_DATASET_FILENAME = "real_world_energy_data.csv"
REAL_DATASET_PATH = str(REAL_DATA_DIR / REAL_DATASET_FILENAME)
BEST_MODEL_FILENAME = "ppo_best_model.zip"
FINAL_MODEL_FILENAME = "ppo_final_model.zip"
BEST_MODEL_REAL_DATA_FILENAME = "ppo_best_model_real_data.zip"
FINAL_MODEL_REAL_DATA_FILENAME = "ppo_final_model_real_data.zip"
DEMO_MODEL_FILENAME = "ppo_demo_model.zip"
SUMMARY_FILENAME = "summary.csv"
SUMMARY_REAL_DATA_FILENAME = "summary_real_data.csv"
EPISODE_TRACE_FILENAME = "episode_trace_rl.csv"
EPISODE_TRACE_REAL_DATA_FILENAME = "episode_trace_rl_real_data.csv"
MONITOR_FILENAME = "monitor.csv"

# Plot output file names
PLOT_GRID_IMPORT_VS_PV_FILENAME = "grid_import_vs_pv.png"
PLOT_BATTERY_SOC_FILENAME = "battery_soc.png"
PLOT_BATTERY_HEALTH_FILENAME = "battery_health.png"
PLOT_COMFORT_SCORE_FILENAME = "comfort_score.png"
PLOT_OVERVIEW_FILENAME = "overview.png"

# Flask app settings
FLASK_DEBUG = True
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Callback and evaluation defaults
CHECKPOINT_SAVE_FREQ = 10000
BEST_MODEL_EVAL_FREQ = 5000
BEST_MODEL_N_EVAL_EPISODES = 5

# Comfort thresholds used in reward shaping
COMFORT_PENALTY_THRESHOLD = 0.80
ADAPTIVE_COMFORT_THRESHOLD = 0.75
SEVERE_COMFORT_THRESHOLD = 0.70

# HVAC reduction ratio used by action 4
HVAC_REDUCTION_FRACTION = 0.2

# Demo settings
DEMO_TIMESTEPS = 5000

# Battery parameters
BATTERY_CAPACITY_KWH = 10.0  # kWh
BATTERY_MAX_CHARGE_POWER = 5.0  # kW
BATTERY_MAX_DISCHARGE_POWER = 5.0  # kW
BATTERY_CHARGE_EFFICIENCY = 0.95
BATTERY_DISCHARGE_EFFICIENCY = 0.95
INITIAL_SOC = 0.5  # Initial state of charge (0-1)
BATTERY_DEGRADATION_RATE = 0.0001  # Capacity fade rate per cycle
BATTERY_MIN_HEALTH = 0.5  # Lower bound on battery health (0-1)
INITIAL_BATTERY_HEALTH = 1.0
LOW_SOC_THRESHOLD = 0.20  # Penalize deep discharge below 20% SOC

# PV generation parameters
PV_PEAK_POWER = 8.0  # kW (peak generation at midday)
PV_NOISE_STD = 0.5  # Standard deviation for Gaussian noise

# Load model parameters
BASE_LOAD_MIN = 1.0  # kW (minimum base load)
BASE_LOAD_MAX = 3.0  # kW (maximum base load)
MORNING_PEAK_HOUR = 7  # Hour of morning peak (0-23)
EVENING_PEAK_HOUR = 19  # Hour of evening peak (0-23)
MORNING_PEAK_MULTIPLIER = 1.8
EVENING_PEAK_MULTIPLIER = 2.2
FLEXIBLE_LOAD_FRACTION = 0.15  # Fraction of load that can be shifted

# Price model parameters (TOU - Time of Use)
PRICE_OFF_PEAK = 0.10  # $/kWh (night hours)
PRICE_MID_PEAK = 0.15  # $/kWh (day hours)
PRICE_ON_PEAK = 0.25  # $/kWh (evening hours 18-22)
PEAK_HOURS = [18, 19, 20, 21]  # Peak pricing hours

# Comfort parameters
INITIAL_COMFORT = 1.0  # Initial comfort score (0-1)
COMFORT_DECAY_SHIFT = 0.05  # Comfort decrease per load shift action
COMFORT_DECAY_HVAC = 0.10  # Comfort decrease per HVAC reduction action
COMFORT_MIN_THRESHOLD = 0.60  # Hard constraint threshold
COMFORT_RECOVERY_RATE = 0.02  # Comfort recovery per hour (when no actions taken)

# Reward parameters
PEAK_DEMAND_THRESHOLD = 6.0  # kW (threshold for peak demand penalty)
PEAK_PENALTY_BASE = -10.0  # Base penalty for exceeding peak threshold
PEAK_PENALTY_MULTIPLIER = 2.0  # Multiplier during peak hours
COST_WEIGHT = 1.0  # Weight for grid cost component
COMFORT_VIOLATION_PENALTY = -5.0  # Penalty when comfort below minimum
COMFORT_VIOLATION_MULTIPLIER = 3.0  # Multiplier when comfort is low
HARD_CONSTRAINT_PENALTY = -100.0  # Large penalty for comfort < 0.60
BATTERY_WEIGHT = 1.0  # lambda_battery: optional 4th objective weight
SOC_SWING_PENALTY_WEIGHT = 2.0  # Penalize large SOC swings
LOW_SOC_PENALTY_WEIGHT = 5.0  # Penalize operation in deep discharge region
DEGRADATION_COST_WEIGHT = 100.0  # Penalize battery health loss

# Default objective weights
REWARD_WEIGHT_PEAK = 1.0
REWARD_WEIGHT_COST = 1.0
REWARD_WEIGHT_COMFORT = 1.0
REWARD_WEIGHT_BATTERY = BATTERY_WEIGHT

# Adaptive weight parameters
ADAPTIVE_PEAK_WEIGHT_MULTIPLIER = 1.5  # Increase peak weight during peak hours
ADAPTIVE_COMFORT_WEIGHT_MULTIPLIER = 2.0  # Increase comfort weight when comfort is low

# Episode parameters
EPISODE_LENGTH = 24  # Hours per episode

# Training parameters
TOTAL_TIMESTEPS = 50000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5

# Evaluation parameters
N_EVAL_EPISODES = 10

# Pareto sweep defaults
PARETO_WEIGHT_GRID = [0.5, 1.0, 1.5]
PARETO_TIMESTEPS = TOTAL_TIMESTEPS
PARETO_EVAL_EPISODES = N_EVAL_EPISODES
PARETO_DATA_FILENAME = "pareto_data.json"
PARETO_FRONT_PLOT_FILENAME = "pareto_front.png"
PARETO_2D_PLOT_FILENAME = "pareto_2d.png"

# Real-world dataset parameters
USE_REAL_DATA = False  # Set to True to use real-world dataset

# Display currency: cost shown in INR (internal computation remains in USD)
USD_TO_INR = 83.0  # Conversion rate for display (adjust as needed)
CURRENCY = "INR"
