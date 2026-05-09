"""
Flask web application for MO-RL-PeakShaving project presentation
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, send_file, request
from flask_cors import CORS

from env.smart_grid_env import SmartGridEnv
from evaluation.baseline_controller import BaselineController
import config
from utils.logger import ensure_project_dirs, get_logger

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)
logger = get_logger("app")

# Configuration
RESULTS_DIR = str(config.RESULTS_DIR)
LOGS_DIR = str(config.RESULTS_LOGS_DIR)
PLOTS_DIR = str(config.RESULTS_PLOTS_DIR)
MODELS_DIR = str(config.MODELS_DIR)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/summary')
def get_summary():
    """Get evaluation summary data"""
    try:
        summary_path = os.path.join(LOGS_DIR, "summary.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            return jsonify(df.to_dict('records'))
        else:
            return jsonify({"error": "Summary file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/episode_trace')
def get_episode_trace():
    """Get RL episode trace data"""
    try:
        trace_path = os.path.join(LOGS_DIR, "episode_trace_rl.csv")
        if os.path.exists(trace_path):
            df = pd.read_csv(trace_path)
            # Convert to list of dictionaries
            data = df.to_dict('records')
            return jsonify(data)
        else:
            return jsonify({"error": "Episode trace file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training_stats')
def get_training_stats():
    """Get training statistics from monitor logs"""
    try:
        monitor_path = os.path.join(LOGS_DIR, "monitor.csv")
        if os.path.exists(monitor_path):
            # Read monitor.csv - skip comment lines and header
            with open(monitor_path, 'r') as f:
                lines = f.readlines()
                # Find the actual header line (starts with 'r,l,t')
                header_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('r,l,t'):
                        header_idx = i
                        break
                
                # Read CSV starting from header
                df = pd.read_csv(monitor_path, skiprows=header_idx)
            
            # Calculate statistics
            stats = {
                "total_episodes": len(df),
                "mean_reward": float(df['r'].mean()) if 'r' in df.columns and len(df) > 0 else 0,
                "std_reward": float(df['r'].std()) if 'r' in df.columns and len(df) > 0 else 0,
                "mean_length": float(df['l'].mean()) if 'l' in df.columns and len(df) > 0 else 0,
                "rewards": df['r'].tolist() if 'r' in df.columns else [],
                "episode_lengths": df['l'].tolist() if 'l' in df.columns else []
            }
            return jsonify(stats)
        else:
            return jsonify({"error": "Monitor file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/project_info')
def get_project_info():
    """Get project information"""
    info = {
        "name": "MO-RL-PeakShaving",
        "full_name": "Adaptive Multi-Objective Reinforcement Learning for Peak Load Shaving",
        "description": "RL system for peak load shaving in solar-integrated smart grids with consumer comfort constraints",
        "features": [
            "24-hour episodic simulation with stochastic PV generation",
            "Residential load patterns with morning and evening peaks",
            "Time-of-Use (TOU) electricity pricing",
            "Battery storage with charge/discharge dynamics",
            "Consumer comfort modeling",
            "Multi-objective optimization with adaptive weights"
        ],
        "metrics": [
            "Mean Reward (higher is better)",
            "Peak Demand in kW (lower is better)",
            "Total Cost in INR (lower is better)",
            "Minimum Comfort (higher is better, must be > 0.60)"
        ],
        "currency": config.CURRENCY,
        "usd_to_inr": config.USD_TO_INR,
        "actions": [
            "Do Nothing",
            "Discharge Battery",
            "Charge Battery",
            "Shift Flexible Load",
            "Comfort Sacrifice (HVAC Reduction)"
        ]
    }
    return jsonify(info)


@app.route('/api/plots/<plot_name>')
def get_plot(plot_name):
    """Serve plot images"""
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    else:
        return jsonify({"error": "Plot not found"}), 404


@app.route('/api/models/status')
def get_models_status():
    """Check if models exist"""
    models = {
        "best_model": os.path.exists(os.path.join(MODELS_DIR, "ppo_best_model.zip")),
        "final_model": os.path.exists(os.path.join(MODELS_DIR, "ppo_final_model.zip"))
    }
    return jsonify(models)


@app.route('/api/algorithm_comparison')
def get_algorithm_comparison():
    """Get PPO vs SAC comparison results."""
    try:
        path = os.path.join(RESULTS_DIR, "algorithm_comparison.csv")
        if not os.path.exists(path):
            return jsonify({
                "available": False,
                "error": "algorithm_comparison.csv not found",
                "hint": "Run: python experiments/compare_algorithms.py --seed 42"
            })
        df = pd.read_csv(path)
        return jsonify({"available": True, "rows": df.to_dict("records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ablation_summary')
def get_ablation_summary():
    """Get reward-weight ablation results."""
    try:
        path = os.path.join(RESULTS_DIR, "ablation_results.csv")
        if not os.path.exists(path):
            return jsonify({
                "available": False,
                "error": "ablation_results.csv not found",
                "hint": "Run: python experiments/ablation_study.py --seed 42"
            })
        df = pd.read_csv(path)
        return jsonify({"available": True, "rows": df.to_dict("records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pareto_data')
def get_pareto_data():
    """Get Pareto sweep JSON data."""
    try:
        path = os.path.join(RESULTS_DIR, "pareto_data.json")
        if not os.path.exists(path):
            return jsonify({
                "available": False,
                "error": "pareto_data.json not found",
                "hint": "Run: python experiments/pareto_sweep.py --seed 42"
            })
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["available"] = True
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/final_summary')
def get_final_summary():
    """Get consolidated final summary table."""
    try:
        path = os.path.join(RESULTS_DIR, "final_summary.csv")
        if not os.path.exists(path):
            return jsonify({
                "available": False,
                "error": "final_summary.csv not found",
                "hint": "Run: python results/generate_report.py"
            })
        df = pd.read_csv(path)
        # Replace NaN with None so JSON is standards-compliant for browsers.
        rows = df.astype(object).where(pd.notnull(df), None).to_dict("records")
        return jsonify({"available": True, "rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/dataset_info')
def get_dataset_info():
    """
    Return basic information about the configured real-world dataset.
    This helps prove to the panel that a real time-series dataset is used.
    """
    dataset_path = getattr(config, "REAL_DATASET_PATH", config.REAL_DATASET_PATH)

    info = {
        "path": dataset_path,
        "exists": False,
        "num_rows": 0,
        "date_min": None,
        "date_max": None,
    }

    if not os.path.exists(dataset_path):
        return jsonify(info)

    try:
        # Read a small sample to get metadata without loading everything heavy
        df = pd.read_csv(dataset_path)
        info["exists"] = True
        info["num_rows"] = int(len(df))

        if "datetime" in df.columns:
            try:
                dt = pd.to_datetime(df["datetime"])
                info["date_min"] = str(dt.min())
                info["date_max"] = str(dt.max())
            except Exception:
                pass

        # Column checks
        required_cols = ["pv_generation_kw", "load_kw", "price_per_kwh", "hour"]
        missing = [c for c in required_cols if c not in df.columns]
        info["missing_columns"] = missing
    except Exception as e:
        info["error"] = str(e)

    return jsonify(info)


@app.route('/api/dataset_preview')
def get_dataset_preview():
    """
    Return an aggregated 24-hour profile from the real dataset
    (average PV, load, and price per hour). This is used to plot
    a "Real Dataset Insights" chart on the frontend.
    """
    dataset_path = getattr(config, "REAL_DATASET_PATH", config.REAL_DATASET_PATH)

    if not os.path.exists(dataset_path):
        return jsonify(
            {
                "exists": False,
                "error": f"Dataset not found at {dataset_path}",
            }
        )

    try:
        df = pd.read_csv(dataset_path)

        required_cols = ["pv_generation_kw", "load_kw", "price_per_kwh"]
        for col in required_cols:
            if col not in df.columns:
                return jsonify(
                    {
                        "exists": True,
                        "error": f"Missing required column: {col}",
                    }
                )

        # Ensure hour column exists
        if "hour" not in df.columns:
            if "datetime" in df.columns:
                dt = pd.to_datetime(df["datetime"])
                df["hour"] = dt.dt.hour
            else:
                # Fallback: assume already hourly and create a repeating 0-23 cycle
                df["hour"] = df.index % 24

        grouped = (
            df.groupby("hour")[["pv_generation_kw", "load_kw", "price_per_kwh"]]
            .mean()
            .reset_index()
            .sort_values("hour")
        )

        return jsonify(
            {
                "exists": True,
                "hours": grouped["hour"].astype(int).tolist(),
                "pv": grouped["pv_generation_kw"].astype(float).tolist(),
                "load": grouped["load_kw"].astype(float).tolist(),
                "price": grouped["price_per_kwh"].astype(float).tolist(),
            }
        )
    except Exception as e:
        return jsonify({"exists": True, "error": str(e)})


@app.route('/api/run_simulation')
def run_simulation():
    """
    Run a single 24-hour simulation and return timestep data as JSON.
    This makes the project feel "live": the panel can trigger a fresh run.
    """
    policy = request.args.get("policy", "rl")  # 'rl' or 'baseline'
    mode = request.args.get("mode", "synthetic")  # 'synthetic' or 'real'

    try:
        use_real_data = mode == "real"

        # Use config-driven dataset path for real-data mode
        env_kwargs = {}
        if use_real_data:
            dataset_path = getattr(config, "REAL_DATASET_PATH", config.REAL_DATASET_PATH)
            env_kwargs.update(
                use_real_data=True,
                dataset_path=dataset_path,
            )

        env = SmartGridEnv(**env_kwargs)

        # Reset environment (Gymnasium-style API: obs, info)
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _info = reset_result
        else:
            obs = reset_result

        # Choose policy
        controller = None
        model = None

        if policy == "baseline":
            controller = BaselineController(env)
        else:
            # Default: RL PPO agent
            if PPO is None:
                return jsonify({"error": "PPO is not installed in the environment"}), 500

            best_path = os.path.join(MODELS_DIR, config.BEST_MODEL_FILENAME)
            final_path = os.path.join(MODELS_DIR, config.FINAL_MODEL_FILENAME)

            model_path = best_path if os.path.exists(best_path) else final_path
            if not os.path.exists(model_path):
                return jsonify(
                    {
                        "error": "No trained PPO model found. "
                        "Please run training first (python train/train_ppo.py)."
                    }
                ), 400

            model = PPO.load(model_path)

        timesteps = []

        for _ in range(24):
            # Record state BEFORE action
            state = {
                "hour": int(env.hour),
                "pv": float(env.pv),
                "base_load": float(getattr(env, "base_load", 0.0)),
                "grid_import": float(getattr(env, "grid_import", 0.0)),
                "soc": float(getattr(env, "soc", 0.0)),
                "battery_health": float(getattr(env, "battery_health", 1.0)),
                "cycle_count": float(getattr(env, "cycle_count", 0.0)),
                "comfort": float(getattr(env, "comfort", 0.0)),
                "price": float(getattr(env, "price", 0.0)),
            }

            # Select action
            if controller is not None:
                action = controller.predict(obs)
            else:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = int(action[0]) if action.shape else int(action)

            state["action"] = int(action)

            # Step environment (Gymnasium 0.29-style: obs, reward, terminated, truncated, info)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result  # fallback for older API

            state["reward"] = float(reward)
            timesteps.append(state)

            if done:
                break

        # Aggregate metrics for this live run (cost converted to INR for display)
        if timesteps:
            peak_demand = max(t["grid_import"] for t in timesteps)
            total_cost_usd = sum(t["grid_import"] * t["price"] for t in timesteps)
            total_cost = total_cost_usd * config.USD_TO_INR  # Display in INR
            min_comfort = min(t["comfort"] for t in timesteps)
            total_reward = sum(t["reward"] for t in timesteps)
        else:
            peak_demand = total_cost = min_comfort = total_reward = 0.0

        return jsonify(
            {
                "policy": policy,
                "mode": mode,
                "timesteps": timesteps,
                "metrics": {
                    "peak_demand_kw": peak_demand,
                    "total_cost": total_cost,
                    "min_comfort": min_comfort,
                    "total_reward": total_reward,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Create necessary directories
    ensure_project_dirs()
    
    print("=" * 70)
    print("MO-RL-PeakShaving Web Interface")
    print("=" * 70)
    print("\nStarting Flask server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    logger.info("Starting Flask server at http://localhost:%s", config.FLASK_PORT)
    app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)

