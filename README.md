
# ⚡ MO-SAC-EnergyOpt

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20RL-red?logo=pytorch)

> **Multi-Objective Soft Actor-Critic (SAC) for Smart Grid Energy Optimization and Peak Shaving**

A deep reinforcement learning system that trains a **Soft Actor-Critic agent** to optimize energy consumption in a smart grid environment — balancing cost, peak demand reduction, and battery longevity using a **multi-objective reward formulation** with adaptive weights.

----------

## 🧠 What This Project Does

Smart grids face a three-way conflict: minimize electricity cost, avoid demand peaks, and preserve consumer comfort. Classical controllers handle one objective at a time. This project trains an RL agent to handle **all three simultaneously**.

The agent learns to:

-   🔋 Charge/discharge the battery based on price signals and solar availability
-   ☀️ Integrate solar PV generation into dispatch decisions
-   📉 Shave peak demand loads during high-tariff hours (6–10 PM)
-   ⚖️ Balance competing objectives via adaptive reward weighting

**Three core objectives:**

-   ✅ Reduce peak electricity demand
-   ✅ Minimize total electricity cost
-   ✅ Maintain consumer comfort (hard constraint: comfort must stay above 0.60)

----------

## 🏗️ Project Structure

```
MO-SAC-EnergyOpt/
├── agents/               # SAC agent implementation
├── env/                  # Smart grid RL environment
│   ├── smart_grid_env.py     # Core Gym environment
│   ├── load_model.py         # Load demand model
│   ├── pv_model.py           # Solar PV generation model
│   ├── price_model.py        # TOU electricity price model
│   └── real_data_loader.py   # Real-world data pipeline
├── train/                # Training scripts (SAC + PPO)
│   ├── train_ppo.py          # Train with synthetic data
│   ├── train_ppo_real_data.py# Train with real-world data
│   └── callbacks.py          # Checkpoint + best model callbacks
├── evaluation/           # Baselines, metrics, plots
├── experiments/          # Ablation, algorithm comparison, Pareto sweep
├── data/                 # Real-world energy dataset (8,760 records)
├── dashboard/            # Interactive Flask dashboard
├── models/               # Saved model checkpoints
├── results/              # Logs and plots output
│   ├── logs/             # summary.csv, episode_trace_rl.csv
│   └── plots/            # grid_import.png, soc.png, etc.
├── config.py             # Global configuration
└── train.py              # Main training entry point

```

----------

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/MO-SAC-EnergyOpt.git
cd MO-SAC-EnergyOpt
pip install -r requirements.txt
pip install -r requirements_dashboard.txt  # for dashboard

```

----------

## 🚀 Quick Start

```bash
# Train (synthetic data)
python train.py

# Train (real-world data)
python data/download_dataset.py
python train/train_ppo_real_data.py

# Quick demo
python quick_demo.py

# Evaluate
python evaluation/evaluate.py
python evaluation/evaluate_real_data.py

# Launch dashboard
python app.py  # → http://localhost:5000

```

----------

## 🌐 Environment Details

### Observation Space

```python
observation = [hour, pv, load, price, soc, comfort]
# hour:    0–23        (time of day)
# pv:      0–15 kW    (solar generation)
# load:    0–10 kW    (household consumption)
# price:   4.28–5.63 ₹/kWh (TANGEDCO TOU rate)
# soc:     0–1        (battery state of charge)
# comfort: 0–1        (consumer comfort level)

```

<details> <summary>📋 Action Space (5 discrete actions)</summary>

Action

Description

`0`

Do nothing

`1`

Discharge battery (use stored energy)

`2`

Charge battery (store energy)

`3`

Shift flexible load (delay some consumption)

`4`

Comfort sacrifice (reduce HVAC slightly)

</details> <details> <summary>🔋 Battery Specs</summary>

Parameter

Value

Capacity

10 kWh

Max charge/discharge

5 kW

Round-trip efficiency

95%

</details> <details> <summary>💰 TOU Pricing (TANGEDCO — 2024–25)</summary>

Period

Hours

Price

Night off-peak

22:00–05:00

₹4.28/kWh (−5% night rebate)

Normal hours

05:00–06:00, 10:00–18:00

₹4.50/kWh

Peak hours

06:00–10:00 & 18:00–22:00

₹5.63/kWh (+25% surcharge)

> Based on TNERC Tariff Order effective July 2024.

</details>

----------

## 🏆 Reward Function

Multi-objective reward with **adaptive weights** that shift priority based on the current situation.

<details> <summary>📊 Reward Components</summary>

Component

Base Penalty

Adaptive Condition

Peak demand (> 6 kW)

-10.0

×2.0 during peak hours

Grid cost

`grid_import × ₹/kWh`

Higher price = higher penalty

Comfort violation (< 0.80)

-5.0

×2.0 when < 0.75, ×3.0 when < 0.70

Hard constraint (comfort < 0.60)

**-100.0 + episode end**

Safety termination

</details>

----------

## 📊 Dataset

**File:** `data/real_world/real_world_energy_data.csv` **Size:** 8,760 records · 365 days × 24 hours · Date range: 2023-01-01 to 2023-12-31

```python
# Synthetic mode (default)
env = SmartGridEnv(use_real_data=False)

# Real data mode
env = SmartGridEnv(use_real_data=True, dataset_path="data/real_world/real_world_energy_data.csv")

```

----------

## 🧪 Experiments

```bash
python experiments/ablation_study.py       # Ablation study
python experiments/compare_algorithms.py   # SAC vs PPO vs Baselines
python experiments/pareto_sweep.py         # Pareto front sweep

```

----------

## 📈 Key Results

The SAC agent outperforms rule-based baselines across all objectives:

-   ✅ Lower electricity cost
-   ✅ Reduced peak demand
-   ✅ Better comfort maintained over episodes

> Plots and metrics saved to `results/logs/` and `results/plots/` after evaluation.

----------

## ⚙️ Configuration

<details> <summary>🔧 Full Config Parameters (config.py)</summary>

Category

Parameter

Value

Battery

Capacity

10 kWh

Battery

Max charge/discharge

5 kW

Battery

Efficiency

95%

PV

Peak power

8.0 kW

Load

Morning peak

7 AM (1.8× multiplier)

Load

Evening peak

7 PM (2.2× multiplier)

Load

Flexible portion

15%

Training

Timesteps

50,000

Training

Learning rate

3e-4

Training

Batch size

64

Training

Gamma

0.99

</details>

----------

## ⚠️ Known Issue

When using `use_real_data=True`, `self.price_model` is set to `None`, causing `self.price_model.is_peak_hour()` to throw an error.

**Fix:**

```python
is_peak = self.hour in config.PEAK_HOURS

```

----------

## 🗂️ Development Timeline

Day

Milestone

Day 1

Project setup, config, RL environment, PV/load/price models

Day 2

SAC agent implementation, training pipeline

Day 3

Evaluation framework, baseline controllers, unit tests

Day 4

Real-world data integration, experiments, ablation study

Day 5

Dashboard, frontend UI, demo scripts, documentation

----------

## 🛠️ Tech Stack

-   **RL Framework** — Stable Baselines3, custom SAC
-   **Environment** — OpenAI Gym / Gymnasium
-   **Deep Learning** — PyTorch
-   **Data** — Pandas, NumPy
-   **Visualization** — Matplotlib, Flask dashboard
-   **Testing** — Pytest

----------

## 📚 References

-   Sutton & Barto — _Reinforcement Learning: An Introduction_
-   Schulman et al. — _Proximal Policy Optimization Algorithms_
-   IEEE papers on demand response and peak shaving

----------

## 📄 Documentation

File

Description

[`EVALUATION_METRICS.md`](https://claude.ai/chat/EVALUATION_METRICS.md)

Metrics explained

[`UNDERSTANDING_RL_METRICS.md`](https://claude.ai/chat/UNDERSTANDING_RL_METRICS.md)

RL concepts

[`REAL_DATA_INTEGRATION.md`](https://claude.ai/chat/REAL_DATA_INTEGRATION.md)

Data pipeline details

[`COMPLETE_PROJECT_EXPLANATION.md`](https://claude.ai/chat/COMPLETE_PROJECT_EXPLANATION.md)

Full project walkthrough

