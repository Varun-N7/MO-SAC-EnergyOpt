# MO-RL-PeakShaving — Full Project Explanation & Post-Review Roadmap

Use this document to **explain the project in your review** and to plan **what you will add in the coming days** after the review.

---

# PART 1: WHAT THE PROJECT IS AND WHAT IS MADE (FOR YOUR EXPLANATION)

## 1.1 Project in One Sentence

**MO-RL-PeakShaving** is an **adaptive multi-objective reinforcement learning (RL) system** that learns how to **reduce peak electricity demand** (peak load shaving) in a **solar-integrated smart grid** while **minimizing grid cost** and **keeping consumer comfort above a safe level**, using **Proximal Policy Optimization (PPO)** and a **Gymnasium-based 24-hour simulation environment**.

---

## 1.2 Problem Being Solved

- **Peak demand** causes high costs and stress on the grid.
- We have **solar (PV)**, **battery**, and **flexible load** (e.g. shiftable appliances, HVAC) that we can control.
- We must **balance**:
  - **Lower peak demand** (shave peaks)
  - **Lower total energy cost** (use cheap hours, avoid expensive peak hours)
  - **Consumer comfort** (do not shift load or cut HVAC so much that comfort drops below 0.60 — hard constraint)

The project builds an **RL agent** that learns this balance from experience in simulation, and compares it to a **rule-based baseline**.

---

## 1.3 What Is Made — Component-by-Component

### A. Configuration (`config.py`)

**What it is:** Single place for all parameters.

**What it does:**
- **Battery:** Capacity (10 kWh), max charge/discharge power (5 kW), efficiencies (0.95), initial SOC (0.5).
- **PV:** Peak power (8 kW), noise (Gaussian std 0.5).
- **Load:** Base load range (1–3 kW), morning peak (hour 7, multiplier 1.8), evening peak (hour 19, multiplier 2.2), flexible fraction (15%).
- **Price (TOU):** Off-peak $0.10, mid-peak $0.15, on-peak $0.25/kWh; peak hours 18–21.
- **Comfort:** Initial 1.0; decay per load-shift (0.05) and per HVAC reduction (0.10); minimum allowed 0.60; recovery rate 0.02 when doing nothing.
- **Reward:** Peak threshold 6 kW, penalties and weights for peak, cost, comfort; hard-constraint penalty -100.
- **Training:** 50k timesteps, learning rate 3e-4, n_steps 2048, batch 64, epochs 10, gamma 0.99, GAE lambda 0.95, etc.
- **Evaluation:** 10 episodes.
- **Real data:** Flag and path for real-world dataset CSV.

You can say: *“All physics, economics, and training hyperparameters are in one config file so we can tune the system without touching the core code.”*

---

### B. Smart Grid Environment (`env/smart_grid_env.py`)

**What it is:** A **Gymnasium environment** that simulates one **24-hour day** (one episode). The agent sees a state, picks an action, and gets a reward every hour.

**What it does:**

- **Observation (state) — 6 numbers:**
  - `hour` (0–23)
  - `pv` — PV generation (kW)
  - `load` — current load (kW)
  - `price` — electricity price ($/kWh)
  - `soc` — battery state of charge (0–1)
  - `comfort` — consumer comfort score (0–1)

- **Actions — 5 discrete choices:**
  1. **Do nothing**
  2. **Discharge battery** — reduces grid import (up to 5 kW, limited by SOC and efficiency)
  3. **Charge battery** — uses excess PV or grid (up to 5 kW, limited by capacity)
  4. **Shift flexible load** — reduces current load by flexible fraction; comfort drops by 0.05
  5. **Comfort sacrifice (HVAC reduction)** — reduces load by 20%; comfort drops by 0.10

- **Dynamics:**
  - Each step = 1 hour. PV, load, and price come either from **synthetic models** (PV, load, price) or from a **real-world CSV** (RealDataLoader).
  - Battery: charge/discharge updates SOC; efficiency 0.95 both ways.
  - Grid import = max(0, load - PV) after battery action (import from grid).
  - Comfort recovers by 0.02 per hour when action is “do nothing.” Load-shift effect lasts one hour.

- **Hard constraint:** If **comfort < 0.60**, the episode **terminates** and the agent gets a **large penalty (-100)**. So the agent must learn not to push comfort below 0.60.

- **Reward (multi-objective, adaptive):**
  - **Peak penalty:** If grid import > 6 kW, apply base penalty; **stronger during peak hours (18–21)**.
  - **Cost:** Negative reward proportional to (grid_import × price).
  - **Comfort penalty:** If comfort < 0.80, apply penalty; **stronger when comfort is low** (< 0.75, < 0.70).
  - **Hard constraint:** -100 if comfort < 0.60.

You can say: *“The environment is a 24-hour simulator. The agent sees hour, PV, load, price, battery SOC, and comfort, and chooses among five actions. Reward combines peak shaving, cost, and comfort, with a hard stop if comfort goes below 0.60.”*

---

### C. PV Model (`env/pv_model.py`)

**What it is:** Model of **solar generation** for each hour.

**What it does:**
- **Synthetic:** Sine-like curve from 6–18 (zero outside). Peak at midday (e.g. 12). Scaled to config peak power (8 kW). Adds **Gaussian noise** for randomness.
- Provides `get_generation(hour)` and `get_deterministic_generation(hour)` for tests.

You can say: *“PV is modeled with a realistic daily curve and random noise so the agent sees different days.”*

---

### D. Load Model (`env/load_model.py`)

**What it is:** Model of **residential electricity demand**.

**What it does:**
- **Synthetic:** Base load (random between 1–3 kW) plus **morning peak** (around hour 7) and **evening peak** (around hour 19) using Gaussian-like multipliers.
- If `shifted=True`, load is reduced by the flexible fraction (15%).
- `get_flexible_load_amount(hour)` returns how much load can be shifted at that hour.

You can say: *“Load has morning and evening peaks like real homes; a fraction is flexible and can be shifted at a comfort cost.”*

---

### E. Price Model (`env/price_model.py`)

**What it is:** **Time-of-Use (TOU)** electricity price.

**What it does:**
- Returns price per hour: **off-peak** (e.g. night) $0.10, **mid-peak** (day) $0.15, **on-peak** (18–21) $0.25/kWh.
- `is_peak_hour(hour)` tells whether the hour is peak (used for adaptive reward).

You can say: *“Prices are TOU: higher in the evening peak so the agent learns to shift demand and use the battery then.”*

---

### F. Real Data Loader (`env/real_data_loader.py`)

**What it is:** Loads a **CSV** with real-world hourly data and feeds it into the environment instead of synthetic models.

**What it does:**
- Expects columns: `pv_generation_kw`, `load_kw`, `price_per_kwh`, and `datetime` (or `hour`).
- Can **reset** to a random 24-hour window (or a given date). **Advances hour by hour** and returns current PV, load, price, hour.
- `is_episode_complete()` after 24 steps.

You can say: *“We can switch from synthetic to real data: the loader reads a CSV and runs 24-hour episodes from it.”*

---

### G. PPO Training (`train/train_ppo.py`)

**What it is:** Trains the **RL agent** with **PPO** (Stable-Baselines3).

**What it does:**
- Creates `SmartGridEnv` (synthetic), wraps with **Monitor** (for reward/length logs) and **DummyVecEnv**.
- Builds **PPO** with MlpPolicy and config hyperparameters (learning rate, n_steps, batch_size, etc.); logs to TensorBoard.
- **CheckpointCallback:** Saves model every 10k steps to `models/`.
- **SaveBestModelCallback:** Every 5k steps, runs 5 evaluation episodes; if mean reward is best so far, saves to `models/ppo_best_model.zip`.
- At the end saves **final** model to `models/ppo_final_model.zip`.

You can say: *“We train with PPO, save checkpoints and the best model by evaluation reward, and keep the final model for comparison.”*

---

### H. Training Callbacks (`train/callbacks.py`)

**What it is:** Custom callback for **saving the best model** during training.

**What it does:**
- Every `eval_freq` steps (5000), evaluates the current policy on a separate env for 5 episodes.
- If mean reward is **higher than any previous**, saves the model to `ppo_best_model.zip`.

You can say: *“We don’t only keep the last model; we keep the one that performed best during training.”*

---

### I. Baseline Controller (`evaluation/baseline_controller.py`)

**What it is:** **Rule-based heuristic** policy (no learning).

**What it does:**
- **Rule 1:** If hour 18–22 and SOC > 0.5 → **discharge**.
- **Rule 2:** If hour 10–16, PV > 4 kW, SOC < 0.9 → **charge**.
- **Rule 3:** If PV > 5 kW and comfort > 0.7 → **shift load**.
- **Else:** **do nothing.**

Same interface as the RL model (`predict(observation)`), so we can compare them in the same evaluation pipeline.

You can say: *“The baseline is a simple rule set: charge when sunny and battery not full, discharge at peak, shift load when PV is high and comfort is good. We compare RL against this.”*

---

### J. Evaluation (`evaluation/evaluate.py`)

**What it is:** Runs **RL policy** and **baseline** on the same environment and compares them.

**What it does:**
- Loads PPO from `ppo_best_model.zip` (or `ppo_final_model.zip`).
- For each policy: runs **10 episodes**, collects **mean ± std** of:
  - **Episode reward**
  - **Peak demand (kW)** — max grid import in the episode
  - **Total cost ($)** — sum of (grid_import × price) per hour
  - **Minimum comfort** in the episode
- Saves **summary** to `results/logs/summary.csv` (one row per policy).
- Saves **first-episode trace** (hour-by-hour) for RL to `results/logs/episode_trace_rl.csv`.
- Calls `plot_results()` to generate plots.

You can say: *“Evaluation runs both policies for 10 episodes and records reward, peak demand, cost, and minimum comfort, then saves a summary and an example trace.”*

---

### K. Plot Results (`evaluation/plot_results.py`)

**What it is:** Generates **matplotlib figures** from one episode trace.

**What it does:**
- **Grid import vs PV** over 24 hours.
- **Battery SOC (%)** over 24 hours.
- **Comfort score** over 24 hours (with 0.60 and 0.80 lines).
- **Overview:** 2×2 with grid/PV, SOC, comfort, and **action taken per hour** (do nothing, discharge, charge, shift, comfort sacrifice).
- Saves all to `results/plots/` (e.g. `grid_import_vs_pv.png`, `battery_soc.png`, `comfort_score.png`, `overview.png`).

You can say: *“We plot one episode: grid vs PV, SOC, comfort, and which actions the agent chose each hour.”*

---

### L. Run-All Pipeline (`run_all.py`)

**What it is:** Single entry point to **train then evaluate**.

**What it does:**
1. Creates `models/`, `results/logs/`, `results/plots/`.
2. Runs `train_ppo()` (full PPO training).
3. Runs `evaluate_all()` (RL + baseline, summary, trace, plots).
4. Exits with error if either step fails.

You can say: *“One script runs the full pipeline: train PPO, then evaluate and plot.”*

---

### M. Real-Data Training & Evaluation (optional path)

- **`data/download_real_dataset.py`** (or `download_dataset.py`): Builds a **realistic CSV** (e.g. 8760 rows, hourly) from published patterns (seasonal PV, load, TOU price) and saves to `data/real_world/real_world_energy_data.csv`. Used when you don’t have an external real dataset.
- **`train/train_ppo_real_data.py`**: Same as `train_ppo.py` but creates env with `use_real_data=True` and the dataset path.
- **`evaluation/evaluate_real_data.py`**: Same as `evaluate.py` but with real-data env.

You can say: *“We can generate or use a real-world-style dataset and train and evaluate on it with the same code.”*

---

### N. Flask Web App (`app.py` + `templates/index.html` + static)

**What it is:** A **web dashboard** to present the project (e.g. in a review or demo).

**What it does:**

- **Routes (APIs):**
  - **`/`** — Serves the dashboard page.
  - **`/api/summary`** — Returns `summary.csv` (RL vs baseline metrics).
  - **`/api/episode_trace`** — Returns `episode_trace_rl.csv` (one episode, hour by hour).
  - **`/api/training_stats`** — Reads `monitor.csv` and returns episode rewards/lengths.
  - **`/api/project_info`** — Project name, description, features, metrics, actions (for UI text).
  - **`/api/plots/<plot_name>`** — Serves PNGs from `results/plots/`.
  - **`/api/models/status`** — Whether best/final PPO models exist.
  - **`/api/dataset_info`** — Whether real dataset exists, path, row count, date range.
  - **`/api/dataset_preview`** — Aggregated 24-hour profile (avg PV, load, price per hour) from the real dataset for charts.
  - **`/api/run_simulation`** — **Runs a live 24-step simulation:** policy = `rl` or `baseline`, mode = `synthetic` or `real`; returns timestep-by-timestep data and metrics (peak demand, total cost, min comfort, total reward).

- **Front-end (index.html):**
  - **Sidebar:** Controller (RL vs Baseline), Data source (synthetic vs real), **Run Simulation** button, Export results, system parameters (battery, PV, comfort threshold, peak hours), dataset status.
  - **Metrics cards:** Peak demand (kW), total cost (USD), comfort score, reward (updated after “Run Simulation” or from saved results).
  - **Charts:** e.g. Grid import vs PV, SOC, comfort, actions over 24 hours (from trace or live run).
  - **Dataset insights:** If real data exists, shows 24h profile from `/api/dataset_preview`.

You can say: *“We have a Flask web app that shows evaluation summary, episode traces, training stats, and plots. The panel can run a live simulation with RL or baseline on synthetic or real data and see the results immediately.”*

---

## 1.4 End-to-End Flow (What the System Does)

1. **Configure** — Set battery, PV, load, price, comfort, reward, and training parameters in `config.py`.
2. **Simulate** — `SmartGridEnv` runs 24-hour days with synthetic or real data; agent gets observations and takes actions; reward and comfort constraint drive learning.
3. **Train** — PPO learns from many episodes; best and final models saved.
4. **Evaluate** — RL and baseline run for 10 episodes; summary (reward, peak, cost, comfort) and one RL trace saved.
5. **Visualize** — Plots (grid/PV, SOC, comfort, actions) saved; same data available in the web app.
6. **Present** — Web UI shows metrics, charts, and live simulation so you can explain and demo the project in the review.

---

## 1.5 Key Technical Points to Mention in Review

- **Gymnasium** — Standard RL API (reset, step, observation/action spaces).
- **PPO** — On-policy, stable, from Stable-Baselines3.
- **Multi-objective reward** — Peak + cost + comfort, with **adaptive weights** (stronger peak penalty at peak hours, stronger comfort penalty when comfort is low).
- **Hard constraint** — Episode ends and -100 reward if comfort < 0.60.
- **Baseline** — Rule-based; RL is compared against it in the same env.
- **Real data option** — Same env and training/eval code work with a CSV of real PV/load/price.
- **Reproducibility** — Config-driven; one script (`run_all.py`) for full pipeline; web app for demos.

---

# PART 2: WHAT WE WILL ADD IN THE COMING DAYS (AFTER THIS REVIEW)

Use this as your **post-review roadmap** and to set tasks in your planner.

---

## 2.1 Short-Term (Next 1–2 Weeks)

| # | What to add | Why |
|---|-------------|-----|
| 1 | **More evaluation episodes** | Run 20–50 episodes for RL and baseline and report mean ± std (and maybe confidence intervals) so results are more convincing. |
| 2 | **Hyperparameter tuning** | Try different learning rates, n_steps, batch size, or reward weights; document which setting works best. |
| 3 | **TensorBoard summary in README** | Add how to run TensorBoard on `results/logs/tensorboard/` and what to look at (reward curve, episode length). |
| 4 | **Export from web app** | Implement “Export Results” to download summary CSV and/or episode trace CSV from the browser. |
| 5 | **Error handling in app** | Friendly messages when model or results are missing (e.g. “Train first” with a short instruction). |
| 6 | **README update** | Add “How to run the web app,” “How to run a live simulation,” and “Where to find results and plots.” |

---

## 2.2 Medium-Term (2–4 Weeks)

| # | What to add | Why |
|---|-------------|-----|
| 7 | **Multiple seeds / variance** | Train 3–5 seeds; report mean ± std of evaluation metrics across seeds to show stability. |
| 8 | **Sensitivity analysis** | Change battery size, PV size, or comfort threshold in config and show how peak/cost/comfort change (tables or plots). |
| 9 | **Ablation** | Train with (a) full reward, (b) no comfort term, (c) no peak term; compare to show each objective matters. |
| 10 | **Real dataset from public source** | If possible, plug in a real public dataset (e.g. household or building) and show one training run and evaluation on real data. |
| 11 | **Baseline variants** | Add a second baseline (e.g. “always do nothing” or “charge only at midday, discharge only at peak”) and compare in summary and UI. |
| 12 | **Training curves in web app** | Load `monitor.csv` and plot episode reward and length over time in the dashboard. |

---

## 2.3 Longer-Term / Optional (1–2 Months)

| # | What to add | Why |
|---|-------------|-----|
| 13 | **Multi-day or week-long episodes** | Extend env to 7 days or 168 hours to study daily patterns and weekly storage use. |
| 14 | **Continuous action space** | Replace discrete 5 actions with continuous (e.g. charge/discharge power in kW); use PPO or SAC with continuous actions. |
| 15 | **Different algorithms** | Train SAC or A2C and add them to evaluation and summary comparison. |
| 16 | **Constraint satisfaction report** | In evaluation, report % of episodes that never hit comfort < 0.60 (constraint satisfaction rate). |
| 17 | **Pareto front** | Vary reward weights and plot Pareto front (peak vs cost vs comfort) for multi-objective analysis. |
| 18 | **Deployment note** | Short doc or script for running the app in production (e.g. gunicorn, no debug mode). |

---

## 2.4 Summary Table for Your Planner

| Phase        | Items | Example tasks |
|-------------|-------|----------------|
| **After review (1–2 wk)**  | 1–6   | More eval episodes; tuning; TensorBoard; export button; README. |
| **Next 2–4 wk**           | 7–12  | Multi-seed; sensitivity; ablation; real data; extra baseline; training curves in UI. |
| **Optional later**       | 13–18 | Longer episodes; continuous actions; other algorithms; constraint rate; Pareto; deployment. |

---

You can copy **Part 1** into your presentation or script for the review, and use **Part 2** to create tasks in Microsoft Planner for the coming days and weeks.
