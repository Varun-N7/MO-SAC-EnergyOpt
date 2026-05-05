# MO-RL-PeakShaving: Complete Project Explanation

## 🎯 What This Project Does (Simple Explanation)

### The Problem We're Solving

Imagine you have a **smart home** with:
- ☀️ **Solar panels** generating electricity
- 🔋 **Battery** to store electricity  
- 🏠 **Appliances** using electricity (AC, lights, TV, etc.)
- 💰 **Electricity prices** that change throughout the day (cheaper at night, expensive in evening)

**The Challenge:**
- Electricity costs **more during peak hours** (evening 6-10 PM)
- You want to **reduce peak demand** to save money
- But you also want to **stay comfortable** (don't want to turn off AC completely!)
- Need to **balance** cost savings with comfort

### The Solution

An **AI agent** (using Reinforcement Learning) learns the best strategy to:
1. **Charge battery** when solar is generating or prices are low
2. **Discharge battery** when prices are high (peak hours)
3. **Shift flexible loads** to cheaper times (like running dishwasher at night)
4. **Slightly reduce comfort** (turn AC down a bit) when absolutely necessary

**The Goal:** Balance three objectives:
- ✅ **Reduce peak electricity demand** (save money on peak charges)
- ✅ **Minimize total electricity cost** (use battery and solar efficiently)
- ✅ **Maintain consumer comfort** (don't make people suffer!)

---

## 🧠 How It Works (Technical Deep Dive)

### 1. Reinforcement Learning Basics

**What is RL?**
- The AI **learns by trial and error**
- Like learning to ride a bike: try, fall, learn, get better
- The agent gets **rewards** for good decisions, **penalties** for bad ones

**Key Components:**
- **Agent**: The AI that makes decisions (PPO algorithm)
- **Environment**: The smart grid simulation (PV, battery, load, prices)
- **State/Observation**: What the agent sees: `[hour, pv, load, price, soc, comfort]`
- **Actions**: What the agent can do (5 discrete actions)
- **Reward**: Points for good decisions, penalties for bad ones
- **Policy**: The learned strategy (what to do in each situation)

### 2. The Environment (Smart Grid Simulation)

#### 2.1 Observation Space
The agent sees 6 values at each step:
```python
observation = [hour, pv, load, price, soc, comfort]
# hour: 0-23 (time of day)
# pv: 0-15 kW (solar generation)
# load: 0-10 kW (household consumption)
# price: 0-0.5 $/kWh (electricity price)
# soc: 0-1 (battery state of charge, 0=empty, 1=full)
# comfort: 0-1 (consumer comfort, 1=perfect, 0=uncomfortable)
```

#### 2.2 Action Space
The agent can choose 1 of 5 actions:
```python
0: Do nothing
1: Discharge battery (use stored energy)
2: Charge battery (store energy)
3: Shift flexible load (delay some consumption)
4: Comfort sacrifice (reduce HVAC, slight discomfort)
```

#### 2.3 Environment Components

**A. PV (Solar) Model** (`env/pv_model.py`)
- Simulates solar panel generation
- Pattern: Low in morning → Peak at noon → Low in evening
- Adds randomness (Gaussian noise) for realistic weather variations
- **With Real Data**: Uses actual PV generation from dataset

**B. Load Model** (`env/load_model.py`)
- Simulates household electricity consumption
- **Morning peak** around 7 AM (people waking up, breakfast)
- **Evening peak** around 7 PM (cooking, TV, lights)
- **Flexible load**: 15% can be shifted to different times
- **With Real Data**: Uses actual residential load patterns from dataset

**C. Price Model** (`env/price_model.py`)
- Implements **Time-of-Use (TOU) pricing**
- **Off-peak** (night): $0.10/kWh
- **Mid-peak** (day): $0.15/kWh
- **On-peak** (evening 6-10 PM): $0.25/kWh
- **With Real Data**: Uses actual pricing from dataset

**D. Battery Model** (in `smart_grid_env.py`)
- Capacity: 10 kWh
- Max charge/discharge: 5 kW
- Efficiency: 95% (some energy lost during charge/discharge)
- Tracks State of Charge (SOC): 0 (empty) to 1 (full)

**E. Comfort Model** (in `smart_grid_env.py`)
- Starts at 1.0 (perfect comfort)
- Decreases when:
  - Load is shifted: -0.05
  - HVAC is reduced: -0.10
- Recovers slowly: +0.02 per hour when no actions taken
- **Hard constraint**: Episode ends if comfort < 0.60

### 3. Reward Function (Multi-Objective with Adaptive Weights)

The reward balances three objectives:

#### 3.1 Peak Demand Penalty
```python
if grid_import > 6.0 kW:  # Threshold exceeded
    penalty = -10.0
    if peak_hour:  # During expensive hours (6-10 PM)
        penalty *= 2.0  # Double penalty
    if peak_hour:
        penalty *= 1.5  # Adaptive weight increase
```

**Adaptive Weighting:**
- During peak hours (6-10 PM), peak penalty weight increases by 1.5x
- This makes the agent prioritize reducing peak demand during expensive times

#### 3.2 Grid Cost
```python
cost = grid_import * price * 1.0
reward -= cost  # Negative (penalty)
```
- Penalty for buying electricity from grid
- Higher price = higher penalty
- Encourages using battery and solar

#### 3.3 Comfort Violation Penalty
```python
if comfort < 0.80:
    penalty = -5.0
    if comfort < 0.75:
        penalty *= 2.0  # Adaptive weight increase
    if comfort < 0.70:
        penalty *= 3.0  # Extra multiplier
```

**Adaptive Weighting:**
- When comfort is low (< 0.75), comfort penalty weight increases by 2.0x
- This makes the agent prioritize comfort when it's getting uncomfortable

#### 3.4 Hard Constraint
```python
if comfort < 0.60:
    reward += -100.0  # Large penalty
    episode_terminated = True  # End episode
```
- **Safety mechanism**: Prevents agent from making people too uncomfortable
- Episode ends immediately if comfort drops too low

### 4. Training Process

#### 4.1 PPO Algorithm (Proximal Policy Optimization)
- **On-policy** algorithm (learns from current policy)
- **Stable** training (uses clipping to prevent large updates)
- **Efficient** for discrete and continuous actions

**Training Steps:**
1. Agent interacts with environment (collects experiences)
2. Computes advantages (how good/bad each action was)
3. Updates policy network (learns better strategy)
4. Repeats for 50,000 timesteps

#### 4.2 Training Script (`train/train_ppo.py`)
```python
# Creates environment
env = SmartGridEnv()
env = DummyVecEnv([lambda: env])  # Vectorized for speed

# Creates PPO agent
model = PPO("MlpPolicy", env, ...)

# Trains for 50,000 timesteps
model.learn(total_timesteps=50000)

# Saves models
model.save("models/ppo_final_model.zip")
```

**Callbacks:**
- **CheckpointCallback**: Saves model every 10,000 steps
- **SaveBestModelCallback**: Saves best model based on evaluation

### 5. Evaluation Process

#### 5.1 Baseline Controller (`evaluation/baseline_controller.py`)
Simple rule-based policy for comparison:
```python
if hour in [18, 19, 20, 21] and soc > 0.7:
    action = DISCHARGE  # Use battery during peak hours
elif 11 <= hour <= 14 and pv > load:
    action = CHARGE  # Charge when solar is high
elif pv > load and comfort > 0.8:
    action = SHIFT_LOAD  # Shift load when conditions good
else:
    action = DO_NOTHING
```

#### 5.2 Evaluation Metrics
- **Peak Demand**: Maximum grid import (kW) during episode
- **Total Cost**: Sum of (grid_import × price) for all hours
- **Minimum Comfort**: Lowest comfort value during episode
- **Mean Reward**: Average reward per step

#### 5.3 Evaluation Script (`evaluation/evaluate.py`)
- Runs 10 episodes for RL policy
- Runs 10 episodes for baseline
- Computes and compares metrics
- Saves results to CSV
- Generates plots

---

## 📊 Real-World Dataset Integration

### Why Real Data?

**Synthetic Data (Before):**
- Mathematical models generate data
- Simplified patterns
- May not reflect real-world complexity

**Real Data (Now):**
- Actual energy consumption patterns
- Real seasonal variations
- Weather effects on PV generation
- More realistic for deployment

### Dataset Details

**Location:** `data/real_world/real_world_energy_data.csv`

**Characteristics:**
- **8,760 records** (365 days × 24 hours)
- **Date range:** 2023-01-01 to 2023-12-31
- **PV generation:** 0.00 - 16.24 kW (with seasonal variation)
- **Load:** 0.50 - 5.76 kW (realistic residential patterns)
- **TOU pricing:** Off-peak $0.10, Mid-peak $0.15, On-peak $0.25

**Based on:**
- Published residential energy consumption patterns
- Real solar irradiance data characteristics
- Actual TOU pricing structures from utilities
- Seasonal and daily variations from real studies

### How to Use Real Data

**1. Create Dataset:**
```bash
python data/download_dataset.py
```

**2. Train with Real Data:**
```bash
python train/train_ppo_real_data.py
```

**3. Evaluate with Real Data:**
```bash
python evaluation/evaluate_real_data.py
```

### Environment Modes

**Synthetic Mode (Default):**
```python
env = SmartGridEnv(use_real_data=False)
# Uses PVModel, LoadModel, PriceModel
```

**Real Data Mode:**
```python
env = SmartGridEnv(use_real_data=True, dataset_path="data/real_world/real_world_energy_data.csv")
# Uses RealDataLoader to read from CSV
```

---

## 🚀 How to Use the Project

### Step 1: Installation
```bash
pip install -r requirements.txt
```

### Step 2: Training Options

**Option A: Train with Synthetic Data**
```bash
python train/train_ppo.py
```

**Option B: Train with Real Data**
```bash
# First, create dataset
python data/download_dataset.py

# Then train
python train/train_ppo_real_data.py
```

### Step 3: Evaluation

**Option A: Evaluate with Synthetic Data**
```bash
python evaluation/evaluate.py
```

**Option B: Evaluate with Real Data**
```bash
python evaluation/evaluate_real_data.py
```

### Step 4: View Results

**Results Location:**
- `results/logs/summary.csv` - Comparison metrics
- `results/logs/episode_trace_rl.csv` - Detailed episode data
- `results/plots/` - Visualization plots

**Web Frontend:**
```bash
python app.py
# Open http://localhost:5000
```

---

## 📁 Project Structure

```
MO-RL-PeakShaving/
├── README.md                    # Main documentation
├── requirements.txt             # Dependencies
├── config.py                    # Configuration parameters
├── run_all.py                   # Run training + evaluation
├── app.py                       # Flask web frontend
│
├── env/                         # Environment components
│   ├── __init__.py
│   ├── smart_grid_env.py        # Main Gymnasium environment
│   ├── pv_model.py              # Solar generation model
│   ├── load_model.py            # Load consumption model
│   ├── price_model.py           # TOU pricing model
│   └── real_data_loader.py      # Real data loader
│
├── train/                       # Training scripts
│   ├── train_ppo.py             # Train with synthetic data
│   ├── train_ppo_real_data.py   # Train with real data
│   └── callbacks.py             # Training callbacks
│
├── evaluation/                  # Evaluation scripts
│   ├── baseline_controller.py   # Rule-based baseline
│   ├── evaluate.py              # Evaluate with synthetic data
│   ├── evaluate_real_data.py   # Evaluate with real data
│   └── plot_results.py         # Generate plots
│
├── data/                        # Dataset files
│   ├── download_dataset.py      # Create dataset
│   ├── download_real_dataset.py # Enhanced dataset creation
│   ├── README.md               # Dataset documentation
│   └── real_world/             # Dataset storage
│       └── real_world_energy_data.csv
│
├── models/                      # Saved models
│   ├── ppo_final_model.zip
│   ├── ppo_best_model.zip
│   └── ...
│
├── results/                     # Results and outputs
│   ├── logs/
│   │   ├── summary.csv
│   │   ├── episode_trace_rl.csv
│   │   └── monitor.csv
│   └── plots/
│       ├── grid_import.png
│       ├── soc.png
│       └── ...
│
└── templates/                   # Web frontend
    ├── index.html
    └── static/
        ├── style.css
        └── app.js
```

---

## 🔧 Configuration

Edit `config.py` to adjust:

**Battery:**
- Capacity: 10 kWh
- Max charge/discharge: 5 kW
- Efficiency: 95%

**PV Generation:**
- Peak power: 8.0 kW
- Noise: 0.5 kW std dev

**Load:**
- Base: 1.0 - 3.0 kW
- Morning peak: 7 AM, 1.8x multiplier
- Evening peak: 7 PM, 2.2x multiplier
- Flexible: 15% of load

**Pricing:**
- Off-peak: $0.10/kWh
- Mid-peak: $0.15/kWh
- On-peak: $0.25/kWh (hours 18-21)

**Rewards:**
- Peak threshold: 6.0 kW
- Peak penalty: -10.0 (base), -20.0 (peak hours)
- Comfort violation: -5.0 (base), -15.0 (low comfort)
- Hard constraint: -100.0 (comfort < 0.60)

**Training:**
- Timesteps: 50,000
- Learning rate: 3e-4
- Batch size: 64
- Gamma: 0.99

---

## 📈 Key Features

### 1. Multi-Objective Optimization
- Balances peak demand, cost, and comfort
- Adaptive weights adjust importance based on situation

### 2. Hard Constraints
- Episode terminates if comfort < 0.60
- Prevents agent from making people too uncomfortable

### 3. Real-World Integration
- Supports both synthetic and real datasets
- Can use actual energy consumption data

### 4. Comprehensive Evaluation
- Compares RL policy vs baseline
- Multiple metrics (peak, cost, comfort)
- Visualizations and plots

### 5. Web Frontend
- Interactive visualization
- Real-time metrics
- Episode analysis
- Professional presentation

---

## 🎓 Key Concepts Explained

### Reinforcement Learning
- **Agent**: The AI learner
- **Environment**: The smart grid simulation
- **State**: Current situation (hour, PV, load, price, SOC, comfort)
- **Action**: What to do (charge, discharge, shift, etc.)
- **Reward**: Feedback (positive for good, negative for bad)
- **Policy**: Learned strategy (what action in each state)

### Multi-Objective Optimization
- **Multiple goals**: Peak reduction, cost minimization, comfort maintenance
- **Trade-offs**: Sometimes need to sacrifice one for another
- **Adaptive weights**: Importance changes based on situation
  - Peak hours → prioritize peak reduction
  - Low comfort → prioritize comfort

### Peak Load Shaving
- **Peak demand**: Maximum electricity drawn from grid
- **Peak hours**: Expensive times (evening 6-10 PM)
- **Shaving**: Reducing peak demand using:
  - Battery discharge
  - Load shifting
  - Solar generation

### Consumer Comfort
- **Comfort score**: 0 (uncomfortable) to 1 (perfect)
- **Decreases** when:
  - Load is shifted (inconvenience)
  - HVAC is reduced (temperature change)
- **Increases** slowly when no actions taken
- **Hard constraint**: Must stay above 0.60

---

## 🐛 Important Note

**Potential Issue with Real Data Mode:**

If you use real data mode (`use_real_data=True`), the environment sets `self.price_model = None`. However, the reward calculation in `_calculate_reward()` uses `self.price_model.is_peak_hour()`, which will cause an error.

**Solution:**
When using real data, check peak hours directly from config:
```python
is_peak = self.hour in config.PEAK_HOURS
```

Or ensure `price_model` is always initialized (even if not used for pricing).

---

## 📚 Further Reading

- **Reinforcement Learning**: Sutton & Barto "Reinforcement Learning: An Introduction"
- **PPO Algorithm**: Schulman et al. "Proximal Policy Optimization Algorithms"
- **Smart Grids**: Various IEEE papers on demand response and peak shaving
- **Multi-Objective RL**: Papers on adaptive weighting and constraint handling

---

## ✅ Summary

This project implements an **AI-powered smart grid controller** that:
1. **Learns** optimal strategies using Reinforcement Learning
2. **Balances** peak demand reduction, cost minimization, and comfort
3. **Uses** real-world data for realistic training
4. **Evaluates** performance against baseline controllers
5. **Visualizes** results through web interface

**Perfect for:**
- Research on smart grid optimization
- Learning Reinforcement Learning
- Demonstrating multi-objective optimization
- Real-world energy management applications
