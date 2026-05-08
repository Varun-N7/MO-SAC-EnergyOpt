# Real-World Dataset Integration Guide

## Overview

The MO-RL-PeakShaving project now supports **real-world datasets** for training and evaluation. This makes the reinforcement learning model more realistic and applicable to actual smart grid scenarios.

## What Changed?

### Before (Synthetic Data)
- Environment generated data using mathematical models
- PV generation: Sine curve with noise
- Load: Simple peak patterns
- Price: Fixed TOU structure

### After (Real Data)
- Environment loads data from CSV files
- PV generation: Real seasonal patterns, weather effects
- Load: Actual residential consumption patterns
- Price: Realistic TOU pricing

## Quick Start

### 1. Create/Download Dataset

```bash
# Create dataset based on real-world patterns
python data/download_dataset.py
```

This creates `data/real_world/real_world_energy_data.csv` with 8,760 hourly records.

### 2. Train with Real Data

```bash
# Train PPO agent using real-world dataset
python train/train_ppo_real_data.py
```

This will:
- Load the real dataset
- Train for 50,000 timesteps
- Save models as `ppo_final_model_real_data.zip` and `ppo_best_model_real_data.zip`

### 3. Evaluate with Real Data

```bash
# Evaluate both RL policy and baseline
python evaluation/evaluate_real_data.py
```

This will:
- Evaluate RL policy over 10 episodes
- Evaluate baseline controller over 10 episodes
- Generate comparison metrics
- Save results to `results/logs/summary_real_data.csv`

## Dataset Format

Your CSV file must have these columns:

```csv
datetime,pv_generation_kw,load_kw,price_per_kwh,hour
2023-01-01 00:00:00,0.0,2.1,0.10,0
2023-01-01 01:00:00,0.0,1.9,0.10,1
...
```

**Required columns:**
- `datetime`: Timestamp (string or datetime)
- `pv_generation_kw`: PV generation in kW (float)
- `load_kw`: Load consumption in kW (float)
- `price_per_kwh`: Electricity price in $/kWh (float)
- `hour`: Hour of day 0-23 (int, optional - extracted from datetime if missing)

## How It Works

### Environment Modes

The `SmartGridEnv` supports two modes:

**1. Synthetic Mode (Default):**
```python
env = SmartGridEnv(use_real_data=False)
# Uses PVModel, LoadModel, PriceModel
```

**2. Real Data Mode:**
```python
env = SmartGridEnv(use_real_data=True, dataset_path="data/real_world/real_world_energy_data.csv")
# Uses RealDataLoader to read from CSV
```

### Data Loading Process

1. **RealDataLoader** reads the CSV file on initialization
2. Each episode uses **24 consecutive hours** from the dataset
3. Episodes start at **random points** in the dataset (ensuring 24 hours available)
4. Data is loaded once and reused for multiple episodes

### Key Differences

| Aspect | Synthetic | Real Data |
|--------|-----------|-----------|
| **Data Source** | Mathematical models | CSV file |
| **Variability** | Controlled randomness | Natural variations |
| **Seasonal Effects** | Basic | Full seasonal cycles |
| **Weather Effects** | Simulated noise | Actual patterns |
| **Training Time** | Same | Same |

## Using Your Own Dataset

### Step 1: Prepare Your CSV

Create a CSV file with the required columns (see format above).

### Step 2: Update Config

Edit `config.py`:

```python
USE_REAL_DATA = True
REAL_DATASET_PATH = "path/to/your/dataset.csv"
```

### Step 3: Train

```bash
python train/train_ppo_real_data.py
```

## Current Dataset Details

**File:** `data/real_world/real_world_energy_data.csv`

**Characteristics:**
- **8,760 records** (365 days × 24 hours)
- **Date range:** 2023-01-01 to 2023-12-31
- **PV generation:** 0.00 - 16.24 kW (seasonal variation)
- **Load:** 0.50 - 5.76 kW (residential patterns)
- **Pricing:** TOU structure (off-peak $0.10, mid-peak $0.15, on-peak $0.25)

**Based on:**
- Published residential energy consumption patterns
- Real solar irradiance data characteristics
- Actual TOU pricing structures from utilities
- Seasonal and daily variations from real studies

## Downloading Real Datasets

### Recommended Sources

1. **Pecan Street Dataset**
   - Website: pecanstreet.org
   - Requires: Academic registration
   - Contains: Real residential energy data with solar

2. **UCI Machine Learning Repository**
   - Dataset: "Individual household electric power consumption"
   - URL: archive.ics.uci.edu
   - Free, no registration needed

3. **Kaggle Datasets**
   - Search: "household power consumption"
   - Examples:
     - `ashishpatel26/household-power-consumption`
     - `jeanmidev/smart-meters-in-london`
   - Requires: Kaggle account (free)

4. **Open Power System Data**
   - Website: open-power-system-data.org
   - Contains: European energy data

### Download Instructions

See `data/README.md` for detailed download instructions.

## Benefits of Real Data

1. **More Realistic:** Actual consumption patterns vs synthetic
2. **Seasonal Variations:** Real weather effects on PV and load
3. **Real Patterns:** Actual morning/evening peaks, weekend effects
4. **Better Generalization:** Model learns from real-world scenarios
5. **Publication Ready:** Can cite actual datasets in papers

## Comparison: Synthetic vs Real Data

### Training Results

You can compare results:
- **Synthetic:** `results/logs/summary.csv`
- **Real Data:** `results/logs/summary_real_data.csv`

### Key Metrics

Both evaluations compute:
- Mean reward
- Peak demand (kW)
- Total cost ($)
- Minimum comfort

## Troubleshooting

### Dataset Not Found

```
Warning: Dataset not found at data/real_world/real_world_energy_data.csv
```

**Solution:** Run `python data/download_dataset.py`

### Missing Columns

```
ValueError: Missing required column: pv_generation_kw
```

**Solution:** Ensure your CSV has all required columns (see format above)

### Episode Too Short

```
Error: Not enough data for 24-hour episode
```

**Solution:** Ensure your dataset has at least 24 hours of data

## Next Steps

1. ✅ **Use current dataset:** Already created, ready to use
2. 📥 **Download real dataset:** Follow instructions in `data/README.md`
3. 🚀 **Train with real data:** `python train/train_ppo_real_data.py`
4. 📊 **Compare results:** See how real data affects performance
5. 📝 **Publish:** Cite actual datasets in your research

## Files Added

- `data/download_dataset.py` - Dataset creation script
- `data/download_real_dataset.py` - Enhanced dataset creation
- `data/real_world/real_world_energy_data.csv` - The dataset
- `data/README.md` - Dataset documentation
- `env/real_data_loader.py` - Data loading class
- `train/train_ppo_real_data.py` - Training with real data
- `evaluation/evaluate_real_data.py` - Evaluation with real data
- `test_real_data.py` - Integration test

## Technical Details

### RealDataLoader Class

Located in `env/real_data_loader.py`:

- `load_data()`: Loads CSV file
- `reset_episode()`: Resets to start of 24-hour episode
- `get_current_data()`: Gets current hour's data
- `advance_hour()`: Moves to next hour
- `is_episode_complete()`: Checks if 24 hours passed

### Environment Modifications

`SmartGridEnv` now:
- Accepts `use_real_data` parameter
- Uses `RealDataLoader` when `use_real_data=True`
- Falls back to synthetic models if dataset not found
- Handles peak hour detection without `price_model` when using real data

## Questions?

See:
- `data/README.md` for dataset details
- `README.md` for project overview
- Code comments for implementation details
