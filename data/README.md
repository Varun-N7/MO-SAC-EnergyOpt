# Real-World Dataset Integration

## Overview

The project now supports using **real-world datasets** instead of synthetic data generation. This makes the RL training more realistic and applicable to actual scenarios.

## Current Dataset

**Location:** `data/real_world/real_world_energy_data.csv`

**Characteristics:**
- **8,760 records** (365 days × 24 hours)
- **Date range:** 2023-01-01 to 2023-12-31
- **Based on:** Published real-world energy consumption patterns
- **Includes:**
  - PV generation (solar) with seasonal variations
  - Residential load with morning/evening peaks
  - TOU pricing structure
  - Weekend/weekday patterns
  - Seasonal effects

## Using Real Data in Training

### Option 1: Use Current Dataset

The dataset is already created. To use it:

1. **Train with real data:**
   ```bash
   python train/train_ppo_real_data.py
   ```

2. **Evaluate with real data:**
   ```bash
   python evaluation/evaluate_real_data.py
   ```

### Option 2: Use Your Own Dataset

1. **Prepare your CSV file** with columns:
   - `datetime`: Timestamp
   - `hour`: Hour of day (0-23)
   - `pv_generation_kw`: PV generation in kW
   - `load_kw`: Load consumption in kW
   - `price_per_kwh`: Electricity price in $/kWh

2. **Update config.py:**
   ```python
   USE_REAL_DATA = True
   REAL_DATASET_PATH = "path/to/your/dataset.csv"
   ```

3. **Train:**
   ```bash
   python train/train_ppo_real_data.py
   ```

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

### Download Script Example

```python
# Example: Download from UCI
import pandas as pd

# UCI dataset (if available)
url = "https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption"
# Process and save as CSV with required columns
```

## Dataset Format

Your CSV file should have these columns:

```csv
datetime,pv_generation_kw,load_kw,price_per_kwh,hour
2023-01-01 00:00:00,0.0,2.1,0.10,0
2023-01-01 01:00:00,0.0,1.9,0.10,1
...
```

**Required columns:**
- `datetime`: Timestamp (can be string or datetime)
- `pv_generation_kw`: PV generation in kW (float)
- `load_kw`: Load consumption in kW (float)
- `price_per_kwh`: Price in $/kWh (float)
- `hour`: Hour of day 0-23 (int, optional - will be extracted from datetime)

## How It Works

### Environment Integration

The `SmartGridEnv` now supports two modes:

1. **Synthetic Mode (default):**
   ```python
   env = SmartGridEnv(use_real_data=False)
   # Uses PVModel, LoadModel, PriceModel
   ```

2. **Real Data Mode:**
   ```python
   env = SmartGridEnv(use_real_data=True, dataset_path="data/real_world/real_world_energy_data.csv")
   # Uses RealDataLoader to read from CSV
   ```

### Data Loading

- `RealDataLoader` reads the CSV file
- Each episode uses 24 consecutive hours from the dataset
- Episodes can start at random points in the dataset
- Data is loaded once and reused for multiple episodes

## Benefits of Real Data

1. **More Realistic:** Actual consumption patterns vs synthetic
2. **Seasonal Variations:** Real weather effects on PV and load
3. **Real Patterns:** Actual morning/evening peaks, weekend effects
4. **Better Generalization:** Model learns from real-world scenarios

## Comparison: Synthetic vs Real Data

| Aspect | Synthetic | Real Data |
|--------|-----------|-----------|
| **Source** | Mathematical models | Actual measurements |
| **Variability** | Controlled randomness | Natural variations |
| **Patterns** | Simplified | Complex, realistic |
| **Seasonal Effects** | Basic | Full seasonal cycles |
| **Weather Effects** | Simulated noise | Actual weather impact |

## Next Steps

1. **Use current dataset:** Already created, ready to use
2. **Download real dataset:** Follow instructions above
3. **Train with real data:** `python train/train_ppo_real_data.py`
4. **Compare results:** See how real data affects performance

## Notes

- The current dataset is based on **real-world patterns** from published studies
- For **actual measured data**, download from sources listed above
- Dataset must have at least 24 hours of data for one episode
- More data = more diverse training scenarios
