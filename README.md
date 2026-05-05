# MO-RL-PeakShaving

Adaptive multi-objective reinforcement learning for peak load shaving in solar-integrated smart grids with comfort constraints.

## Project Description and Objectives

`MO-RL-PeakShaving` learns energy-control policies that balance four competing objectives:

- reduce grid **peak demand**
- minimize electricity **cost**
- maintain consumer **comfort**
- preserve **battery health** (degradation-aware control)

The project includes:
- a Gymnasium smart-grid environment with synthetic + real-data modes
- PPO and SAC training/evaluation pipelines
- ablation, Pareto, and algorithm-comparison experiments
- a Streamlit dashboard for interactive demonstration

## Installation

```bash
pip install -r requirements.txt
```

For dashboard:

```bash
pip install -r requirements_dashboard.txt
```

## How To Train

Primary unified entrypoint:

```bash
python train.py --algorithm ppo --seed 42
```

Alternative:

```bash
python train.py --algorithm sac --seed 42
```

## How To Run Each Experiment

### 1) Standard evaluation (baseline vs PPO)

```bash
python evaluation/evaluate.py
```

### 2) Real-data training/evaluation

```bash
python data/download_dataset.py
python train/train_ppo_real_data.py --seed 42
python evaluation/evaluate_real_data.py
```

### 3) Ablation study

```bash
python experiments/ablation_study.py --seed 42
python results/plot_ablation.py
```

### 4) Pareto front sweep

```bash
python experiments/pareto_sweep.py --seed 42
python results/plot_pareto.py
```

### 5) PPO vs SAC comparison

```bash
python experiments/compare_algorithms.py --seed 42
python results/plot_comparison.py
```

### 6) Final consolidated report

```bash
python results/generate_report.py
```

## How To Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard features:
- normalized reward-weight sliders (`λ_cost`, `λ_peak`, `λ_comfort`)
- algorithm selection (`PPO`, `SAC` if available)
- real-data episode filtering by month/season/date
- baseline overlay toggle
- multi-panel episode visualization + metric cards + battery health gauge

## Results Summary (Best Performing Configuration)

Run `python results/generate_report.py` after experiments to produce:

- `results/final_summary.csv`
- `results/final_summary.png`

Use this as the canonical comparison across:
- Rule-Based Baseline
- PPO
- SAC

> Best configuration depends on your latest trained artifacts and random seed. Re-generate the report after each full experiment run.

## Folder Structure Diagram

```text
MO-RL-PeakShaving/
├── config.py
├── seeds.py
├── train.py
├── run_all.py
├── README.md
├── requirements.txt
├── requirements_dashboard.txt
├── agents/
│   ├── __init__.py
│   └── sac_agent.py
├── env/
│   ├── __init__.py
│   ├── smart_grid_env.py
│   ├── pv_model.py
│   ├── load_model.py
│   ├── price_model.py
│   └── real_data_loader.py
├── train/
│   ├── train_ppo.py
│   ├── train_ppo_real_data.py
│   └── callbacks.py
├── evaluation/
│   ├── baseline_controller.py
│   ├── evaluate.py
│   ├── evaluate_real_data.py
│   └── plot_results.py
├── experiments/
│   ├── ablation_study.py
│   ├── pareto_sweep.py
│   └── compare_algorithms.py
├── results/
│   ├── logs/
│   ├── plots/
│   ├── plot_ablation.py
│   ├── plot_pareto.py
│   ├── plot_comparison.py
│   └── generate_report.py
├── dashboard/
│   └── app.py
├── data/
│   ├── download_dataset.py
│   ├── download_real_dataset.py
│   ├── README.md
│   └── real_world/
│       └── real_world_energy_data.csv
└── tests/
    ├── test_environment.py
    ├── test_battery_degradation.py
    └── test_data_loader.py
```

## Notes

- Observation space: `[hour, pv, load, price, soc, comfort, battery_health]`
- Action space: 5 discrete control actions
- Real dataset size target: 8,760 hourly records
- Reproducibility: all train/experiment scripts support `--seed`

