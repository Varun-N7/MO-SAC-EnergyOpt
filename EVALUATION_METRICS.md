# Evaluation Metrics for RL Project

## Why No Dataset or Accuracy?

This is a **Reinforcement Learning (RL)** project, not supervised learning:

- **Supervised Learning**: Uses labeled dataset → measures accuracy
- **Reinforcement Learning**: Agent learns by interacting with environment → measures performance

## How RL Works Here

1. **No Dataset Needed**: The environment (`smart_grid_env.py`) generates data on-the-fly
   - PV generation (stochastic, changes each episode)
   - Load patterns (morning/evening peaks)
   - Prices (TOU pricing)
   - All generated dynamically during training/evaluation

2. **Agent Learns by Trial and Error**: 
   - Takes actions (discharge, charge, shift load, etc.)
   - Receives rewards/penalties
   - Learns optimal policy through experience

## Evaluation Metrics (Instead of Accuracy)

The project evaluates performance using these metrics:

### 1. **Mean Reward** (Primary Metric)
   - Higher is better
   - Measures overall policy performance
   - Combines: peak shaving, cost reduction, comfort maintenance

### 2. **Peak Demand (kW)**
   - Lower is better
   - Maximum grid import during 24-hour episode
   - Goal: Keep below threshold (6.0 kW)

### 3. **Total Cost ($)**
   - Lower is better
   - Sum of (grid_import × price) over 24 hours
   - Measures economic efficiency

### 4. **Minimum Comfort**
   - Higher is better (must stay > 0.60)
   - Lowest comfort score during episode
   - Hard constraint: episode terminates if < 0.60

## Example Evaluation Output

After running `python evaluation/evaluate.py`, you'll see:

```
Mean Reward: -45.23 ± 12.34
Mean Peak Demand: 5.67 ± 0.89 kW
Mean Total Cost: $12.45 ± $2.10
Mean Min Comfort: 0.75 ± 0.08
```

## Comparison: RL vs Baseline

The evaluation compares:
- **RL_PPO**: Learned policy (trained agent)
- **Baseline**: Rule-based heuristic

Results saved to: `results/logs/summary.csv`

## Training Progress Metrics

During training, you'll see:
- **Episode reward**: Cumulative reward per episode
- **Mean reward**: Average reward (evaluated periodically)
- **Learning curve**: Improving over time

## Visual Evaluation

Plots generated in `results/plots/`:
- Grid import vs PV generation
- Battery SOC over time
- Comfort score over time
- Action distribution

These show how well the agent performs, not accuracy.

## Summary

- ❌ No dataset (environment generates data)
- ❌ No accuracy metric (not classification)
- ✅ Performance metrics: reward, peak demand, cost, comfort
- ✅ Comparison: RL agent vs baseline controller
- ✅ Visual analysis: plots showing agent behavior

The "accuracy" in RL is measured by how well the agent achieves the objectives:
- Reducing peak demand
- Minimizing costs
- Maintaining comfort
- Balancing all objectives optimally

