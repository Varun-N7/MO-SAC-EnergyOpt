# Understanding RL Metrics vs Traditional ML

## Key Difference

| Traditional ML (Supervised) | This RL Project |
|----------------------------|-----------------|
| Uses labeled dataset | Environment generates data dynamically |
| Measures **accuracy** | Measures **performance** |
| Train/test split | Train on environment, evaluate on episodes |
| Classification/Regression | Sequential decision-making |

## What Gets Measured Instead of Accuracy

### 1. **Reward** (Primary Success Metric)
```
Higher reward = Better performance
```
- Combines all objectives into single score
- Peak shaving: -10 points if demand > 6 kW
- Cost: -1 point per $1 spent
- Comfort: -5 points if comfort < 0.80
- **Goal**: Maximize cumulative reward over 24 hours

### 2. **Peak Demand** (kW)
```
Lower is better
```
- Maximum grid import during day
- Target: Keep below 6.0 kW threshold
- Shows how well agent shaves peaks

### 3. **Total Cost** ($)
```
Lower is better
```
- Sum of (grid_import × price) for 24 hours
- Measures economic efficiency
- Agent learns to minimize costs

### 4. **Minimum Comfort**
```
Higher is better (must be > 0.60)
```
- Lowest comfort during episode
- Hard constraint: episode ends if < 0.60
- Shows comfort maintenance

## Example Evaluation Output

When you run evaluation, you'll see:

```
Evaluating RL Policy...
------------------------------------------------------------
Mean Reward: -42.15 ± 8.23
Mean Peak Demand: 5.23 ± 0.67 kW
Mean Total Cost: $11.89 ± $1.45
Mean Min Comfort: 0.78 ± 0.05

Evaluating Baseline Controller...
------------------------------------------------------------
Mean Reward: -58.34 ± 12.11
Mean Peak Demand: 6.89 ± 1.23 kW
Mean Total Cost: $15.67 ± $2.34
Mean Min Comfort: 0.72 ± 0.09
```

**Interpretation**: RL agent performs better than baseline:
- Higher reward (less negative)
- Lower peak demand
- Lower cost
- Better comfort maintenance

## Training Progress (Like Learning Curve)

During training, you'll see improving metrics:
- Episode 1: Mean reward = -120
- Episode 100: Mean reward = -80
- Episode 500: Mean reward = -50
- Episode 1000: Mean reward = -40

This shows the agent is learning!

## Data Generation (No Dataset Needed)

The environment creates data on-the-fly:

```python
# Each episode generates:
- PV generation (stochastic, Gaussian noise)
- Load curve (morning + evening peaks)
- TOU prices (time-based)
- Battery state (SOC dynamics)
- Comfort score (based on actions)
```

No pre-collected dataset needed - the environment simulates realistic scenarios.

## Success Criteria

The agent succeeds if it:
1. ✅ Keeps peak demand < 6.0 kW
2. ✅ Minimizes total electricity cost
3. ✅ Maintains comfort > 0.60 (hard constraint)
4. ✅ Balances all objectives optimally

This is measured by **reward**, not accuracy!

