# MO-RL-PeakShaving: Complete Project Explanation

## 🎯 Simple Explanation (Easy Words)

### What is this project about?

Imagine you have a house with:
- **Solar panels** on the roof (generates electricity from sun)
- **Battery** to store electricity
- **Appliances** that use electricity (TV, AC, lights, etc.)
- **Electricity prices** that change throughout the day (cheaper at night, expensive in evening)

**The Problem:** 
- Electricity costs more during peak hours (evening 6-10 PM)
- You want to use less electricity from the grid during expensive times
- But you also want to stay comfortable (don't want to turn off AC completely!)

**The Solution:**
- An AI agent (like a smart assistant) learns the best times to:
  - Charge the battery (when solar is generating or prices are low)
  - Discharge the battery (when prices are high)
  - Shift some loads to cheaper times
  - Slightly reduce comfort (like turning AC down a bit) when needed

**The Goal:**
Balance three things:
1. ✅ Reduce peak electricity demand (save money)
2. ✅ Minimize total electricity cost
3. ✅ Keep you comfortable (don't make you suffer!)

---

## 📚 Detailed Technical Explanation

### 1. What is Reinforcement Learning?

**Simple:** The AI learns by trying different actions and seeing what works best. Like learning to ride a bike - you try, fall, learn, and get better.

**Technical:** 
- **Agent**: The AI that makes decisions
- **Environment**: The smart grid system (PV, battery, load, prices)
- **Actions**: What the agent can do (charge, discharge, shift load, etc.)
- **Rewards**: Points given for good decisions, penalties for bad ones
- **Policy**: The learned strategy (what to do in each situation)

### 2. The Environment (Smart Grid Simulation)

#### 2.1 PV (Solar) Model (`env/pv_model.py`)

**What it does:**
- Simulates solar panel electricity generation
- Follows a daily pattern: low in morning, peaks at noon, low in evening
- Adds randomness (Gaussian noise) to make it realistic

**Technical Details:**
```python
# Base generation follows sine curve
base_generation = sin(π * normalized_hour)  # 0 at 6 AM, peak at 12 PM, 0 at 6 PM
generation = base_generation * peak_power + noise
```

**Parameters:**
- Peak power: 8.0 kW (maximum generation at midday)
- Noise standard deviation: 0.5 kW (randomness)

#### 2.2 Load Model (`env/load_model.py`)

**What it does:**
- Simulates household electricity consumption
- Creates realistic patterns: morning peak (7 AM) and evening peak (7 PM)
- Models flexible load (15% can be shifted)

**Technical Details:**
```python
# Base load with random variation
base_load = random.uniform(1.0, 3.0) kW

# Morning peak effect (Gaussian around 7 AM)
morning_effect = 1.8 * exp(-distance² / 2)

# Evening peak effect (Gaussian around 7 PM)
evening_effect = 2.2 * exp(-distance² / 2)

total_load = base_load * (1 + morning_effect + evening_effect)
```

**Parameters:**
- Base load: 1.0 - 3.0 kW
- Morning peak: 7 AM, multiplier 1.8x
- Evening peak: 7 PM, multiplier 2.2x
- Flexible load: 15% of total can be shifted

#### 2.3 Price Model (`env/price_model.py`)

**What it does:**
- Implements Time-of-Use (TOU) pricing
- Different prices for different times of day

**Technical Details:**
```python
if hour in [18, 19, 20, 21]:  # Peak hours
    price = $0.25/kWh
elif 6 <= hour < 22:  # Daytime
    price = $0.15/kWh
else:  # Night
    price = $0.10/kWh
```

**Price Structure:**
- Off-peak (night): $0.10/kWh
- Mid-peak (day): $0.15/kWh
- On-peak (evening 6-10 PM): $0.25/kWh

#### 2.4 Battery Model (in `smart_grid_env.py`)

**What it does:**
- Models battery storage with charge/discharge dynamics
- Tracks State of Charge (SOC) - how full the battery is

**Technical Details:**
```python
# Battery specifications
Capacity: 10.0 kWh
Max charge power: 5.0 kW
Max discharge power: 5.0 kW
Charge efficiency: 95%
Discharge efficiency: 95%

# SOC update
SOC_new = SOC_old ± (power / capacity) * efficiency
```

**Constraints:**
- Can't charge beyond 100% SOC
- Can't discharge below 0% SOC
- Limited by max power ratings

#### 2.5 Comfort Model (in `smart_grid_env.py`)

**What it does:**
- Tracks consumer comfort score (0.0 to 1.0)
- Decreases when comfort-sacrificing actions are taken
- Recovers slowly when no actions taken

**Technical Details:**
```python
# Initial comfort
comfort = 1.0

# Actions reduce comfort
if action == "shift_load":
    comfort -= 0.05
if action == "comfort_sacrifice":
    comfort -= 0.10

# Recovery when doing nothing
if action == "do_nothing":
    comfort += 0.02  # Up to max 1.0

# Hard constraint
if comfort < 0.60:
    episode terminates with large penalty
```

**Parameters:**
- Initial comfort: 1.0 (100%)
- Shift load penalty: -0.05
- HVAC reduction penalty: -0.10
- Recovery rate: +0.02 per hour
- Hard threshold: 0.60 (episode ends if below)

### 3. The Environment (`env/smart_grid_env.py`)

#### 3.1 Observation Space

**What the agent sees:**
```
[hour, pv, load, price, soc, comfort]
```

**Details:**
- `hour`: 0-23 (time of day)
- `pv`: 0-15 kW (current solar generation)
- `load`: 0-10 kW (current electricity demand)
- `price`: 0-0.5 $/kWh (current electricity price)
- `soc`: 0-1.0 (battery state of charge, 0% to 100%)
- `comfort`: 0-1.0 (consumer comfort score)

#### 3.2 Action Space

**What the agent can do:**
1. **Do Nothing (0)**: No action, comfort recovers slightly
2. **Discharge Battery (1)**: Use battery to power house, reduce grid import
3. **Charge Battery (2)**: Store excess solar or grid electricity
4. **Shift Flexible Load (3)**: Move some loads to different time, reduces comfort
5. **Comfort Sacrifice (4)**: Reduce HVAC (AC/heating), reduces comfort more

#### 3.3 Reward Function (Multi-Objective)

**Simple:** Agent gets points for good decisions, loses points for bad ones.

**Technical Components:**

```python
reward = 0.0

# 1. Hard Constraint Penalty (BIG penalty if comfort too low)
if comfort < 0.60:
    reward += -100.0  # Episode ends
    return reward

# 2. Peak Demand Penalty (adaptive weight during peak hours)
if grid_import > 6.0 kW:
    peak_penalty = -10.0
    if peak_hour:
        peak_penalty *= 2.0  # Double penalty during peak hours
    if peak_hour:
        peak_penalty *= 1.5  # Adaptive weight multiplier
    reward += peak_penalty

# 3. Grid Cost (always negative)
cost = grid_import * price * 1.0
reward -= cost

# 4. Comfort Violation Penalty (adaptive weight when comfort low)
if comfort < 0.80:
    comfort_penalty = -5.0
    if comfort < 0.75:
        comfort_penalty *= 2.0  # Adaptive weight
    if comfort < 0.70:
        comfort_penalty *= 3.0  # Extra multiplier
    reward += comfort_penalty
```

**Adaptive Weights:**
- **Peak hours (18-22)**: Peak penalty weight × 1.5
- **Low comfort (< 0.75)**: Comfort penalty weight × 2.0

**Why Adaptive?**
- During peak hours, reducing demand is more important
- When comfort is low, maintaining comfort becomes priority
- Agent learns to balance objectives dynamically

### 4. Training Process (`train/train_ppo.py`)

#### 4.1 PPO Algorithm

**Simple:** Proximal Policy Optimization - a method that learns gradually, not making huge changes at once.

**Technical:**
- **Policy**: Neural network that decides actions
- **Value Function**: Estimates expected future rewards
- **Clipping**: Prevents policy from changing too fast
- **GAE (Generalized Advantage Estimation)**: Better estimates of how good actions are

**Hyperparameters:**
```python
Learning rate: 3e-4 (how fast to learn)
Batch size: 64 (samples per update)
Epochs: 10 (how many times to update per batch)
Gamma: 0.99 (discount factor - how much to value future rewards)
GAE lambda: 0.95 (bias-variance tradeoff)
Clip range: 0.2 (how much policy can change)
```

#### 4.2 Training Process

**Steps:**
1. Agent interacts with environment for 2048 steps
2. Collects experiences (state, action, reward)
3. Computes advantages (how good each action was)
4. Updates policy network 10 times on collected data
5. Repeats until 50,000 total steps

**Callbacks:**
- **CheckpointCallback**: Saves model every 10,000 steps
- **SaveBestModelCallback**: Evaluates every 5,000 steps, saves best model

### 5. Evaluation (`evaluation/evaluate.py`)

#### 5.1 Baseline Controller

**Simple:** A rule-based controller (like simple if-then rules) for comparison.

**Rules:**
```python
if 18 <= hour <= 22 and soc > 0.5:
    discharge  # Use battery during expensive peak hours

if 10 <= hour <= 16 and pv > 4.0 and soc < 0.9:
    charge  # Store solar energy during midday

if pv > 5.0 and comfort > 0.7:
    shift_load  # Shift load when lots of solar available

else:
    do_nothing  # Default action
```

#### 5.2 Evaluation Metrics

**Why No Accuracy?**
- This is **Reinforcement Learning**, not classification
- No labeled dataset (no "correct" answers)
- Agent learns by trial and error
- We measure **performance**, not accuracy

**Metrics:**

1. **Mean Reward**
   - Higher is better
   - Combines all objectives into single score
   - Shows overall policy quality

2. **Peak Demand (kW)**
   - Lower is better
   - Maximum grid import during 24 hours
   - Target: Keep below 6.0 kW threshold

3. **Total Cost ($)**
   - Lower is better
   - Sum of (grid_import × price) over 24 hours
   - Measures economic efficiency

4. **Minimum Comfort**
   - Higher is better (must be > 0.60)
   - Lowest comfort during episode
   - Hard constraint violation if < 0.60

### 6. Project Structure

```
MO-RL-PeakShaving/
├── env/                    # Environment components
│   ├── smart_grid_env.py   # Main Gymnasium environment
│   ├── pv_model.py         # Solar generation model
│   ├── load_model.py       # Electricity demand model
│   └── price_model.py      # TOU pricing model
│
├── train/                  # Training scripts
│   ├── train_ppo.py        # Main training script
│   └── callbacks.py        # Training callbacks
│
├── evaluation/             # Evaluation scripts
│   ├── evaluate.py         # Main evaluation script
│   ├── baseline_controller.py  # Rule-based baseline
│   └── plot_results.py     # Visualization generation
│
├── models/                 # Trained models
│   ├── ppo_best_model.zip  # Best performing model
│   └── ppo_final_model.zip # Final trained model
│
├── results/                # Results and outputs
│   ├── logs/               # CSV files with metrics
│   └── plots/              # Visualization images
│
├── config.py               # All configuration parameters
├── app.py                  # Flask web interface
└── templates/ & static/    # Frontend files
```

### 7. How It All Works Together

#### 7.1 Episode Flow (24 Hours)

```
Hour 0:  Reset environment
         - SOC = 50%
         - Comfort = 1.0
         - Get initial PV, load, price

Hour 1-23: For each hour:
           1. Agent observes: [hour, pv, load, price, soc, comfort]
           2. Agent chooses action (0-4)
           3. Environment executes action:
              - Updates battery SOC
              - Updates comfort
              - Calculates grid import
           4. Environment calculates reward
           5. Check constraints (comfort < 0.60?)
           6. Move to next hour

End:      Episode complete or terminated
          Calculate final metrics
```

#### 7.2 Learning Process

```
1. Initialize: Random policy (agent doesn't know what to do)

2. Exploration: Agent tries random actions, collects experiences

3. Learning: 
   - Analyzes which actions led to better rewards
   - Updates policy to favor good actions
   - Gradually improves over time

4. Convergence: Policy converges to optimal strategy
   - Charge battery during midday (solar available)
   - Discharge during peak hours (expensive)
   - Balance comfort and cost
```

### 8. Technical Implementation Details

#### 8.1 State Transitions

**Battery SOC Update:**
```python
# Discharge
if action == "discharge":
    discharge_power = min(max_discharge, available_energy)
    energy_discharged = discharge_power * efficiency
    soc_new = soc_old - (discharge_power / capacity)
    grid_import = max(0, grid_import - energy_discharged)

# Charge
if action == "charge":
    available_power = max(0, pv - load)  # Excess solar
    charge_power = min(max_charge, available_power, remaining_capacity)
    soc_new = soc_old + (charge_power / capacity)
    if pv < load:
        grid_import += charge_power  # Charging from grid
```

**Grid Import Calculation:**
```python
# Base grid import (before battery actions)
base_import = max(0, load - pv)

# After battery discharge
grid_import = max(0, base_import - battery_discharge)

# After battery charge (if from grid)
if charging_from_grid:
    grid_import = base_import + charge_power
```

#### 8.2 Reward Calculation Example

**Scenario:** Hour 19 (peak hour), grid_import = 7.0 kW, comfort = 0.75

```python
reward = 0.0

# 1. Peak demand penalty (adaptive weight)
if 7.0 > 6.0:  # Exceeds threshold
    peak_weight = 1.0 * 1.5  # Peak hour multiplier
    peak_penalty = -10.0 * 1.5 * 2.0  # Base * weight * peak multiplier
    reward += -30.0

# 2. Grid cost
cost = 7.0 * 0.25 * 1.0  # import * price * weight
reward -= 1.75

# 3. Comfort violation (adaptive weight)
if 0.75 < 0.80:  # Below threshold
    comfort_weight = 1.0 * 2.0  # Low comfort multiplier
    comfort_penalty = -5.0 * 2.0 * 3.0  # Base * weight * low multiplier
    reward += -30.0

Total reward = -30.0 - 1.75 - 30.0 = -61.75
```

### 9. Why This Approach Works

#### 9.1 Multi-Objective Optimization

**Challenge:** Balance conflicting objectives
- Reduce peak demand (discharge battery)
- Minimize cost (use cheap electricity)
- Maintain comfort (don't sacrifice too much)

**Solution:** Adaptive reward weights
- During peak hours: Weight peak reduction more
- When comfort low: Weight comfort more
- Agent learns to adapt strategy based on context

#### 9.2 Reinforcement Learning Advantages

**vs Rule-Based:**
- ✅ Learns optimal policy automatically
- ✅ Adapts to different scenarios
- ✅ Can find non-obvious strategies
- ✅ Handles complex trade-offs

**vs Supervised Learning:**
- ✅ No labeled dataset needed
- ✅ Learns from environment interaction
- ✅ Can improve over time
- ✅ Handles sequential decision-making

### 10. Results Interpretation

#### 10.1 What Good Performance Looks Like

**RL Agent (Trained):**
- Mean Reward: -28.95 (higher is better)
- Peak Demand: 6.44 kW (below 6.0 threshold is ideal)
- Total Cost: $6.95 (lower is better)
- Min Comfort: 1.000 (maintains full comfort)

**Baseline (Rule-based):**
- Mean Reward: -102.58 (much worse)
- Peak Demand: 6.68 kW (slightly worse)
- Total Cost: $7.08 (slightly worse)
- Min Comfort: 0.725 (comfort sacrificed)

**Interpretation:**
- RL agent learned to balance all objectives better
- Maintains comfort while reducing costs
- Better peak shaving performance
- Overall 73 points better reward

#### 10.2 Learning Progress

**Early Training (Episodes 1-100):**
- Reward: -200 to -150 (agent exploring, making mistakes)
- Strategy: Random actions, learning basics

**Mid Training (Episodes 100-500):**
- Reward: -100 to -50 (agent improving)
- Strategy: Starting to charge during day, discharge during peak

**Late Training (Episodes 500+):**
- Reward: -40 to -20 (agent optimized)
- Strategy: Sophisticated balance of all objectives

### 11. Key Technical Concepts

#### 11.1 Gymnasium Environment

**What:** Standard interface for RL environments

**Why:** 
- Compatibility with RL libraries (stable-baselines3)
- Standardized API (reset, step, observation, action spaces)
- Easy to test and debug

**Implementation:**
```python
class SmartGridEnv(gym.Env):
    def reset() -> observation, info
    def step(action) -> observation, reward, done, truncated, info
    observation_space: Box([hour, pv, load, price, soc, comfort])
    action_space: Discrete(5)
```

#### 11.2 Vectorized Environment

**What:** Wraps environment for parallel processing

**Why:**
- Faster training (can run multiple environments)
- Better sample efficiency
- Required by stable-baselines3

**Implementation:**
```python
env = DummyVecEnv([lambda: SmartGridEnv()])
# Now actions/observations are arrays
```

#### 11.3 Monitor Wrapper

**What:** Logs training statistics

**Why:**
- Track learning progress
- Save episode rewards/lengths
- Generate training curves

**Output:**
- `monitor.csv`: Episode rewards, lengths, timestamps

### 12. Real-World Applications

#### 12.1 Where This Could Be Used

1. **Residential Smart Homes**
   - Home energy management systems
   - Integration with smart thermostats
   - Automated load shifting

2. **Commercial Buildings**
   - Office building energy optimization
   - Peak demand management
   - Cost reduction strategies

3. **Microgrids**
   - Community energy systems
   - Renewable energy integration
   - Grid stability

4. **Electric Vehicle Charging**
   - Optimal charging schedules
   - Grid-friendly charging
   - Cost minimization

#### 12.2 Extensions Possible

- **Multiple Buildings**: Scale to neighborhood level
- **Weather Forecasts**: Include predictions for better planning
- **Demand Response**: Participate in utility programs
- **Electric Vehicles**: Include EV charging/discharging
- **Thermal Storage**: Add heating/cooling storage
- **Real-time Pricing**: Dynamic pricing instead of TOU

### 13. Summary

**In Simple Terms:**
An AI learns to manage your home's electricity use - charging batteries when solar is available, using batteries during expensive times, and keeping you comfortable - all automatically!

**Technical Summary:**
A PPO-based reinforcement learning agent trained on a Gymnasium environment that simulates a solar-integrated smart grid. The agent learns a policy to minimize peak demand and costs while maintaining consumer comfort through adaptive multi-objective reward shaping. Evaluation shows the RL agent outperforms rule-based baselines across all metrics.

**Key Achievement:**
Successfully balances three conflicting objectives (peak shaving, cost minimization, comfort maintenance) using adaptive reward weights, demonstrating the power of RL for complex energy management problems.

