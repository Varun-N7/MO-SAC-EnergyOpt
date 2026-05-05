"""
Smart Grid Environment for Peak Load Shaving
Gymnasium-compatible environment with multi-objective rewards
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config
from env.pv_model import PVModel
from env.load_model import LoadModel
from env.price_model import PriceModel
from env.real_data_loader import RealDataLoader


class SmartGridEnv(gym.Env):
    """
    Smart Grid Environment for peak load shaving with comfort constraints
    
    Observation: [hour, pv, load, price, soc, comfort, battery_health]
    Actions: 0=do nothing, 1=discharge, 2=charge, 3=shift load, 4=comfort sacrifice
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, render_mode=None, use_real_data=False, dataset_path=None, reward_weights=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.use_real_data = use_real_data
        self.reward_weights = {
            "peak": config.REWARD_WEIGHT_PEAK,
            "cost": config.REWARD_WEIGHT_COST,
            "comfort": config.REWARD_WEIGHT_COMFORT,
            "battery": config.REWARD_WEIGHT_BATTERY,
        }
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)
        
        # Initialize models
        if use_real_data:
            # Use real-world dataset
            self.data_loader = RealDataLoader(dataset_path)
            if self.data_loader.data is None:
                print("Warning: Real data not available, falling back to synthetic models")
                self.use_real_data = False
                self.pv_model = PVModel()
                self.load_model = LoadModel()
                self.price_model = PriceModel()
            else:
                self.pv_model = None
                self.load_model = None
                self.price_model = None
        else:
            # Use synthetic models
            self.pv_model = PVModel()
            self.load_model = LoadModel()
            self.price_model = PriceModel()
            self.data_loader = None
        
        # Observation space: [hour, pv, load, price, soc, comfort, battery_health]
        # Normalized to [0, 1] or appropriate ranges
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, config.BATTERY_MIN_HEALTH], dtype=np.float32),
            high=np.array([23.0, 15.0, 10.0, 0.5, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Action meanings
        self.action_names = [
            "do_nothing",
            "discharge",
            "charge",
            "shift_load",
            "comfort_sacrifice"
        ]
        
        # Reset state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        options = options or {}
        
        # Episode state
        self.hour = 0
        self.soc = config.INITIAL_SOC
        self.comfort = config.INITIAL_COMFORT
        self.nominal_battery_capacity_kwh = config.BATTERY_CAPACITY_KWH
        self.battery_health = config.INITIAL_BATTERY_HEALTH
        self.battery_capacity_kwh = self.nominal_battery_capacity_kwh
        self.cumulative_charge_kwh = 0.0
        self.cumulative_discharge_kwh = 0.0
        self.cycle_count = 0.0
        self.load_shifted = False  # Track if load is currently shifted
        self.episode_terminated = False
        
        # Get initial observations
        if self.use_real_data and self.data_loader is not None:
            # Reset data loader for new episode
            self.data_loader.reset_episode(start_date=options.get("start_date"))
            data = self.data_loader.get_current_data()
            if data:
                self.pv = data['pv']
                self.base_load = data['load']
                self.price = data['price']
                self.hour = data['hour']
            else:
                # Fallback to synthetic
                self.pv = self.pv_model.get_generation(self.hour) if self.pv_model else 0.0
                self.base_load = self.load_model.get_load(self.hour, shifted=self.load_shifted) if self.load_model else 0.0
                self.price = self.price_model.get_price(self.hour) if self.price_model else 0.1
        else:
            # Use synthetic models
            self.pv = self.pv_model.get_generation(self.hour)
            self.base_load = self.load_model.get_load(self.hour, shifted=self.load_shifted)
            self.price = self.price_model.get_price(self.hour)
        
        # Calculate grid import (load - pv, before battery actions)
        self.grid_import = max(0.0, self.base_load - self.pv)
        
        # Episode history for tracking
        self.episode_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.episode_terminated:
            # If already terminated, return zero reward and same state
            observation = self._get_observation()
            return observation, 0.0, True, False, self._get_info()
        
        # Store pre-action state for history
        pre_action_state = {
            'hour': self.hour,
            'pv': self.pv,
            'base_load': self.base_load,
            'soc': self.soc,
            'comfort': self.comfort,
            'battery_health': self.battery_health,
            'cycle_count': self.cycle_count,
            'action': action
        }
        
        # Execute action
        self._execute_action(action)
        
        # Update hour
        self.hour += 1
        
        # Check if episode is done
        if self.use_real_data and self.data_loader is not None:
            done = self.data_loader.is_episode_complete() or self.episode_terminated
            if not done:
                self.data_loader.advance_hour()
                data = self.data_loader.get_current_data()
                if data:
                    self.pv = data['pv']
                    self.base_load = data['load']
                    self.price = data['price']
                    self.hour = data['hour']
                else:
                    done = True
                    self.pv = 0.0
                    self.base_load = 0.0
            else:
                self.pv = 0.0
                self.base_load = 0.0
        else:
            done = (self.hour >= config.EPISODE_LENGTH) or self.episode_terminated
            # Get new observations
            self.pv = self.pv_model.get_generation(self.hour) if not done else 0.0
            self.base_load = self.load_model.get_load(self.hour, shifted=self.load_shifted) if not done else 0.0
            self.price = self.price_model.get_price(self.hour) if not done else self.price_model.get_price(self.hour - 1)
        
        # Calculate grid import after battery actions
        self.grid_import = max(0.0, self.base_load - self.pv)
        
        # Update comfort (recovery if no action taken)
        if not done:
            if action == 0:  # Do nothing - comfort recovers slightly
                self.comfort = min(1.0, self.comfort + config.COMFORT_RECOVERY_RATE)
            # Load shift effect persists for one hour
            if action != 3:
                self.load_shifted = False
        
        # Check hard constraint
        if self.comfort < config.COMFORT_MIN_THRESHOLD:
            self.episode_terminated = True
            done = True
        
        # Calculate reward
        reward = self._calculate_reward(pre_action_state, action)
        
        # Store in history
        self.episode_history.append({
            'hour': pre_action_state['hour'],
            'pv': pre_action_state['pv'],
            'base_load': pre_action_state['base_load'],
            'grid_import': self.grid_import,
            'soc': pre_action_state['soc'],
            'comfort': pre_action_state['comfort'],
            'battery_health': pre_action_state['battery_health'],
            'cycle_count': pre_action_state['cycle_count'],
            'action': action,
            'reward': reward
        })
        
        observation = self._get_observation()
        info = self._get_info()
        info['action_name'] = self.action_names[action]
        
        return observation, reward, done, False, info
    
    def _execute_action(self, action):
        """Execute the given action"""
        charge_energy_kwh = 0.0
        discharge_energy_kwh = 0.0

        if action == 0:  # Do nothing
            pass
        
        elif action == 1:  # Discharge battery
            discharge_power = min(
                config.BATTERY_MAX_DISCHARGE_POWER,
                self.soc * self.battery_capacity_kwh  # Can't discharge more than available
            )
            energy_discharged = discharge_power * config.BATTERY_DISCHARGE_EFFICIENCY
            self.soc = max(0.0, self.soc - (discharge_power / self.battery_capacity_kwh))
            # Reduce grid import by discharged energy
            self.grid_import = max(0.0, self.grid_import - energy_discharged)
            discharge_energy_kwh = max(0.0, discharge_power)
        
        elif action == 2:  # Charge battery
            # Available power for charging (excess PV or grid)
            available_power = max(0.0, self.pv - self.base_load)
            charge_power = min(
                config.BATTERY_MAX_CHARGE_POWER,
                available_power,
                (1.0 - self.soc) * self.battery_capacity_kwh  # Can't charge beyond capacity
            )
            self.soc = min(1.0, self.soc + (charge_power / self.battery_capacity_kwh))
            # Increase grid import if charging from grid (when PV < load)
            if self.pv < self.base_load:
                self.grid_import += charge_power / config.BATTERY_CHARGE_EFFICIENCY
            charge_energy_kwh = max(0.0, charge_power)
        
        elif action == 3:  # Shift flexible load
            self.load_shifted = True
            self.comfort = max(0.0, self.comfort - config.COMFORT_DECAY_SHIFT)
            # Reduce current load
            if self.use_real_data:
                # For real data, assume 15% of load is flexible
                flexible_amount = self.base_load * config.FLEXIBLE_LOAD_FRACTION
            else:
                flexible_amount = self.load_model.get_flexible_load_amount(self.hour)
            self.base_load = max(0.0, self.base_load - flexible_amount)
            self.grid_import = max(0.0, self.grid_import - flexible_amount)
        
        elif action == 4:  # Comfort sacrifice (HVAC reduction)
            self.comfort = max(0.0, self.comfort - config.COMFORT_DECAY_HVAC)
            # Reduce load by HVAC reduction (assume 20% of base load is HVAC)
            hvac_reduction = self.base_load * config.HVAC_REDUCTION_FRACTION
            self.base_load = max(0.0, self.base_load - hvac_reduction)
            self.grid_import = max(0.0, self.grid_import - hvac_reduction)

        self._update_battery_degradation(charge_energy_kwh, discharge_energy_kwh)

    def _update_battery_degradation(self, charge_energy_kwh, discharge_energy_kwh):
        """Update equivalent cycles, battery health, and effective capacity."""
        self.cumulative_charge_kwh += max(0.0, charge_energy_kwh)
        self.cumulative_discharge_kwh += max(0.0, discharge_energy_kwh)

        throughput_kwh = self.cumulative_charge_kwh + self.cumulative_discharge_kwh
        self.cycle_count = throughput_kwh / (2.0 * self.nominal_battery_capacity_kwh)

        raw_health = 1.0 - (config.BATTERY_DEGRADATION_RATE * self.cycle_count)
        self.battery_health = float(np.clip(raw_health, config.BATTERY_MIN_HEALTH, 1.0))
        self.battery_capacity_kwh = self.nominal_battery_capacity_kwh * self.battery_health
    
    def _calculate_reward(self, pre_state, action):
        """Calculate multi-objective reward with adaptive weights"""
        reward = 0.0
        
        # Hard constraint penalty
        if self.comfort < config.COMFORT_MIN_THRESHOLD:
            reward += config.HARD_CONSTRAINT_PENALTY
            return reward
        
        # 1. Peak demand penalty (adaptive weight during peak hours)
        peak_weight = 1.0
        if self._is_peak_hour(self.hour):
            peak_weight *= config.ADAPTIVE_PEAK_WEIGHT_MULTIPLIER
        
        if self.grid_import > config.PEAK_DEMAND_THRESHOLD:
            peak_penalty = config.PEAK_PENALTY_BASE * peak_weight
            if self._is_peak_hour(self.hour):
                peak_penalty *= config.PEAK_PENALTY_MULTIPLIER
            reward += self.reward_weights["peak"] * peak_penalty
        
        # 2. Grid cost (energy import × price)
        cost = self.grid_import * self.price * config.COST_WEIGHT
        reward -= self.reward_weights["cost"] * cost
        
        # 3. Comfort violation penalty (adaptive weight when comfort is low)
        comfort_weight = 1.0
        if self.comfort < config.ADAPTIVE_COMFORT_THRESHOLD:  # Low comfort threshold
            comfort_weight *= config.ADAPTIVE_COMFORT_WEIGHT_MULTIPLIER
        
        if self.comfort < config.COMFORT_PENALTY_THRESHOLD:  # Comfort violation threshold
            comfort_penalty = config.COMFORT_VIOLATION_PENALTY * comfort_weight
            if self.comfort < config.SEVERE_COMFORT_THRESHOLD:
                comfort_penalty *= config.COMFORT_VIOLATION_MULTIPLIER
            reward += self.reward_weights["comfort"] * comfort_penalty

        # 4. Battery longevity objective (optional lambda_battery via config.BATTERY_WEIGHT)
        soc_swing = abs(self.soc - pre_state["soc"])
        soc_swing_cost = soc_swing * config.SOC_SWING_PENALTY_WEIGHT

        low_soc_cost = 0.0
        if self.soc < config.LOW_SOC_THRESHOLD:
            low_soc_depth = (config.LOW_SOC_THRESHOLD - self.soc) / config.LOW_SOC_THRESHOLD
            low_soc_cost = low_soc_depth * config.LOW_SOC_PENALTY_WEIGHT

        degradation_delta = max(0.0, pre_state["battery_health"] - self.battery_health)
        degradation_cost = degradation_delta * config.DEGRADATION_COST_WEIGHT

        battery_cost = soc_swing_cost + low_soc_cost + degradation_cost
        reward -= self.reward_weights["battery"] * battery_cost
        
        return reward

    def _is_peak_hour(self, hour):
        """Peak-hour utility that works in both synthetic and real-data modes."""
        if self.price_model is not None:
            return self.price_model.is_peak_hour(hour)
        return hour in config.PEAK_HOURS
    
    def _get_observation(self):
        """Get current observation"""
        return np.array([
            float(self.hour),
            self.pv,
            self.base_load,
            self.price,
            self.soc,
            self.comfort,
            self.battery_health,
        ], dtype=np.float32)
    
    def _get_info(self):
        """Get additional info dictionary"""
        return {
            'hour': self.hour,
            'pv': self.pv,
            'load': self.base_load,
            'price': self.price,
            'soc': self.soc,
            'comfort': self.comfort,
            'battery_health': self.battery_health,
            'cycle_count': self.cycle_count,
            'battery_capacity_kwh': self.battery_capacity_kwh,
            'grid_import': self.grid_import,
            'terminated': self.episode_terminated
        }
    
    def render(self):
        """Render the environment state"""
        if self.render_mode == "human":
            print(f"Hour: {self.hour:2d} | PV: {self.pv:5.2f} kW | "
                  f"Load: {self.base_load:5.2f} kW | Price: ${self.price:.3f} | "
                  f"SOC: {self.soc:.2f} | Comfort: {self.comfort:.2f} | "
                  f"Health: {self.battery_health:.3f} | Cycles: {self.cycle_count:.3f} | "
                  f"Grid Import: {self.grid_import:5.2f} kW")

