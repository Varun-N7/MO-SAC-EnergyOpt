"""
Rule-based baseline controller for Smart Grid peak load shaving
Heuristic policy for comparison with RL agent
"""

import numpy as np
import config
from env.smart_grid_env import SmartGridEnv


class BaselineController:
    """
    Rule-based heuristic controller
    
    Policy:
    - Discharge during peak hours (18-22) if SOC is high (>0.5)
    - Charge during midday (10-16) if PV generation is high (>4 kW)
    - Shift load when PV is high and comfort is good (>0.7)
    - Do nothing otherwise
    """
    
    def __init__(self, env):
        """
        Initialize baseline controller
        
        Args:
            env: SmartGridEnv instance
        """
        self.env = env
    
    def predict(self, observation, deterministic=True):
        """
        Predict action based on observation
        
        Args:
            observation: Current observation [hour, pv, load, price, soc, comfort, battery_health]
            deterministic: Whether to use deterministic policy (ignored for baseline)
            
        Returns:
            action: Action to take
        """
        hour = int(observation[0])
        pv = observation[1]
        load = observation[2]
        price = observation[3]
        soc = observation[4]
        comfort = observation[5]
        
        # Rule 1: Discharge during peak hours (18-22) if SOC is high
        if 18 <= hour <= 22 and soc > 0.5:
            return 1  # Discharge
        
        # Rule 2: Charge during midday (10-16) if PV is high
        if 10 <= hour <= 16 and pv > 4.0 and soc < 0.9:
            return 2  # Charge
        
        # Rule 3: Shift load when PV is high and comfort is good
        if pv > 5.0 and comfort > 0.7:
            return 3  # Shift load
        
        # Rule 4: Do nothing (default)
        return 0
    
    def get_action_probabilities(self, observation):
        """
        Get action probabilities (for compatibility)
        Returns deterministic probabilities (one-hot)
        
        Args:
            observation: Current observation
            
        Returns:
            probs: Action probabilities (one-hot)
        """
        action = self.predict(observation)
        probs = np.zeros(5)
        probs[action] = 1.0
        return probs

