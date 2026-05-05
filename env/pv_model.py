"""
PV (Photovoltaic) Generation Model
Models stochastic solar generation with Gaussian noise on midday peak curve
"""

import numpy as np
import config


class PVModel:
    """Models PV generation with stochastic behavior"""
    
    def __init__(self, peak_power=None, noise_std=None):
        """
        Initialize PV model
        
        Args:
            peak_power: Peak generation power in kW (default from config)
            noise_std: Standard deviation for Gaussian noise (default from config)
        """
        self.peak_power = peak_power or config.PV_PEAK_POWER
        self.noise_std = noise_std or config.PV_NOISE_STD
        
    def get_generation(self, hour):
        """
        Get PV generation for a given hour
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            PV generation in kW (non-negative)
        """
        # Base PV curve: peaks at midday (hour 12), follows sine-like pattern
        # Normalized to [0, 1] range
        normalized_hour = (hour - 6) / 12.0  # Shift to start at 6 AM
        normalized_hour = np.clip(normalized_hour, 0, 1)
        
        # Sine curve for smooth generation pattern (0 at 6 AM, peak at 12 PM, 0 at 6 PM)
        if 6 <= hour <= 18:
            base_generation = np.sin(np.pi * normalized_hour)
        else:
            base_generation = 0.0
        
        # Scale to peak power
        base_power = base_generation * self.peak_power
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std)
        generation = base_power + noise
        
        # Ensure non-negative
        generation = max(0.0, generation)
        
        return generation
    
    def get_deterministic_generation(self, hour):
        """
        Get deterministic PV generation (without noise) for testing
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            PV generation in kW (non-negative)
        """
        normalized_hour = (hour - 6) / 12.0
        normalized_hour = np.clip(normalized_hour, 0, 1)
        
        if 6 <= hour <= 18:
            base_generation = np.sin(np.pi * normalized_hour)
        else:
            base_generation = 0.0
        
        base_power = base_generation * self.peak_power
        return max(0.0, base_power)

