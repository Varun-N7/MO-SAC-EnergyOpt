"""
Residential Load Model
Models load curve with morning and evening peaks
"""

import numpy as np
import config


class LoadModel:
    """Models residential load consumption with peak patterns"""
    
    def __init__(self, base_min=None, base_max=None, 
                 morning_peak_hour=None, evening_peak_hour=None,
                 morning_mult=None, evening_mult=None,
                 flexible_fraction=None):
        """
        Initialize load model
        
        Args:
            base_min: Minimum base load in kW (default from config)
            base_max: Maximum base load in kW (default from config)
            morning_peak_hour: Hour of morning peak (default from config)
            evening_peak_hour: Hour of evening peak (default from config)
            morning_mult: Morning peak multiplier (default from config)
            evening_mult: Evening peak multiplier (default from config)
            flexible_fraction: Fraction of load that can be shifted (default from config)
        """
        self.base_min = base_min or config.BASE_LOAD_MIN
        self.base_max = base_max or config.BASE_LOAD_MAX
        self.morning_peak_hour = morning_peak_hour or config.MORNING_PEAK_HOUR
        self.evening_peak_hour = evening_peak_hour or config.EVENING_PEAK_HOUR
        self.morning_mult = morning_mult or config.MORNING_PEAK_MULTIPLIER
        self.evening_mult = evening_mult or config.EVENING_PEAK_MULTIPLIER
        self.flexible_fraction = flexible_fraction or config.FLEXIBLE_LOAD_FRACTION
        
    def get_load(self, hour, shifted=False):
        """
        Get load consumption for a given hour
        
        Args:
            hour: Hour of day (0-23)
            shifted: Whether flexible load has been shifted (reduces current load)
            
        Returns:
            Load consumption in kW
        """
        # Base load with some variation
        base_load = np.random.uniform(self.base_min, self.base_max)
        
        # Morning peak (around specified hour)
        morning_effect = 0.0
        if abs(hour - self.morning_peak_hour) <= 2:
            distance = abs(hour - self.morning_peak_hour)
            morning_effect = self.morning_mult * np.exp(-distance**2 / 2.0)
        
        # Evening peak (around specified hour)
        evening_effect = 0.0
        if abs(hour - self.evening_peak_hour) <= 3:
            distance = abs(hour - self.evening_peak_hour)
            evening_effect = self.evening_mult * np.exp(-distance**2 / 2.0)
        
        # Combine effects
        peak_multiplier = 1.0 + morning_effect + evening_effect
        total_load = base_load * peak_multiplier
        
        # If load is shifted, reduce by flexible fraction
        if shifted:
            total_load *= (1.0 - self.flexible_fraction)
        
        return max(0.0, total_load)
    
    def get_flexible_load_amount(self, hour):
        """
        Get the amount of flexible load available for shifting
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Flexible load amount in kW
        """
        base_load = (self.base_min + self.base_max) / 2.0
        
        # Morning peak effect
        morning_effect = 0.0
        if abs(hour - self.morning_peak_hour) <= 2:
            distance = abs(hour - self.morning_peak_hour)
            morning_effect = self.morning_mult * np.exp(-distance**2 / 2.0)
        
        # Evening peak effect
        evening_effect = 0.0
        if abs(hour - self.evening_peak_hour) <= 3:
            distance = abs(hour - self.evening_peak_hour)
            evening_effect = self.evening_mult * np.exp(-distance**2 / 2.0)
        
        peak_multiplier = 1.0 + morning_effect + evening_effect
        total_load = base_load * peak_multiplier
        
        return total_load * self.flexible_fraction

