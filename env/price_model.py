"""
Time of Use (TOU) Electricity Price Model
"""

import config


class PriceModel:
    """Models TOU electricity pricing"""
    
    def __init__(self, off_peak=None, mid_peak=None, on_peak=None, peak_hours=None):
        """
        Initialize price model
        
        Args:
            off_peak: Off-peak price in $/kWh (default from config)
            mid_peak: Mid-peak price in $/kWh (default from config)
            on_peak: On-peak price in $/kWh (default from config)
            peak_hours: List of peak pricing hours (default from config)
        """
        self.off_peak = off_peak or config.PRICE_OFF_PEAK
        self.mid_peak = mid_peak or config.PRICE_MID_PEAK
        self.on_peak = on_peak or config.PRICE_ON_PEAK
        self.peak_hours = peak_hours or config.PEAK_HOURS
        
    def get_price(self, hour):
        """
        Get electricity price for a given hour
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Price in $/kWh
        """
        if hour in self.peak_hours:
            return self.on_peak
        elif 6 <= hour < 22:  # Daytime hours (excluding peak)
            return self.mid_peak
        else:  # Night hours
            return self.off_peak
    
    def is_peak_hour(self, hour):
        """
        Check if hour is a peak pricing hour
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            True if peak hour, False otherwise
        """
        return hour in self.peak_hours

