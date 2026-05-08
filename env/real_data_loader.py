"""
Real-world data loader for the environment
Loads actual PV, load, and price data from CSV files
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import config
from utils.logger import get_logger


class RealDataLoader:
    """
    Loads and manages real-world energy data
    """
    
    def __init__(self, dataset_path=None):
        """
        Initialize data loader
        
        Args:
            dataset_path: Path to CSV file with real-world data
        """
        self.logger = get_logger("env.real_data_loader")
        if dataset_path is None:
            dataset_path = config.REAL_DATASET_PATH
        
        self.dataset_path = dataset_path
        self.data = None
        self.current_index = 0
        self.episode_start_index = 0
        
        if os.path.exists(dataset_path):
            self.load_data()
        else:
            self.logger.warning("Dataset not found at %s", dataset_path)
            self.logger.warning("Run: python data/download_dataset.py to create dataset")
            self.data = None
    
    def load_data(self):
        """Load dataset from CSV"""
        try:
            self.data = pd.read_csv(self.dataset_path)
            
            # Convert datetime if string
            if 'datetime' in self.data.columns:
                if isinstance(self.data['datetime'].iloc[0], str):
                    self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            
            # Ensure required columns exist
            required_cols = ['pv_generation_kw', 'load_kw', 'price_per_kwh']
            for col in required_cols:
                if col not in self.data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Extract hour from datetime if not present
            if 'hour' not in self.data.columns and 'datetime' in self.data.columns:
                self.data['hour'] = pd.to_datetime(self.data['datetime']).dt.hour
            
            self.logger.info("Loaded dataset: %s records", len(self.data))
            self.logger.info(
                "Date range: %s to %s",
                self.data["datetime"].min(),
                self.data["datetime"].max(),
            )
            
        except Exception as e:
            self.logger.exception("Error loading dataset: %s", e)
            self.data = None
    
    def reset_episode(self, start_date=None):
        """
        Reset to start of episode (24 hours)
        
        Args:
            start_date: Specific date to start episode (optional)
        """
        if self.data is None:
            return None
        
        if start_date is None:
            # Random start point (but ensure 24 hours available)
            max_start = len(self.data) - 24
            self.episode_start_index = np.random.randint(0, max(1, max_start))
        else:
            # Find closest date
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            mask = self.data['datetime'] >= start_date
            if mask.any():
                self.episode_start_index = self.data[mask].index[0]
            else:
                self.episode_start_index = 0
        
        self.current_index = self.episode_start_index
    
    def get_current_data(self):
        """
        Get current hour's data
        
        Returns:
            dict with pv, load, price, hour, or None if no data
        """
        if self.data is None or self.current_index >= len(self.data):
            return None
        
        row = self.data.iloc[self.current_index]
        
        # Get hour (from column or extract from datetime)
        if 'hour' in row:
            hour = int(row['hour'])
        elif 'datetime' in row:
            hour = int(pd.to_datetime(row['datetime']).hour)
        else:
            hour = 0
        
        return {
            'pv': float(row['pv_generation_kw']),
            'load': float(row['load_kw']),
            'price': float(row['price_per_kwh']),
            'hour': hour,
            'datetime': row.get('datetime', None)
        }
    
    def advance_hour(self):
        """Move to next hour"""
        if self.data is not None:
            self.current_index += 1
    
    def is_episode_complete(self):
        """Check if 24 hours have passed"""
        if self.data is None:
            return True
        return (self.current_index - self.episode_start_index) >= 24
    
    def get_episode_data(self):
        """Get all data for current episode (24 hours)"""
        if self.data is None:
            return None
        
        end_index = min(self.episode_start_index + 24, len(self.data))
        return self.data.iloc[self.episode_start_index:end_index].copy()
