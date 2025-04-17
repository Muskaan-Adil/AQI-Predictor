import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """Class for generating features from raw data."""
    
    def __init__(self):
        """Initialize the feature generator."""
        pass
    
    def extract_time_features(self, df, timestamp_col='timestamp'):
        """Extract time-based features from timestamp.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            timestamp_col (str, optional): Timestamp column name. Defaults to 'timestamp'.
            
        Returns:
            pd.DataFrame: DataFrame with time features added.
        """
        result_df = df.copy()
        
        if timestamp_col in result_df.columns:
            result_df[timestamp_col] = pd.to_datetime(result_df[timestamp_col])
            
            result_df['hour'] = result_df[timestamp_col].dt.hour
            result_df['day'] = result_df[timestamp_col].dt.day
            result_df['month'] = result_df[timestamp_col].dt.month
            result_df['year'] = result_df[timestamp_col].dt.year
            result_df['dayofweek'] = result_df[timestamp_col].dt.dayofweek
            result_df['is_weekend'] = result_df['dayofweek'].isin([5, 6]).astype(int)
            result_df['quarter'] = result_df[timestamp_col].dt.quarter
            result_df['dayofyear'] = result_df[timestamp_col].dt.dayofyear
            result_df['weekofyear'] = result_df[timestamp_col].dt.isocalendar().week
        
        return result_df
    
    def calculate_rolling_features(self, df, target_col, window_sizes=None):
        """Calculate rolling window features (e.g., averages, min, max).
        
        Args:
            df (pd.DataFrame): Input DataFrame (should be sorted by time).
            target_col (str): Target column for rolling calculations.
            window_sizes (list, optional): List of window sizes. Defaults to [3, 6, 12, 24].
            
        Returns:
            pd.DataFrame: DataFrame with rolling features added.
        """
        if window_sizes is None:
            window_sizes = [3, 6, 12, 24]
            
        result_df = df.copy()
        
        if target_col not in result_df.columns:
            logger.warning(f"Target column {target_col} not found in DataFrame")
            return result_df
        
        if 'city' in result_df.columns:
            grouped = result_df.groupby('city')
            
            for window in window_sizes:
                result_df[f'{target_col}_rolling_mean_{window}'] = grouped[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                result_df[f'{target_col}_rolling_max_{window}'] = grouped[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max()
                )
                
                result_df[f'{target_col}_rolling_min_{window}'] = grouped[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min()
                )
                
                result_df[f'{target_col}_rolling_std_{window}'] = grouped[target_col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        else:
            for window in window_sizes:
                result_df[f'{target_col}_rolling_mean_{window}'] = result_df[target_col].rolling(
                    window=window, min_periods=1
                ).mean()
                
                result_df[f'{target_col}_rolling_max_{window}'] = result_df[target_col].rolling(
                    window=window, min_periods=1
                ).max()
                
                result_df[f'{target_col}_rolling_min_{window}'] = result_df[target_col].rolling(
                    window=window, min_periods=1
                ).min()
                
                result_df[f'{target_col}_rolling_std_{window}'] = result_df[target_col].rolling(
                    window=window, min_periods=1
                ).std()
        
        return result_df
    
    def calculate_lag_features(self, df, target_col, lag_values=None):
        """Calculate lag features for a target column.
        
        Args:
            df (pd.DataFrame): Input DataFrame (should be sorted by time).
            target_col (str): Target column for lag calculations.
            lag_values (list, optional): List of lag values. Defaults to [1, 3, 6, 12, 24, 48].
            
        Returns:
            pd.DataFrame: DataFrame with lag features added.
        """
        if lag_values is None:
            lag_values = [1, 3, 6, 12, 24, 48]
            
        result_df = df.copy()
        
        if target_col not in result_df.columns:
            logger.warning(f"Target column {target_col} not found in DataFrame")
            return result_df
        
        if 'city' in result_df.columns:
            for lag in lag_values:
                result_df[f'{target_col}_lag_{lag}'] = result_df.groupby('city')[target_col].shift(lag)
        else:
            for lag in lag_values:
                result_df[f'{target_col}_lag_{lag}'] = result_df[target_col].shift(lag)
        
        return result_df
    
    def calculate_change_rate(self, df, target_col):
        """Calculate rate of change features.
        
        Args:
            df (pd.DataFrame): Input DataFrame (should be sorted by time).
            target_col (str): Target column for change rate calculations.
            
        Returns:
            pd.DataFrame: DataFrame with change rate features added.
        """
        result_df = df.copy()
        
        if target_col not in result_df.columns:
            logger.warning(f"Target column {target_col} not found in DataFrame")
            return result_df
        
        if 'city' in result_df.columns:
            result_df[f'{target_col}_pct_change'] = result_df.groupby('city')[target_col].pct_change()
            
            result_df[f'{target_col}_diff'] = result_df.groupby('city')[target_col].diff()
            
            result_df[f'{target_col}_change_rate_24h'] = result_df.groupby('city')[target_col].transform(
                lambda x: x.diff(24) / 24
            )
        else:
            result_df[f'{target_col}_pct_change'] = result_df[target_col].pct_change()
            
            result_df[f'{target_col}_diff'] = result_df[target_col].diff()
            
            result_df[f'{target_col}_change_rate_24h'] = result_df[target_col].diff(24) / 24
        
        return result_df
    
    def create_weather_interaction_features(self, df):
        """Create interaction features between weather variables.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with interaction features added.
        """
        result_df = df.copy()
        
        weather_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds', 'rain']
        existing_cols = [col for col in weather_cols if col in result_df.columns]
        
        if len(existing_cols) < 2:
            logger.warning("Not enough weather columns to create interaction features")
            return result_df
        
        for i in range(len(existing_cols)):
            for j in range(i+1, len(existing_cols)):
                col1 = existing_cols[i]
                col2 = existing_cols[j]
                
                result_df[f'{col1}_{col2}_product'] = result_df[col1] * result_df[col2]
                
                result_df[f'{col1}_{col2}_ratio'] = result_df[col1] / result_df[col2].replace(0, np.nan)
        
        return result_df
    
    def generate_all_features(self, df, target_cols=None):
        """Generate all features from raw data.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            target_cols (list, optional): List of target columns. Defaults to ['pm25', 'pm10'].
            
        Returns:
            pd.DataFrame: DataFrame with all features added.
        """
        if target_cols is None:
            target_cols = ['pm25', 'pm10']
        
        result_df = df.copy()
        
        result_df = self.extract_time_features(result_df)
        
        result_df = self.create_weather_interaction_features(result_df)
        
        for target_col in target_cols:
            if target_col in result_df.columns:
                result_df = self.calculate_rolling_features(result_df, target_col)
                
                result_df = self.calculate_lag_features(result_df, target_col)
                
                result_df = self.calculate_change_rate(result_df, target_col)
        
        result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return result_df
