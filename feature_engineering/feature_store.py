import hopsworks
import pandas as pd
import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    """Hopsworks feature storage with proper feature view handling"""

    def __init__(self):
        # Disable Kafka
        os.environ['ENABLE_HOPSWORKS_KAFKA'] = '0'

        self.api_key = Config.HOPSWORKS_API_KEY
        if not self.api_key:
            raise ValueError("HOPSWORKS_API_KEY not set")

        try:
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project="AQI_Pred_10Pearls"  # Your project name
            )
            self.fs = self.project.get_feature_store()
            logger.info("Connected to Hopsworks Feature Store")
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            raise

    def store_features(self, features: List[Dict]) -> None:
        """Store features with proper type conversion"""
        try:
            df = self._prepare_data(features)
            self._store_to_hopsworks(df)
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            raise

    def get_training_data(self, feature_view_name: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data from feature view with proper tuple handling"""
        try:
            feature_view = self.get_feature_view(feature_view_name)
            
            # Get training data (returns tuple: (X, y, metadata))
            training_data = feature_view.get_training_data(
                training_dataset_version=1,
                read_options={"use_hive": True}  # Ensure DataFrame format
            )
            
            # Unpack tuple correctly
            X = training_data[0]  # Features DataFrame
            y = training_data[1][target_col]  # Target Series
            
            logger.info(f"Retrieved {X.shape[0]} samples with {X.shape[1]} features")
            return X, y
        except Exception as e:
            logger.error(f"Training data error: {str(e)}")
            raise

    def get_feature_view(self, name: str, version: int = 1):
        """Get or create feature view with proper initialization"""
        try:
            # Try to get existing feature view
            feature_view = self.fs.get_feature_view(
                name=name,
                version=version
            )
            
            if feature_view is None:
                logger.info(f"Creating new feature view: {name}")
                feature_view = self._create_feature_view(name, version)
                
            return feature_view
        except Exception as e:
            logger.error(f"Failed to get feature view: {str(e)}")
            raise

    def _create_feature_view(self, name: str, version: int):
        """Create feature view from feature group with proper query"""
        try:
            fg = self.fs.get_feature_group(
                name="karachi_aqi_realtime",
                version=1
            )
            
            # Explicitly exclude target columns from features
            features = [feat.name for feat in fg.features 
                       if feat.name not in ["pm25", "pm10"]]
            
            # Create query with feature selection
            query = fg.select(features)
            
            # Create feature view
            feature_view = self.fs.create_feature_view(
                name=name,
                version=version,
                query=query,
                labels=["pm25", "pm10"]  # Target columns
            )
            
            # Create training dataset
            feature_view.create_training_data(
                description="Training dataset for AQI prediction",
                data_format="csv",
                training_dataset_version=1,
                write_options={"wait_for_job": True}
            )
            
            logger.info(f"Created feature view {name} v{version}")
            return feature_view
        except Exception as e:
            logger.error(f"Feature view creation failed: {str(e)}")
            raise

    def _prepare_data(self, features: List[Dict]) -> pd.DataFrame:
        """Convert data types to match Hopsworks schema"""
        df = pd.DataFrame(features)

        # Convert timestamp to milliseconds
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6

        # Type mapping
        type_mapping = {
            'lat': 'float32',
            'lon': 'float32',
            'o3': 'float32',
            'no2': 'float32',
            'so2': 'float32',
            'aqi': 'int32',
            'pm25': 'int32',
            'pm10': 'int32',
            'co': 'int32',
            'city': 'str'
        }

        # Apply type conversion
        for col, dtype in type_mapping.items():
            if col in df.columns:
                try:
                    if dtype == 'str':
                        df[col] = df[col].fillna('').astype(dtype)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
                except Exception as e:
                    logger.warning(f"Type conversion failed for {col}: {str(e)}")
                    df[col] = 0 if dtype != 'str' else ''

        return df

    def _store_to_hopsworks(self, df: pd.DataFrame) -> None:
        """Store data to feature group"""
        try:
            fg = self.fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp',
                online_enabled=False,
                statistics_config={"enabled": False}
            )
            
            fg.insert(df, overwrite=False)
            logger.info(f"Stored {len(df)} records to feature group")
        except Exception as e:
            logger.error(f"Failed to store data: {str(e)}")
            raise

if __name__ == "__main__":
    # Test data
    test_data = [{
        'city': 'Karachi',
        'lat': 24.8607,
        'lon': 67.0011,
        'timestamp': datetime.now(),
        'aqi': 152,
        'pm25': 152,
        'pm10': 79,
        'o3': 21.2,
        'no2': 3.1,
        'so2': 1.5,
        'co': 3.9
    }]

    try:
        store = FeatureStore()
        
        # Test storage
        store.store_features(test_data)
        
        # Test training data retrieval
        X, y = store.get_training_data(
            feature_view_name="karachi_aqi_features",
            target_col="pm25"
        )
        print(f"Retrieved training data - Features: {X.shape}, Target: {y.shape}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
