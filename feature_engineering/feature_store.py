import hopsworks
import pandas as pd
import os
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    """Hopsworks feature storage with exact type matching"""

    def __init__(self):
        # Disable Kafka
        os.environ['ENABLE_HOPSWORKS_KAFKA'] = '0'

        self.api_key = Config.HOPSWORKS_API_KEY
        if not self.api_key:
            raise ValueError("HOPSWORKS_API_KEY not set")

        try:
            self.project = hopsworks.login(api_key_value=self.api_key)
            self.fs = self.project.get_feature_store()
            logger.info("Connected to Hopsworks Feature Store")
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            raise

    def store_features(self, features: List[Dict]):
        """Store features with exact type matching"""
        try:
            df = self._prepare_data(features)
            self._store_to_hopsworks(df)
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            raise

    def get_feature_view(self, name: str, version: int = 1):
        """Fetch a feature view by its name."""
        try:
            feature_view = self.fs.get_feature_view(name=name, version=version)
            if feature_view is None:
                logger.error(f"Feature view '{name}' with version {version} not found")
                return None
            logger.info(f"Successfully fetched feature view: {name}, version: {version}")
            return feature_view
        except Exception as e:
            logger.error(f"Failed to fetch feature view '{name}': {str(e)}")
            raise

    def get_training_data(self, feature_view_name: str, target_cols: List[str]):
        """Fetch training data from the feature store by feature view name"""
        try:
            # Retrieve feature view by name
            feature_view = self.fs.get_feature_view(name=feature_view_name)

            # Fetch training data as a DataFrame
            df = feature_view.get_batch_data()

            # Filter columns based on target_cols and return the data
            X = df.drop(columns=target_cols)
            y = df[target_cols]

            logger.info(f"Successfully fetched {len(df)} records for training")
            return X, y
        except Exception as e:
            logger.error(f"Failed to fetch training data: {str(e)}")
            raise

    def _prepare_data(self, features: List[Dict]) -> pd.DataFrame:
        """Convert data types to match Hopsworks schema exactly"""
        df = pd.DataFrame(features)

        # Convert timestamp to milliseconds (bigint)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6

        # Exact type mapping based on error message
        type_mapping = {
            # Float columns (32-bit)
            'lat': ('float32', np.nan),
            'lon': ('float32', np.nan),
            'o3': ('float32', np.nan),
            'no2': ('float32', np.nan),
            'so2': ('float32', np.nan),

            # Integer columns (32-bit)
            'aqi': ('int32', -1),
            'pm25': ('int32', -1),
            'pm10': ('int32', -1),
            'co': ('int32', -1),
            'temp': ('int32', -1),
            'feels_like': ('int32', -1),
            'pressure': ('int32', -1),
            'humidity': ('int32', -1),
            'wind_speed': ('int32', -1),
            'wind_deg': ('int32', -1),
            'clouds': ('int32', -1),
            'weather_id': ('int32', -1),

            # String columns
            'city': ('str', '')
        }

        # Apply type conversion
        for col, (dtype, fill_val) in type_mapping.items():
            if col in df.columns:
                try:
                    if dtype == 'str':
                        df[col] = df[col].fillna(fill_val).astype(dtype)
                    elif 'float' in dtype:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                        df[col] = df[col].fillna(fill_val)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_val).astype(dtype)
                except Exception as e:
                    logger.warning(f"Type conversion failed for {col}: {str(e)}")
                    df[col] = fill_val

        return df

    def _store_to_hopsworks(self, df: pd.DataFrame):
        """Create or update feature group with exact schema"""
        try:
            # Get or create feature group with explicit schema
            fg = self.fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp',
                online_enabled=False,
                statistics_config={"enabled": False}
            )

            # Insert data
            fg.insert(df)
            logger.info(f"Successfully stored {len(df)} records")

        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Test data with all fields
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
        'co': 3.9,
        'temp': None,
        'feels_like': None,
        'pressure': None,
        'humidity': None,
        'wind_speed': None,
        'wind_deg': None,
        'clouds': None,
        'weather_id': None
    }]

    try:
        store = FeatureStore()
        store.store_features(test_data)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
