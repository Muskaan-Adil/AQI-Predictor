import hopsworks
import pandas as pd
import os
import logging
from datetime import datetime
from typing import List, Dict
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Optimized Hopsworks feature storage with strict type handling"""

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

    def _prepare_data(self, features: List[Dict]) -> pd.DataFrame:
        """Convert data types to match Hopsworks schema exactly"""
        df = pd.DataFrame(features)

        # Convert timestamp to milliseconds (bigint)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6

        # Type conversion mapping
        type_rules = {
            'aqi': ('int64', -1),
            'pm25': ('int64', -1),
            'pm10': ('int64', -1),
            'temp': ('int64', -1),
            'feels_like': ('int64', -1),
            'pressure': ('int64', -1),
            'humidity': ('int64', -1),
            'wind_speed': ('int64', -1),
            'wind_deg': ('int64', -1),
            'clouds': ('int64', -1),
            'weather_id': ('int64', -1),
            'o3': ('float64', -1.0),
            'no2': ('float64', -1.0),
            'so2': ('float64', -1.0),
            'co': ('int64', -1),  # Fix for 'co' schema mismatch
            'city': ('str', ''),
            'weather_main': ('str', ''),  # Fix for weather_main schema mismatch
            'lat': ('float64', -1.0),
            'lon': ('float64', -1.0)
        }

        # Apply type conversion
        for col, (dtype, fill_val) in type_rules.items():
            if col in df.columns:
                try:
                    if dtype == 'str':
                        df[col] = df[col].fillna(fill_val).astype(dtype)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_val).astype(dtype)
                except Exception as e:
                    logger.warning(f"Type conversion failed for {col}: {str(e)}")
                    df[col] = fill_val

        return df

    def _store_to_hopsworks(self, df: pd.DataFrame):
        """Strict schema validation with feature group creation"""
        try:
            fg = self.fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp',
                online_enabled=False,
                statistics_config={"enabled": False}
            )

            fg.insert(df)
            logger.info(f"Successfully stored {len(df)} records")

        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise

if __name__ == "__main__":
    test_data = [{
        'city': 'Karachi',
        'lat': 24.8607,
        'lon': 67.0011,
        'timestamp': datetime.now(),
        'aqi': 154,
        'pm25': 154,
        'pm10': 81,
        'o3': 18.6,
        'no2': 4.1,
        'so2': 1.4,
        'co': 5,
        'temp': None,
        'feels_like': None,
        'pressure': None,
        'humidity': None,
        'wind_speed': None,
        'wind_deg': None,
        'clouds': None,
        'weather_id': None,
        'weather_main': None
    }]

    try:
        store = FeatureStore()
        store.store_features(test_data)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
