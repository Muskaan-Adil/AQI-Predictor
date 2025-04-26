import hopsworks
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Hopsworks-only feature storage with strict type handling"""
    
    def __init__(self):
        # Disable Kafka to prevent connection issues
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
            raise  # Fail fast if connection fails

    def store_features(self, features: List[Dict]):
        """Store features with strict type validation"""
        try:
            df = self._prepare_data(features)
            self._store_to_hopsworks(df)
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            raise  # Let the caller handle the error

    def _prepare_data(self, features: List[Dict]) -> pd.DataFrame:
        """Convert data types to match Hopsworks schema exactly"""
        df = pd.DataFrame(features)
        
        # Convert timestamp to milliseconds (bigint)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
        # Define exact type mapping
        type_mapping = {
            # BigInt columns
            'aqi': 'int64',
            'pm25': 'int64',
            'pm10': 'int64',
            'temp': 'int64',
            'feels_like': 'int64',
            'pressure': 'int64',
            'humidity': 'int64',
            'wind_speed': 'int64',
            'wind_deg': 'int64',
            'clouds': 'int64',
            'weather_id': 'int64',
            
            # Double columns
            'o3': 'float64',
            'no2': 'float64',
            'so2': 'float64',
            'co': 'float64',
            
            # String columns
            'city': 'str',
            'weather_main': 'str'
        }
        
        # Apply type conversion
        for col, dtype in type_mapping.items():
            if col in df.columns:
                if dtype == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype('int64')
                elif dtype == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1.0).astype('float64')
                else:
                    df[col] = df[col].fillna('').astype('str')
        
        return df

    def _store_to_hopsworks(self, df: pd.DataFrame):
        """Strict schema validation with feature group creation"""
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
            
            # Validate schema before insert
            fg.validate(dataframe=df)
            
            # Insert data
            fg.insert(df)
            logger.info(f"Successfully stored {len(df)} records")
            
        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    test_data = [{
        'city': 'Karachi',
        'timestamp': datetime.now(),
        'aqi': 154,
        'pm25': 154,
        'pm10': 81,
        'o3': 18.6,
        'no2': 4.1,
        'so2': 1.4,
        'co': 5.0,
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
