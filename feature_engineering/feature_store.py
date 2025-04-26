import hopsworks
import pandas as pd
import os
import logging
import time
import json
from datetime import datetime
from typing import List, Dict
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Production-ready feature storage with Kafka workaround"""
    
    def __init__(self):
        # Disable Kafka at environment level
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
            self.fs = None

    def store_features(self, features: List[Dict]):
        """Guaranteed storage with multiple fallbacks"""
        try:
            df = self._prepare_dataframe(features)
            
            # 1. Always store locally first
            self._store_locally(df)
            
            # 2. Try Hopsworks if connected
            if self.fs:
                self._store_to_hopsworks(df)
                
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            self._emergency_save(features)

    def _prepare_dataframe(self, features: List[Dict]) -> pd.DataFrame:
        """Data cleaning and formatting"""
        df = pd.DataFrame(features)
        
        # Convert timestamp to milliseconds
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
        # Handle null values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(Config.NULL_FILL_VALUE)
            else:
                df[col] = df[col].fillna(Config.NULL_FILL_STRING)
                
        return df

    def _store_to_hopsworks(self, df: pd.DataFrame):
        """Hopsworks storage with offline mode"""
        try:
            fg = self.fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp',
                online_enabled=False,  # Disable online access
                statistics_config={"enabled": False}
            )
            
            fg.insert(df, write_options={"start_offline_backfill": True})
            logger.info(f"Stored {len(df)} records to Hopsworks")
            
        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise

    def _store_locally(self, df: pd.DataFrame):
        """Atomic local storage"""
        os.makedirs('data', exist_ok=True)
        filename = "data/karachi_aqi.csv"
        
        try:
            # Atomic write with temp file
            temp_file = f"{filename}.tmp"
            df.to_csv(temp_file, index=False)
            
            if os.path.exists(filename):
                os.replace(filename, f"{filename}.bak")
            os.replace(temp_file, filename)
            
            logger.info(f"Local storage successful: {filename}")
        except Exception as e:
            logger.error(f"Local storage failed: {str(e)}")
            raise

    def _emergency_save(self, data: List[Dict]):
        """Final JSON fallback"""
        try:
            os.makedirs('data/emergency', exist_ok=True)
            ts = int(time.time())
            with open(f"data/emergency/aqi_{ts}.json", 'w') as f:
                json.dump(data, f)
            logger.critical(f"Emergency backup: data/emergency/aqi_{ts}.json")
        except Exception as e:
            logger.critical(f"FATAL: All storage failed! {str(e)}")

# Example usage
if __name__ == "__main__":
    test_data = [{
        'city': 'Karachi',
        'timestamp': datetime.now(),
        'aqi': 150,
        'pm25': 120,
        'temp': None  # Will be converted to -1
    }]
    
    store = FeatureStore()
    store.store_features(test_data)
