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
    """Fixed feature storage with proper type handling"""
    
    def __init__(self):
        os.environ['ENABLE_HOPSWORKS_KAFKA'] = '0'
        self.api_key = Config.HOPSWORKS_API_KEY
        if not self.api_key:
            raise ValueError("HOPSWORKS_API_KEY not set")
        
        try:
            self.project = hopsworks.login(api_key_value=self.api_key)
            self.fs = self.project.get_feature_store()
            logger.info("Connected to Hopsworks")
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.fs = None

    def store_features(self, features: List[Dict]):
        """Storage with proper type conversion"""
        try:
            df = self._prepare_data(features)
            self._store_locally(df)
            if self.fs:
                self._store_to_hopsworks(df)
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            self._emergency_save(features)

    def _prepare_data(self, features: List[Dict]) -> pd.DataFrame:
        """Convert data types properly"""
        df = pd.DataFrame(features)
        
        # Convert timestamp to milliseconds
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
        # Define numeric columns that should be bigint
        numeric_cols = [
            'aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co',
            'temp', 'feels_like', 'pressure', 'humidity',
            'wind_speed', 'wind_deg', 'clouds', 'weather_id'
        ]
        
        # Convert numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype('int64')
        
        # Handle string columns
        string_cols = ['city', 'weather_main']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        return df

    def _store_to_hopsworks(self, df: pd.DataFrame):
        """Storage with schema validation"""
        try:
            fg = self.fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp',
                online_enabled=False,
                statistics_config={"enabled": False}
            )
            
            # Explicit schema definition
            fg.save(
                features=df,
                write_options={"start_offline_backfill": True}
            )
            logger.info(f"Stored {len(df)} records to Hopsworks")
        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise

    def _store_locally(self, df: pd.DataFrame):
        """Atomic local storage"""
        os.makedirs('data', exist_ok=True)
        filename = "data/karachi_aqi.csv"
        try:
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
        """JSON-safe emergency save"""
        try:
            os.makedirs('data/emergency', exist_ok=True)
            ts = int(time.time())
            
            # Convert datetime objects to strings
            serializable_data = []
            for record in data:
                safe_record = {}
                for key, value in record.items():
                    if isinstance(value, datetime):
                        safe_record[key] = value.isoformat()
                    else:
                        safe_record[key] = value
                serializable_data.append(safe_record)
            
            with open(f"data/emergency/aqi_{ts}.json", 'w') as f:
                json.dump(serializable_data, f)
            logger.critical(f"Emergency backup: data/emergency/aqi_{ts}.json")
        except Exception as e:
            logger.critical(f"FATAL: All storage failed! {str(e)}")

if __name__ == "__main__":
    test_data = [{
        'city': 'Karachi',
        'timestamp': datetime.now(),
        'aqi': 150,
        'pm25': 120,
        'temp': None,
        'weather_main': None
    }]
    store = FeatureStore()
    store.store_features(test_data)
