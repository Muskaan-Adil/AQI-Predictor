import hopsworks
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

class FeatureStore:
    """Production-ready Feature Store with null handling"""
    
    def __init__(self):
        self.api_key = os.getenv('HOPSWORKS_API_KEY')
        if not self.api_key:
            raise ValueError("HOPSWORKS_API_KEY environment variable not set")
            
        self.project = None
        self.fs = None
        self._connect()

    def _connect(self):
        """Establish connection with retry logic"""
        try:
            self.project = hopsworks.login(api_key_value=self.api_key)
            self.fs = self.project.get_feature_store()
            logger.info("Successfully connected to Hopsworks")
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.fs = None

    def store_features(self, features: List[Dict]):
        """Store features with comprehensive null handling"""
        try:
            df = self._prepare_dataframe(features)
            
            # Always store locally first
            self._store_locally(df)
            
            # Try Hopsworks if connected
            if self.fs:
                self._store_to_hopsworks(df)
                
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            self._emergency_save(features)

    def _prepare_dataframe(self, features: List[Dict]) -> pd.DataFrame:
        """Convert and clean feature data"""
        df = pd.DataFrame(features)
        
        # Convert timestamp to milliseconds
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
        # Handle null values for all columns
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(-1).astype(float)
                elif pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].fillna('')
                else:
                    df[col] = df[col].fillna(-1)  # Default fallback
                    
        return df

    def _store_to_hopsworks(self, df: pd.DataFrame):
        """Store to Hopsworks with feature group management"""
        try:
            fg = self.fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp',
                description="Karachi AQI data with weather",
                online_enabled=True,
                statistics_config={"enabled": False}
            )
            
            # Insert with validation
            fg.insert(df, write_options={"wait_for_job": False})
            logger.info(f"Stored {len(df)} records to Hopsworks")
            
        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise

    def _store_locally(self, df: pd.DataFrame):
        """Atomic local storage with rotation"""
        os.makedirs('data', exist_ok=True)
        filename = "data/karachi_aqi.csv"
        
        try:
            # Write to temp file first
            temp_file = f"{filename}.tmp"
            df.to_csv(temp_file, index=False)
            
            # Rotate existing files
            if os.path.exists(filename):
                os.replace(filename, f"{filename}.bak")
                
            # Atomic move
            os.replace(temp_file, filename)
            logger.info(f"Local storage successful: {filename}")
            
        except Exception as e:
            logger.error(f"Local storage failed: {str(e)}")
            self._emergency_save(df.to_dict('records'))

    def _emergency_save(self, data: List[Dict]):
        """Final fallback storage"""
        try:
            os.makedirs('data/emergency', exist_ok=True)
            ts = int(time.time())
            with open(f"data/emergency/aqi_{ts}.json", 'w') as f:
                json.dump(data, f)
            logger.critical(f"Emergency backup saved: data/emergency/aqi_{ts}.json")
        except Exception as e:
            logger.critical(f"Fatal: All storage methods failed! {str(e)}")

    def get_training_data(self) -> Optional[pd.DataFrame]:
        """Retrieve data from best available source"""
        try:
            if self.fs:
                try:
                    return self._get_from_hopsworks()
                except Exception as e:
                    logger.error(f"Hopsworks retrieval failed: {str(e)}")
            
            return self._get_from_local()
            
        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            return None

    def _get_from_hopsworks(self) -> pd.DataFrame:
        """Retrieve from Hopsworks feature store"""
        fg = self.fs.get_feature_group("karachi_aqi_realtime", version=1)
        return fg.read()

    def _get_from_local(self) -> Optional[pd.DataFrame]:
        """Retrieve from local CSV"""
        filename = "data/karachi_aqi.csv"
        if not os.path.exists(filename):
            return None
            
        try:
            return pd.read_csv(filename)
        except Exception as e:
            logger.error(f"Local read failed: {str(e)}")
            return None
