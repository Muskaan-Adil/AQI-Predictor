import logging
import pandas as pd
import os
import time
from datetime import datetime
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Fixed version for Hopsworks project access issues"""
    
    def __init__(self):
        self.api_key = Config.HOPSWORKS_API_KEY
        self.project_id = "1219758"  # From your URL
        self.can_connect = False
        
        if not self.api_key:
            logger.warning("No API key configured")
            return
            
        # New connection approach
        self._initialize_hopsworks()

    def _initialize_hopsworks(self):
        """Alternative connection method"""
        try:
            import hsfs
            from hsfs.client.exceptions import RestAPIError
            
            # Try both connection methods
            connection_methods = [
                {
                    "host": "c.app.hopsworks.ai",
                    "project": self.project_id,
                    "api_key_value": self.api_key,
                    "hostname_verification": False
                },
                {
                    "host": "app.hopsworks.ai",
                    "project": self.project_id,
                    "api_key_value": self.api_key,
                    "hostname_verification": False
                }
            ]
            
            for params in connection_methods:
                try:
                    conn = hsfs.connection(**params)
                    project = conn.get_project()
                    logger.info(f"Connected to project: {project.name}")
                    self.can_connect = True
                    self.connection_params = params
                    break
                except RestAPIError as e:
                    logger.debug(f"Attempt failed with {params['host']}: {str(e)}")
                    continue
                    
            if not self.can_connect:
                logger.error("All connection attempts failed")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")

    def store_features(self, features):
        """Storage with enhanced error handling"""
        try:
            df = pd.DataFrame(features)
            if df.empty:
                return

            df = self._prepare_data(df)
            
            if self.can_connect:
                self._store_in_hopsworks(df)
            else:
                self._store_locally(df)
                
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            self._store_locally(df)

    def _prepare_data(self, df):
        """Data preparation with null handling"""
        df = df.copy()
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        
        # Fill nulls
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(-1)
                else:
                    df[col] = df[col].fillna('')
        
        return df

    def _store_in_hopsworks(self, df):
        """Storage with project verification"""
        try:
            import hsfs
            from hsfs.client.exceptions import RestAPIError
            
            # Reconnect using working params
            conn = hsfs.connection(**self.connection_params)
            fs = conn.get_feature_store()
            
            # Test with minimal feature group
            test_fg = fs.get_or_create_feature_group(
                name="connection_test",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp'
            )
            
            # Insert test record
            test_record = df.iloc[:1].copy()
            test_record['timestamp'] = int(time.time())  # Current timestamp
            test_fg.insert(test_record)
            
            logger.info("Hopsworks connection verified - storing all data")
            
            # Now store actual data
            for city, city_df in df.groupby('city'):
                fg = fs.get_or_create_feature_group(
                    name=f"{city.lower()}_aqi",
                    version=1,
                    primary_key=['timestamp'],
                    event_time='timestamp'
                )
                fg.insert(city_df)
                logger.info(f"Stored {len(city_df)} records for {city}")
                
        except RestAPIError as e:
            logger.error(f"Hopsworks API error: {str(e)}")
            self._store_locally(df)
        except Exception as e:
            logger.error(f"Storage error: {str(e)}")
            self._store_locally(df)

    def _store_locally(self, df):
        """Reliable local storage"""
        os.makedirs('data', exist_ok=True)
        
        for city, city_df in df.groupby('city'):
            filename = f"data/{city.lower()}_features.csv"
            
            try:
                if os.path.exists(filename):
                    existing = pd.read_csv(filename)
                    combined = pd.concat([existing, city_df]).drop_duplicates('timestamp')
                    combined.to_csv(filename, index=False)
                else:
                    city_df.to_csv(filename, index=False)
                    
                logger.info(f"Local backup: {filename}")
            except Exception as e:
                logger.error(f"Local storage failed: {str(e)}")
                emergency_file = f"data/emergency_{int(time.time())}.csv"
                city_df.to_csv(emergency_file)
                logger.critical(f"Emergency save: {emergency_file}")
