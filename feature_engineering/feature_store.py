import logging
import pandas as pd
import os
import time
from datetime import datetime
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Optimized Feature Store for your Hopsworks project"""
    
    def __init__(self):
        # Verified configuration for your specific project
        self.api_key = Config.HOPSWORKS_API_KEY
        self.project_id = "1219758"  # From your URL
        self.feature_store_name = "air_quality_featurestore"  # Verified in your project
        self.can_connect = False
        
        if not self.api_key:
            logger.warning("API key not configured - using local storage")
            return
            
        self._test_connection()

    def _test_connection(self):
        """Direct connection test for your project"""
        try:
            import hsfs
            import requests
            
            # First verify API access
            test_url = f"https://c.app.hopsworks.ai/hopsworks-api/api/project/{self.project_id}/dataset"
            headers = {
                "Authorization": f"ApiKey {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(test_url, headers=headers, timeout=10)
            if response.status_code != 200:
                raise Exception(f"API test failed with status {response.status_code}")
            
            # Now try HSFS connection
            conn = hsfs.connection(
                host="c.app.hopsworks.ai",
                project=self.project_id,
                api_key_value=self.api_key,
                hostname_verification=False
            )
            
            # Verify feature store access
            fs = conn.get_feature_store(name=self.feature_store_name)
            logger.info(f"Connected to feature store: {fs.name}")
            self.can_connect = True
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            self.can_connect = False

    def store_features(self, features):
        """Optimized storage method for your AQI data"""
        try:
            df = pd.DataFrame(features)
            if df.empty:
                return
                
            df = self._clean_data(df)
            
            if self.can_connect:
                self._store_to_hopsworks(df)
            self._store_locally(df)  # Always store locally as backup
            
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            self._emergency_save(features)

    def _clean_data(self, df):
        """Data cleaning specific to your AQI format"""
        # Convert timestamp to milliseconds since epoch
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
        
        # Fill nulls with appropriate defaults
        numeric_cols = ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(-1).astype(float)
                
        return df

    def _store_to_hopsworks(self, df):
        """Optimized storage for your project's feature store"""
        try:
            import hsfs
            
            conn = hsfs.connection(
                host="c.app.hopsworks.ai",
                project=self.project_id,
                api_key_value=self.api_key,
                hostname_verification=False
            )
            fs = conn.get_feature_store(name=self.feature_store_name)
            
            # Create feature group for Karachi
            fg = fs.get_or_create_feature_group(
                name="karachi_aqi_realtime",
                version=1,
                primary_key=['timestamp'],
                event_time='timestamp',
                description="Real-time AQI data for Karachi",
                online_enabled=True,  # Enable online access
                statistics_config={"enabled": False}  # Disable stats for performance
            )
            
            # Insert data
            fg.insert(df)
            logger.info(f"Stored {len(df)} records to Hopsworks")
            
        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise

    def _store_locally(self, df):
        """Atomic local storage with rotation"""
        os.makedirs('data', exist_ok=True)
        filename = "data/karachi_aqi.csv"
        
        try:
            # Write to temp file first
            temp_file = f"{filename}.tmp"
            df.to_csv(temp_file, index=False)
            
            # Rotate files
            if os.path.exists(filename):
                os.replace(filename, f"{filename}.bak")
                
            # Atomic move
            os.replace(temp_file, filename)
            logger.info(f"Local storage successful: {filename}")
            
        except Exception as e:
            logger.error(f"Local storage failed: {str(e)}")
            self._emergency_save(df)

    def _emergency_save(self, data):
        """Final fallback storage"""
        try:
            os.makedirs('data/emergency', exist_ok=True)
            ts = int(time.time())
            with open(f"data/emergency/aqi_{ts}.json", 'w') as f:
                if isinstance(data, pd.DataFrame):
                    data.to_json(f, orient='records')
                else:
                    json.dump(data, f)
            logger.critical(f"Emergency backup saved: data/emergency/aqi_{ts}.json")
        except Exception as e:
            logger.critical(f"Fatal: All storage methods failed! {str(e)}")

    def get_training_data(self):
        """Retrieve data from best available source"""
        try:
            if self.can_connect:
                try:
                    return self._get_from_hopsworks()
                except Exception as e:
                    logger.error(f"Hopsworks retrieval failed: {str(e)}")
            
            return self._get_from_local()
            
        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            return None

    def _get_from_hopsworks(self):
        """Retrieve from your project's feature store"""
        import hsfs
        
        conn = hsfs.connection(
            host="c.app.hopsworks.ai",
            project=self.project_id,
            api_key_value=self.api_key,
            hostname_verification=False
        )
        fs = conn.get_feature_store(name=self.feature_store_name)
        
        fg = fs.get_feature_group("karachi_aqi_realtime", version=1)
        return fg.read()

    def _get_from_local(self):
        """Retrieve from local CSV"""
        filename = "data/karachi_aqi.csv"
        if not os.path.exists(filename):
            return None
            
        return pd.read_csv(filename)
