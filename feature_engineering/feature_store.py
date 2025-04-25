import logging
import pandas as pd
import os
import time
import json
from datetime import datetime
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Ultra-reliable Feature Store with comprehensive error handling"""
    
    def __init__(self):
        self.api_key = Config.HOPSWORKS_API_KEY
        self.project_id = "1219758"  # From your URL
        self.can_connect = False
        self.connection_params = None
        
        if not self.api_key:
            logger.warning("API key not configured - using local storage only")
            return
            
        self._establish_connection()

    def _establish_connection(self):
        """Robust connection establishment with detailed diagnostics"""
        try:
            import hsfs
            import requests
            
            # Try all possible connection combinations
            connection_options = [
                {"host": "c.app.hopsworks.ai", "project": self.project_id},
                {"host": "app.hopsworks.ai", "project": self.project_id},
                {"host": "c.app.hopsworks.ai", "project": str(self.project_id)},
                {"host": "app.hopsworks.ai", "project": str(self.project_id)}
            ]
            
            for option in connection_options:
                params = {
                    **option,
                    "api_key_value": self.api_key,
                    "hostname_verification": False
                }
                
                try:
                    # Test connection via direct API first
                    test_url = f"https://{params['host']}/hopsworks-api/api/project/{params['project']}/get"
                    headers = {"Authorization": f"ApiKey {params['api_key_value']}"}
                    
                    response = requests.get(test_url, headers=headers, verify=False, timeout=10)
                    if response.status_code == 200:
                        logger.info(f"API test successful for {params['host']}")
                        conn = hsfs.connection(**params)
                        project = conn.get_project()
                        logger.info(f"Connected to project: {project.name}")
                        self.can_connect = True
                        self.connection_params = params
                        return
                        
                except Exception as e:
                    logger.debug(f"Connection test failed for {params}: {str(e)}")
                    continue
                    
            logger.error("All connection attempts failed")
            
        except ImportError:
            logger.error("hsfs package not installed - using local storage")
        except Exception as e:
            logger.error(f"Connection initialization failed: {str(e)}")

    def store_features(self, features):
        """Storage with multi-layer fallback"""
        try:
            df = self._prepare_dataframe(features)
            
            if not self.can_connect:
                logger.info("Proceeding with local storage only")
                self._store_locally(df)
                return
                
            try:
                self._store_in_hopsworks(df)
            except Exception as e:
                logger.error(f"Hopsworks storage failed: {str(e)}")
                self._store_locally(df)
                
        except Exception as e:
            logger.critical(f"Storage process failed: {str(e)}")
            self._emergency_save(features)

    def _prepare_dataframe(self, features):
        """Create and clean DataFrame"""
        df = pd.DataFrame(features)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9
            
        # Clean data
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(-1)
                else:
                    df[col] = df[col].fillna('')
                    
        return df

    def _store_in_hopsworks(self, df):
        """Hopsworks storage with validation"""
        import hsfs
        from hsfs.client.exceptions import RestAPIError
        
        try:
            conn = hsfs.connection(**self.connection_params)
            fs = conn.get_feature_store()
            
            # Process by city
            for city, city_df in df.groupby('city'):
                fg_name = f"{city.lower()}_aqi"
                
                try:
                    fg = fs.get_or_create_feature_group(
                        name=fg_name,
                        version=1,
                        primary_key=['timestamp'],
                        event_time='timestamp',
                        online_enabled=False
                    )
                    
                    # Insert with validation
                    result = fg.insert(city_df)
                    if hasattr(result, 'status_code') and result.status_code != 200:
                        raise Exception(f"Insert failed with status {result.status_code}")
                        
                    logger.info(f"Stored {len(city_df)} records for {city} in Hopsworks")
                    
                except RestAPIError as e:
                    logger.error(f"Hopsworks API error for {city}: {str(e)}")
                    self._store_locally(city_df)
                except Exception as e:
                    logger.error(f"Storage error for {city}: {str(e)}")
                    self._store_locally(city_df)
                    
        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            raise  # Trigger fallback

    def _store_locally(self, df):
        """Atomic local storage with rotation"""
        os.makedirs('data', exist_ok=True)
        
        for city, city_df in df.groupby('city'):
            base_name = f"data/{city.lower()}_features"
            
            try:
                # Write to temporary file
                temp_file = f"{base_name}.tmp"
                city_df.to_csv(temp_file, index=False)
                
                # Rotate files
                if os.path.exists(f"{base_name}.csv"):
                    os.replace(f"{base_name}.csv", f"{base_name}.bak")
                
                # Atomic move
                os.replace(temp_file, f"{base_name}.csv")
                logger.info(f"Local storage successful: {base_name}.csv")
                
            except Exception as e:
                logger.error(f"Local storage failed: {str(e)}")
                self._emergency_save(city_df)

    def _emergency_save(self, data):
        """Last-resort storage"""
        try:
            os.makedirs('data/emergency', exist_ok=True)
            timestamp = int(time.time())
            with open(f"data/emergency/save_{timestamp}.json", 'w') as f:
                json.dump(data, f)
            logger.critical(f"Emergency save: data/emergency/save_{timestamp}.json")
        except Exception as e:
            logger.critical(f"FATAL: All storage methods failed! Data lost: {str(e)}")

    def get_training_data(self, feature_view_name, target_cols=None):
        """Retrieve data from best available source"""
        try:
            if self.can_connect:
                try:
                    return self._get_from_hopsworks(feature_view_name, target_cols)
                except Exception as e:
                    logger.error(f"Hopsworks retrieval failed: {str(e)}")
                    
            return self._get_from_local(feature_view_name, target_cols)
            
        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            return None, None
