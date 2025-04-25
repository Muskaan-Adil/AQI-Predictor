import logging
import pandas as pd
import os
import time
import traceback
from datetime import datetime
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Production-ready Feature Store for Hopsworks project 1219758"""

    def __init__(self):
        # Configuration for your specific project
        self.api_key = Config.HOPSWORKS_API_KEY
        self.project_name = "1219758"  # From your URL
        self.feature_store_name = "aqi_prediction"  # Change if different
        self.can_connect = False
        
        # Immediate connection test
        if self.api_key:
            self._verify_connection()

    def _verify_connection(self):
        """Test connection during initialization"""
        try:
            import hsfs
            conn = hsfs.connection(
                host="c.app.hopsworks.ai",
                project=self.project_name,
                api_key_value=self.api_key,
                hostname_verification=False
            )
            # Verify project access
            project = conn.get_project()
            logger.info(f"Connected to Hopsworks project: {project.name}")
            self.can_connect = True
        except Exception as e:
            logger.error(f"Connection verification failed: {str(e)}")
            self.can_connect = False

    def store_features(self, features):
        """Main storage method with automatic fallback"""
        try:
            df = pd.DataFrame(features) if not isinstance(features, pd.DataFrame) else features.copy()
            if df.empty:
                logger.warning("No features to store")
                return

            logger.info(f"Preparing to store {len(df)} records...")
            df = self._prepare_data(df)

            if self.can_connect:
                self._store_in_hopsworks(df)
            else:
                self._store_locally(df)

        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            self._store_locally(df)

    def _prepare_data(self, df):
        """Prepare DataFrame for Hopsworks"""
        df = df.copy()
        
        # Convert timestamp to Unix time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
        
        # Clean data
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(-1)
        
        return df

    def _store_in_hopsworks(self, df):
        """Store data in Hopsworks Feature Store"""
        try:
            import hsfs
            
            # Re-establish connection
            conn = hsfs.connection(
                host="c.app.hopsworks.ai",
                project=self.project_name,
                api_key_value=self.api_key,
                hostname_verification=False
            )
            fs = conn.get_feature_store()

            # Process by city
            for city, city_df in df.groupby('city'):
                fg_name = f"{city.lower()}_aqi_features"
                
                try:
                    # Get or create feature group
                    fg = fs.get_or_create_feature_group(
                        name=fg_name,
                        version=1,
                        primary_key=['timestamp'],
                        event_time='timestamp',
                        description=f"AQI data for {city}",
                        online_enabled=False
                    )

                    # Insert with retry
                    for attempt in range(3):
                        try:
                            fg.insert(city_df)
                            logger.info(f"Stored {len(city_df)} records for {city}")
                            break
                        except Exception as e:
                            if attempt == 2:
                                raise
                            logger.warning(f"Attempt {attempt+1} failed, retrying...")
                            time.sleep(2)

                except Exception as e:
                    logger.error(f"Failed to store {city}: {str(e)}")
                    self._store_locally(city_df)

        except Exception as e:
            logger.error(f"Hopsworks storage failed: {str(e)}")
            self._store_locally(df)

    def _store_locally(self, df):
        """Atomic local storage with backups"""
        os.makedirs('data', exist_ok=True)
        
        for city, city_df in df.groupby('city'):
            filename = f"data/{city.lower()}_features.csv"
            temp_file = f"{filename}.tmp"
            
            try:
                # Write to temp file first
                city_df.to_csv(temp_file, index=False)
                
                # Atomic replace
                if os.path.exists(filename):
                    os.replace(filename, f"{filename}.bak")
                os.replace(temp_file, filename)
                
                logger.info(f"Stored {len(city_df)} records locally at {filename}")
            except Exception as e:
                logger.error(f"Local storage failed: {str(e)}")
                emergency_file = f"data/emergency_{int(time.time())}.csv"
                city_df.to_csv(emergency_file)
                logger.critical(f"Saved emergency backup to {emergency_file}")

    def get_training_data(self, feature_view_name, target_cols=None):
        """Retrieve training data"""
        try:
            if self.can_connect:
                return self._get_from_hopsworks(feature_view_name, target_cols)
            return self._get_from_local(feature_view_name, target_cols)
        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            return None, None

    def _get_from_hopsworks(self, feature_view_name, target_cols):
        """Retrieve from Hopsworks"""
        try:
            import hsfs
            
            conn = hsfs.connection(
                host="c.app.hopsworks.ai",
                project=self.project_name,
                api_key_value=self.api_key,
                hostname_verification=False
            )
            fs = conn.get_feature_store()
            
            fv = fs.get_feature_view(feature_view_name, version=1)
            return fv.training_data(target_cols)

        except Exception as e:
            logger.error(f"Hopsworks retrieval failed: {str(e)}")
            return self._get_from_local(feature_view_name, target_cols)

    def _get_from_local(self, feature_view_name, target_cols):
        """Retrieve from local CSV"""
        city = feature_view_name.replace('_aqi_features', '')
        filename = f"data/{city}_features.csv"
        
        if not os.path.exists(filename):
            return None, None
            
        try:
            df = pd.read_csv(filename)
            if target_cols:
                return df.drop(target_cols, axis=1), df[target_cols]
            return df, None
        except Exception as e:
            logger.error(f"Local read failed: {str(e)}")
            return None, None
