import logging
import pandas as pd
import os
import time
import traceback
from datetime import datetime
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Enhanced Feature Store with robust Hopsworks integration and fallback mechanisms."""

    def __init__(self):
        self.api_key = Config.HOPSWORKS_API_KEY
        self.project_name = Config.HOPSWORKS_PROJECT_NAME
        self.feature_store_name = Config.FEATURE_STORE_NAME
        self.can_connect = bool(self.api_key)
        
        # Connection parameters with defaults
        self.connection_params = {
            "host": "c.app.hopsworks.ai",  # Free tier uses 'c.' prefix
            "project": self.project_name,
            "api_key_value": self.api_key,
            "hostname_verification": False,
            "cert_folder": "/tmp/hopsworks-certs"
        }

        if not self.can_connect:
            logger.warning("Hopsworks API key not set. Using local storage only.")

    def store_features(self, features):
        """
        Main method to store features with automatic fallback.
        
        Args:
            features (list/dict/DataFrame): Features to store
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(features) if not isinstance(features, pd.DataFrame) else features.copy()
            
            if df.empty:
                logger.warning("No features to store")
                return

            logger.info(f"Preparing to store {len(df)} records...")
            
            # Data preparation
            df = self._prepare_dataframe(df)
            
            # Storage decision
            if self.can_connect and self._validate_hopsworks_connection():
                logger.info("Hopsworks connection validated. Attempting cloud storage...")
                self._store_in_hopsworks(df)
            else:
                logger.warning("Using local storage fallback")
                self._store_locally(df)

        except Exception as e:
            logger.error(f"Store features failed: {str(e)}")
            logger.debug(traceback.format_exc())
            self._store_locally(df)  # Final fallback

    def _prepare_dataframe(self, df):
        """Ensure DataFrame meets storage requirements."""
        df = df.copy()
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            if isinstance(df['timestamp'].iloc[0], datetime):
                df['timestamp'] = df['timestamp'].astype('int64') // 10**9  # Unix timestamp
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
        
        # Clean data
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(-1)
        
        return df

    def _validate_hopsworks_connection(self):
        """Test if Hopsworks connection works."""
        try:
            import hsfs
            conn = hsfs.connection(**self.connection_params)
            fs = conn.get_feature_store()
            fs.get_feature_groups()  # Simple API call to test
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def _store_in_hopsworks(self, df):
        """Store features in Hopsworks with enhanced error handling."""
        try:
            import hsfs
            
            logger.info("Initializing Hopsworks connection...")
            conn = hsfs.connection(**self.connection_params)
            fs = conn.get_feature_store()

            for city, city_df in df.groupby('city'):
                fg_name = f"{city.lower().replace(' ', '_')}_aqi_features"
                logger.info(f"Processing feature group: {fg_name}")

                try:
                    # Get or create feature group
                    fg = fs.get_or_create_feature_group(
                        name=fg_name,
                        version=1,
                        primary_key=['timestamp'],
                        event_time='timestamp',
                        description=f"AQI features for {city}",
                        statistics_config={"enabled": False},
                        online_enabled=False
                    )

                    # Insert with retry logic
                    max_attempts = 3
                    for attempt in range(max_attempts):
                        try:
                            fg.insert(city_df, write_options={
                                "wait_for_job": True,
                                "start_offline_backfill": False
                            })
                            logger.info(f"Successfully stored features for {city}")
                            break
                        except Exception as e:
                            if attempt == max_attempts - 1:
                                raise
                            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                            time.sleep(2)

                except Exception as e:
                    logger.error(f"Failed to store {city} in Hopsworks: {str(e)}")
                    self._store_locally(city_df)  # City-level fallback

        except Exception as e:
            logger.error(f"Critical Hopsworks error: {str(e)}")
            raise  # Will trigger the fallback in store_features()

    def _store_locally(self, df):
        """Atomic local storage with backup mechanism."""
        try:
            os.makedirs('data', exist_ok=True)
            
            for city, city_df in df.groupby('city'):
                base_name = city.lower().replace(' ', '_')
                filename = f"data/{base_name}_features.csv"
                temp_file = f"data/{base_name}_features.tmp"
                
                # Write to temp file first
                city_df.to_csv(temp_file, index=False)
                
                # Atomic replace
                if os.path.exists(filename):
                    backup_file = f"data/{base_name}_features.bak"
                    os.replace(filename, backup_file)
                os.replace(temp_file, filename)
                
                logger.info(f"Stored {len(city_df)} records for {city} locally at {filename}")

        except Exception as e:
            logger.error(f"Local storage failed: {str(e)}")
            # Emergency fallback
            emergency_file = f"data/emergency_{int(time.time())}.csv"
            df.to_csv(emergency_file, index=False)
            logger.critical(f"Saved emergency backup to {emergency_file}")

    def get_training_data(self, feature_view_name, target_cols=None):
        """Get training data from the best available source."""
        try:
            if self.can_connect and self._validate_hopsworks_connection():
                logger.info("Attempting to get data from Hopsworks...")
                return self._get_from_hopsworks(feature_view_name, target_cols)
            else:
                logger.info("Using local training data")
                return self._get_from_local(feature_view_name, target_cols)
        except Exception as e:
            logger.error(f"Failed to get training data: {str(e)}")
            return None, None

    def _get_from_hopsworks(self, feature_view_name, target_cols):
        """Retrieve data from Hopsworks Feature Store."""
        try:
            import hsfs
            
            conn = hsfs.connection(**self.connection_params)
            fs = conn.get_feature_store()
            
            # Try to get existing feature view
            try:
                fv = fs.get_feature_view(feature_view_name, version=1)
                logger.info("Found existing feature view")
            except:
                logger.info("Creating new feature view")
                fg = fs.get_feature_group(feature_view_name)
                fv = fs.create_feature_view(
                    name=feature_view_name,
                    version=1,
                    query=fg.select_all()
                )

            # Get training data
            X, y = fv.training_data(
                target_cols,
                read_options={"use_hive": True}
            )
            
            logger.info(f"Retrieved {len(X)} records from Hopsworks")
            return X, y

        except Exception as e:
            logger.error(f"Hopsworks retrieval failed: {str(e)}")
            return self._get_from_local(feature_view_name, target_cols)

    def _get_from_local(self, feature_view_name, target_cols):
        """Get training data from local CSV files."""
        base_name = feature_view_name.replace('_aqi_features', '')
        filename = f"data/{base_name}_features.csv"
        
        if not os.path.exists(filename):
            logger.error(f"Local file {filename} not found")
            return None, None

        try:
            df = pd.read_csv(filename)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            if target_cols:
                y = df[target_cols]
                X = df.drop(columns=target_cols)
            else:
                X = df
                y = None
                
            logger.info(f"Retrieved {len(X)} records from local storage")
            return X, y
            
        except Exception as e:
            logger.error(f"Local data read failed: {str(e)}")
            return None, None
