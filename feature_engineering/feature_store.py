import logging
import pandas as pd
import os
import traceback
from utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Class for storing features in Hopsworks Feature Store."""

    def __init__(self):
        self.api_key = Config.HOPSWORKS_API_KEY
        self.project_name = Config.HOPSWORKS_PROJECT_NAME
        self.feature_store_name = Config.FEATURE_STORE_NAME
        self.can_connect = bool(self.api_key)

        if not self.can_connect:
            logger.warning("Hopsworks API key not set. Using local storage instead.")

    def store_features(self, features):
        """Store features in the feature store."""
        df = pd.DataFrame(features)

        if df.empty:
            logger.warning("No features to store")
            return

        logger.info(f"Storing {len(df)} features...")

        try:
            if self.can_connect:
                self._store_in_hopsworks(df)
            else:
                self._store_locally(df)

            logger.info("Features stored successfully.")
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            logger.debug(traceback.format_exc())

    ouFZ8BhcXFbDQy7S.GDkj3eGXwA4BgwzKSWqeEi53jUsd1fYSf22pxCnqG0tBZTM9RSTE2z1T64N7SErS
                # Insert features into the feature group
                response = fg.insert(city_df)
                if response.status_code != 200 or not response.text:
                    logger.error(f"Received empty response from Hopsworks for feature group {fg_name}")
                    return

                logger.info(f"Stored features for {city} in Hopsworks")

        except Exception as e:
            logger.error(f"Error storing in Hopsworks: {e}")
            logger.debug(traceback.format_exc())
            self._store_locally(df)

    def _store_locally(self, df):
        """Store features locally as CSV files."""
        os.makedirs('data', exist_ok=True)

        for city, city_df in df.groupby('city'):
            file_name = f"data/{city.lower().replace(' ', '_')}_features.csv"

            if os.path.exists(file_name):
                existing_df = pd.read_csv(file_name)
                combined_df = pd.concat([existing_df, city_df]).drop_duplicates(subset=['timestamp'])
                combined_df.to_csv(file_name, index=False)
            else:
                city_df.to_csv(file_name, index=False)

            logger.info(f"Stored features for {city} locally in {file_name}")

    def get_training_data(self, feature_view_name, target_cols=None):
        """Get training data from the feature store."""
        logger.info(f"Getting training data for {feature_view_name}...")

        try:
            if self.can_connect:
                return self._get_from_hopsworks(feature_view_name, target_cols)
            else:
                return self._get_from_local(feature_view_name, target_cols)
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            logger.debug(traceback.format_exc())
            return None, None

    def _get_from_hopsworks(self, feature_view_name, target_cols):
        """Get training data from Hopsworks feature store."""
        try:
            import hsfs

            conn = hsfs.connection(
                host="app.hopsworks.ai",
                project=self.project_name,
                api_key_value=self.api_key
            )
            fs = conn.get_feature_store()
            fg = fs.get_feature_group(feature_view_name)

            try:
                fv = fs.get_feature_view(feature_view_name)
            except Exception as e:
                logger.debug(f"Feature view {feature_view_name} not found, creating new one...")
                query = fg.select_all()
                fv = fs.create_feature_view(
                    name=feature_view_name,
                    description=f"Feature view for {feature_view_name}",
                    query=query
                )

            X, y = fv.training_data(target_cols)
            logger.info(f"Got training data from Hopsworks with {len(X)} rows")
            return X, y

        except Exception as e:
            logger.error(f"Error getting from Hopsworks: {e}")
            logger.debug(traceback.format_exc())
            return self._get_from_local(feature_view_name, target_cols)

    def _get_from_local(self, feature_view_name, target_cols):
        """Get training data from local storage."""
        file_name = f"data/{feature_view_name.replace('_aqi_features', '')}_features.csv"

        if not os.path.exists(file_name):
            logger.error(f"Local file {file_name} not found")
            return None, None

        df = pd.read_csv(file_name)

        if target_cols:
            X = df.drop(target_cols, axis=1)
            y = df[target_cols]
        else:
            X = df
            y = None

        logger.info(f"Got training data from local file with {len(X)} rows")
        return X, y
