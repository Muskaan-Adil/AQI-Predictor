import numpy as np
import pandas as pd
import hopsworks
import logging
from src.utils.config import Config

logger = logging.getLogger(__name__)

class FeatureStore:
    """Class for interacting with the Hopsworks Feature Store."""
    
    def __init__(self, project_name=None, api_key=None):
        """Initialize the feature store connection.
        
        Args:
            project_name (str, optional): Hopsworks project name. Defaults to Config value.
            api_key (str, optional): Hopsworks API key. Defaults to Config value.
        """
        self.project_name = project_name or Config.HOPSWORKS_PROJECT_NAME
        self.api_key = api_key or Config.HOPSWORKS_API_KEY
        self.feature_store_name = Config.FEATURE_STORE_NAME
        self.project = None
        self.fs = None
        
        self._connect()
    
    def _connect(self):
        """Connect to Hopsworks project and feature store."""
        try:
            self.project = hopsworks.login(
                project=self.project_name,
                api_key_value=self.api_key
            )
            self.fs = self.project.get_feature_store()
            logger.info(f"Connected to Hopsworks project: {self.project_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            self.project = None
            self.fs = None
    
    def create_feature_group(self, name, version=1, description="", primary_key=None, event_time_col=None):
        """Create a feature group in the feature store.
        
        Args:
            name (str): Feature group name.
            version (int, optional): Feature group version. Defaults to 1.
            description (str, optional): Feature group description. Defaults to "".
            primary_key (list, optional): List of primary key columns. Defaults to None.
            event_time_col (str, optional): Event time column name. Defaults to None.
            
        Returns:
            object: Feature group object or None if creation fails.
        """
        if self.fs is None:
            logger.error("Not connected to Hopsworks feature store")
            return None
        
        try:
            fg = self.fs.get_or_create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=primary_key,
                event_time=event_time_col,
                online_enabled=True
            )
            logger.info(f"Created/retrieved feature group: {name} (version {version})")
            return fg
        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            return None
    
    def insert_data(self, feature_group, df):
        """Insert data into a feature group.
        
        Args:
            feature_group: Feature group object.
            df (pd.DataFrame): DataFrame with data to insert.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if feature_group is None:
            logger.error("Feature group not provided")
            return False
        
        try:
            feature_group.insert(df)
            logger.info(f"Inserted {len(df)} rows into feature group: {feature_group.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            return False
    
    def create_city_feature_groups(self, cities):
        """Create feature groups for each city.
        
        Args:
            cities (list): List of city dictionaries.
            
        Returns:
            dict: Dictionary mapping city names to feature group objects.
        """
        feature_groups = {}
        
        for city in cities:
            city_name = city['name'].replace(' ', '_').lower()
            fg_name = f"{city_name}_aqi_features"
            
            fg = self.create_feature_group(
                name=fg_name,
                description=f"AQI features for {city['name']}",
                primary_key=['timestamp'],
                event_time_col='timestamp'
            )
            
            if fg is not None:
                feature_groups[city['name']] = fg
        
        return feature_groups
    
    def store_features(self, df):
        """Store features for all cities in the DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with features (must have 'city' column).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if 'city' not in df.columns:
            logger.error("DataFrame must have 'city' column")
            return False
        
        unique_cities = [{'name': city} for city in df['city'].unique()]
        feature_groups = self.create_city_feature_groups(unique_cities)
        
        success = True
        for city, fg in feature_groups.items():
            city_df = df[df['city'] == city].copy()
            if not city_df.empty:
                success = success and self.insert_data(fg, city_df)
        
        return success
    
    def get_training_data(self, feature_view_name, version=1, target_cols=None):
        """Get training data from a feature view.
        
        Args:
            feature_view_name (str): Feature view name.
            version (int, optional): Feature view version. Defaults to 1.
            target_cols (list, optional): List of target columns. Defaults to ['pm25', 'pm10'].
            
        Returns:
            tuple: (X, y) DataFrames for features and targets.
        """
        if target_cols is None:
            target_cols = ['pm25', 'pm10']
        
        if self.fs is None:
            logger.error("Not connected to Hopsworks feature store")
            return None, None
        
        try:
            feature_view = self.fs.get_feature_view(
                name=feature_view_name,
                version=version
            )
            
            td = feature_view.get_training_data()
            
            X = td.drop(columns=target_cols, errors='ignore')
            y = td[target_cols]
            
            return X, y
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None, None
