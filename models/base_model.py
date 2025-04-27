import logging
import os
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from feature_engineering.feature_store import FeatureStore
from models.model_registry import ModelRegistry
from evaluation.feature_importance import FeatureImportanceAnalyzer

class BaseModel(ABC):
    """Abstract base class for all machine learning models."""
    
    def __init__(self, name: str, target_col: str = None):
        """
        Initialize the base model.
        
        Args:
            name (str): Name of the model
            target_col (str, optional): Target column name. Defaults to None.
        """
        self.name = name
        self.target_col = target_col
        self.model = None
        self.feature_importances_ = None
        self.feature_names_ = None
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Make predictions using the trained model."""
        pass
    
    def get_feature_importances(self):
        """Get feature importances if available.
        
        Returns:
            np.array or None: Feature importance scores
        """
        return self.feature_importances_
    
    def save(self, path: str):
        """Save the model to disk."""
        raise NotImplementedError("Save method not implemented")
    
    def load(self, path: str):
        """Load the model from disk."""
        raise NotImplementedError("Load method not implemented")

def load_cities() -> List[dict]:
    """Load cities from YAML configuration file."""
    yaml_path = 'cities.yaml'
    
    default_cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
    ]
    
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                if data and 'cities' in data and isinstance(data['cities'], list):
                    logger.info(f"Loaded {len(data['cities'])} cities from YAML")
                    return data['cities']
        logger.warning("Cities YAML not found, using default cities")
        return default_cities
    except Exception as e:
        logger.error(f"Error loading cities from YAML: {e}")
        return default_cities

def preprocess_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data by handling missing values and scaling."""
    try:
        # Convert all columns to numeric, coercing errors to NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Identify and drop completely empty columns
        empty_cols = X.columns[X.isna().all()]
        if len(empty_cols) > 0:
            logger.warning(f"Dropping empty columns: {list(empty_cols)}")
            X = X.drop(columns=empty_cols)
        
        # Create preprocessing pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        X = pd.DataFrame(X_processed, columns=X.columns, index=X.index)
        
        # Handle target variable
        y = y.fillna(y.mean())
        
        logger.info(f"Preprocessed data shape: {X.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        return None, None

def get_training_data(feature_store: FeatureStore,
                     feature_view_name: str,
                     target_cols: List[str]
                    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Fetch and preprocess training data from feature store.
    """
    try:
        # Get the feature view
        feature_view = feature_store.get_feature_view(
            name=feature_view_name,
            version=1
        )
        
        if feature_view is None:
            logger.error(f"Feature view '{feature_view_name}' not found")
            return None, None
        
        # Get training data
        training_data = feature_view.get_training_data(
            training_dataset_version=1
        )
        
        if training_data is None:
            logger.error(f"No training data available for feature view '{feature_view_name}'")
            return None, None
        
        # Split tuple and preprocess
        X = training_data[0]   # Features
        y = training_data[1][target_cols[0]]  # First target column
        
        logger.info(f"Raw data shape before preprocessing: {X.shape}")
        
        # Preprocess data
        X, y = preprocess_data(X, y)
        
        if X is None or y is None:
            logger.error("Data preprocessing failed")
            return None, None
        
        logger.info(f"Retrieved training data with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y
    
    except Exception as e:
        logger.error(f"Failed to get training data from '{feature_view_name}': {e}")
        return None, None

def clean_metrics(metrics: dict) -> dict:
    """Replace infinite values and handle edge cases for model registry."""
    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if np.isinf(v) or np.isnan(v):
                cleaned[k] = 1e6 if v > 0 else -1e6
            else:
                cleaned[k] = float(v)
        else:
            cleaned[k] = str(v)
    return cleaned
