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

# Add the BaseModel class definition
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

# Keep all your existing utility functions below
def load_cities() -> List[dict]:
    """Load cities from YAML configuration file."""
    # ... (keep existing implementation)

def preprocess_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data by handling missing values and scaling."""
    # ... (keep existing implementation)

def get_training_data(feature_store: FeatureStore,
                     feature_view_name: str,
                     target_cols: List[str]
                    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Fetch and preprocess training data from feature store.
    """
    # ... (keep existing implementation)

def clean_metrics(metrics: dict) -> dict:
    """Replace infinite values and handle edge cases for model registry."""
    # ... (keep existing implementation)
