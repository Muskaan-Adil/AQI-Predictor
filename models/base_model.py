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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from feature_engineering.feature_store import FeatureStore
from models.model_registry import ModelRegistry
from evaluation.feature_importance import FeatureImportanceAnalyzer

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

# Remove the run_training_pipeline() function - it should be in training_pipeline.py
