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

# Suppress TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from feature_engineering.feature_store import FeatureStore
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry
from evaluation.feature_importance import FeatureImportanceAnalyzer

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

def validate_features(X: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Validate and clean feature set."""
    # Remove target if present in features
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    
    # Remove duplicate columns
    X = X.loc[:, ~X.columns.duplicated()]
    
    # Remove constant columns
    constant_cols = X.columns[X.nunique() == 1]
    if constant_cols.any():
        logger.warning(f"Removing constant columns: {list(constant_cols)}")
        X = X.drop(columns=constant_cols)
    
    return X

def preprocess_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data with enhanced validation."""
    try:
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        # Check for valid target
        if y.isna().all():
            raise ValueError("All target values are missing/invalid")
            
        # Remove empty features
        empty_cols = X.columns[X.isna().all()]
        if empty_cols.any():
            logger.warning(f"Dropping empty columns: {list(empty_cols)}")
            X = X.drop(columns=empty_cols)
        
        # Handle remaining missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y = y.fillna(y.mean())
        
        # Scale features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return None, None

def get_training_data(feature_store: FeatureStore,
                     feature_view_name: str,
                     target_col: str
                    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Get and validate training data."""
    try:
        feature_view = feature_store.get_feature_view(feature_view_name, version=1)
        training_data = feature_view.get_training_data(training_dataset_version=1)
        
        X = training_data[0]
        y = training_data[1][target_col]
        
        # Validate features
        X = validate_features(X, target_col)
        X, y = preprocess_data(X, y)
        
        return X, y
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        return None, None

def run_training_pipeline() -> None:
    """Main training workflow with enhanced validation."""
    logger.info("Starting training pipeline...")
    
    try:
        cities = load_cities()
        feature_store = FeatureStore()
        model_registry = ModelRegistry()
        
        for city in cities:
            city_name = city['name']
            logger.info(f"Processing {city_name}")
            
            for target in ['pm25', 'pm10']:
                logger.info(f"Training {target} model")
                
                fv_name = f"{city_name.lower().replace(' ', '_')}_aqi_features"
                X, y = get_training_data(feature_store, fv_name, target)
                
                if X is None or y is None:
                    continue
                
                # Initialize trainer with cross-validation
                trainer = ModelTrainer(
                    target_col=target,
                    n_splits=5,
                    random_state=42
                )
                
                models = {
                    'linear': 'linear',
                    'ridge': 'ridge',
                    'lasso': 'lasso',
                    'random_forest': 'forest',
                    'gradient_boosting': 'boosting'
                }
                
                results = trainer.cross_validate_models(X, y, models)
                best_model = trainer.select_best_model(results)
                
                if best_model:
                    # Save model with metadata
                    model_registry.save_model(
                        model=best_model['model'],
                        name=f"{city_name}_{target}",
                        metrics=best_model['metrics'],
                        tags={
                            'city': city_name,
                            'target': target,
                            'model_type': best_model['name']
                        }
                    )
                    
                    # Feature importance for tree-based models
                    if best_model['name'] in ['random_forest', 'gradient_boosting']:
                        analyzer = FeatureImportanceAnalyzer(best_model['model'], X)
                        analyzer.generate_report()
        
        logger.info("Training pipeline completed")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_training_pipeline()
