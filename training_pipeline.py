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

# Suppress TensorFlow warnings
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
    """Load cities from config file."""
    yaml_path = 'cities.yaml'
    default_cities = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
    ]
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                if data := yaml.safe_load(f):
                    if isinstance(data.get('cities'), list):
                        logger.info(f"Loaded {len(data['cities'])} cities")
                        return data['cities']
        logger.warning("Using default cities")
        return default_cities
    except Exception as e:
        logger.error(f"Error loading cities: {e}")
        return default_cities

def validate_features(X: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Clean and validate features."""
    # Remove target if present
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    
    # Remove duplicates and constants
    X = X.loc[:, ~X.columns.duplicated()]
    constant_cols = X.columns[X.nunique() == 1]
    if not constant_cols.empty:
        logger.warning(f"Removing constant columns: {list(constant_cols)}")
        X = X.drop(columns=constant_cols)
    
    return X

def preprocess_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess data with robust handling."""
    try:
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')
        
        if y.isna().all():
            raise ValueError("Invalid target values")
            
        # Handle missing data
        empty_cols = X.columns[X.isna().all()]
        if not empty_cols.empty:
            logger.warning(f"Dropping empty columns: {list(empty_cols)}")
            X = X.drop(columns=empty_cols)
        
        # Impute and scale
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y = y.fillna(y.mean())
        
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        return X, y
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None, None

def get_training_data(feature_store: FeatureStore,
                     feature_view_name: str,
                     target_col: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Retrieve and prepare training data."""
    try:
        fv = feature_store.get_feature_view(feature_view_name, version=1)
        if not fv:
            raise ValueError("Feature view not found")
            
        training_data = fv.get_training_data(training_dataset_version=1)
        if not training_data:
            raise ValueError("No training data")
        
        X = validate_features(training_data[0], target_col)
        y = training_data[1][target_col]
        return preprocess_data(X, y)
    except Exception as e:
        logger.error(f"Data retrieval failed: {e}")
        return None, None

def run_training_pipeline() -> None:
    """Enhanced training pipeline with standardized naming and auto-deployment."""
    logger.info("Starting enhanced training pipeline...")
    
    try:
        cities = load_cities()
        feature_store = FeatureStore()
        model_registry = ModelRegistry()
        
        for city in cities:
            # Standardize city name (lowercase, underscore)
            city_name = city['name'].lower().replace(' ', '_')
            logger.info(f"Processing {city_name}")
            
            for target in ['pm25', 'pm10']:
                target_col = target.lower()
                logger.info(f"Training {target_col} model")
                
                # Standardized feature view name
                fv_name = f"{city_name}_aqi_features"
                X, y = get_training_data(feature_store, fv_name, target_col)
                
                if X is None or y is None or X.empty or y.empty:
                    logger.warning(f"Skipping {city_name} - {target_col} due to data issues")
                    continue
                
                # Initialize trainer with validation checks
                trainer = ModelTrainer(
                    target_col=target_col,
                    n_splits=5,
                    random_state=42
                )
                
                model_config = {
                    'linear': 'linear',
                    'ridge': 'ridge',
                    'lasso': 'lasso',
                    'random_forest': 'random_forest',
                    'gradient_boosting': 'gradient_boosting'
                }
                
                # Run cross-validation
                results = trainer.cross_validate_models(X, y, model_config)
                best_model = trainer.select_best_model(results)
                
                if best_model:
                    # Standardized model name
                    model_name = f"{city_name}_{target_col}"
                    
                    # Check for suspiciously good performance
                    if best_model['metrics']['rmse'] < 0.01:
                        logger.warning(f"Potentially overfit model: {model_name} with RMSE {best_model['metrics']['rmse']:.4f}")
                    
                    # Save with comprehensive metrics
                    version = model_registry.save_model(
                        model=best_model['model'].model,
                        name=model_name,
                        metrics=best_model['metrics'],
                        tags={
                            'city': city_name,
                            'target': target_col,
                            'model_type': best_model['name'],
                            'deployable': 'true',
                            'cv_folds': str(trainer.n_splits)
                        },
                        description=f"Best {target_col} model for {city_name} (CV RMSE: {best_model['metrics']['rmse']:.4f} ± {best_model['metrics']['rmse_std']:.4f})"
                    )
                    
                    # Auto-deploy if version was saved successfully
                    if version:
                        try:
                            deployment = model_registry.deploy_model(
                                model_name=model_name,
                                version=version,
                                deployment_name=f"{model_name}_deployment"
                            )
                            logger.info(f"Deployed {model_name} version {version}")
                        except Exception as e:
                            logger.error(f"Deployment failed: {str(e)}")
                    
                    # Feature importance analysis for tree-based models
                    if best_model['name'] in ['random_forest', 'gradient_boosting']:
                        try:
                            analyzer = FeatureImportanceAnalyzer(best_model['model'].model, X)
                            analyzer.generate_report()
                            # Add visualization if available
                            if hasattr(analyzer, 'plot_importance'):
                                analyzer.plot_importance(top_n=10)
                        except Exception as e:
                            logger.error(f"Feature importance analysis failed: {str(e)}")
        
        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    run_training_pipeline()
