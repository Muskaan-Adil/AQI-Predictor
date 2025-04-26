import logging
import os
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import your actual FeatureStore class from its module
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

def get_training_data(feature_store: FeatureStore,
                    feature_view_name: str,
                    target_cols: List[str]
                   ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Fetch training data (features X and target y) from a Hopsworks Feature View.
    
    Args:
        feature_store: Initialized FeatureStore instance
        feature_view_name: Name of the feature view
        target_cols: List of target column names
        
    Returns:
        Tuple of (features DataFrame, target Series)
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
        
        # Split into features and target
        X = training_data.drop(target_cols, axis=1)
        y = training_data[target_cols[0]]  # Using first target column
        
        logger.info(f"Retrieved training data with {X.shape[0]} samples and {X.shape[1]} features")
        
        return X, y
    except Exception as e:
        logger.error(f"Failed to get training data from '{feature_view_name}': {e}")
        return None, None

def run_training_pipeline() -> None:
    """Run the model training pipeline."""
    logger.info("Starting training pipeline...")
    
    try:
        cities = load_cities()
        
        # Initialize connections
        feature_store = FeatureStore()
        model_registry = ModelRegistry()
        
        for city in cities:
            city_name = city['name']
            logger.info(f"Training models for {city_name}...")
            
            for target_col in ['pm25', 'pm10']:
                logger.info(f"Training models for {city_name} - {target_col}")
                
                feature_view_name = f"{city_name.lower().replace(' ', '_')}_aqi_features"
                
                # Get training data
                X, y = get_training_data(feature_store, feature_view_name, [target_col])
                
                if X is None or y is None or X.empty or y.empty:
                    logger.warning(f"No training data available for {city_name} - {target_col}")
                    continue
                
                # Train models
                model_trainer = ModelTrainer(target_col=target_col)
                models, metrics, best_model_name = model_trainer.train_models(X, y)
                
                if not best_model_name:
                    logger.warning(f"No best model found for {city_name} - {target_col}")
                    continue
                
                logger.info(f"Best model for {city_name} - {target_col}: {best_model_name}")
                
                # Save best model
                best_model = models[best_model_name]
                best_metrics = metrics[best_model_name]
                
                # Feature importance analysis
                try:
                    analyzer = FeatureImportanceAnalyzer(best_model.model, X=X)
                    analyzer.generate_explainer()
                    analyzer.calculate_shap_values()
                    importance_df = analyzer.get_feature_importance_df()
                except Exception as e:
                    logger.error(f"Error in feature importance analysis: {e}")
                
                # Save to model registry
                model_registry.save_model(
                    model=best_model.model,
                    name=f"{city_name.lower().replace(' ', '_')}_{target_col}",
                    metrics=best_metrics,
                    tags={
                        'city': city_name,
                        'target': target_col,
                        'model_type': best_model_name
                    },
                    description=f"Best model for {city_name} - {target_col} ({best_model_name})"
                )
                
                logger.info(f"Saved best model for {city_name} - {target_col} to registry")
        
        logger.info("Training pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_training_pipeline()
