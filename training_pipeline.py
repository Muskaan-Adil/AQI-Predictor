import logging
import os
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from feature_engineering.feature_store import FeatureStore
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry
from evaluation.feature_importance import FeatureImportanceAnalyzer

def load_cities():
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

class FeatureStore:
    def __init__(self):
        self.fs = None  # Initialize your connection to Hopsworks or Feature Store here
    
def get_training_data(self,
                      feature_view_name: str,
                      target_cols: List[str]
                     ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetch training data (features X and target y) from a Hopsworks Feature View.
    If the feature view does not exist, create it.
    """
    try:
        # 1) Try to load the feature view (version=1, adjust if you use another)
        fv = self.fs.get_feature_view(name=feature_view_name, version=1)
        
        # If feature view doesn't exist, create one
        if fv is None:
            logger.info(f"Feature view '{feature_view_name}' not found. Creating new feature view.")
            
            # You should have a predefined process for creating a new feature view.
            # This could be done by registering a new feature view. Assuming you have the necessary data and schema:
            feature_view = self.fs.create_feature_view(
                name=feature_view_name,
                description="Feature view for AQI data",
                entities=["city"],  # Specify your entity, e.g., "city"
                features=[  # Define the features based on your dataset
                    {"name": "pm25", "type": "float"},
                    {"name": "pm10", "type": "float"},
                    # Add other features you need here
                ],
                time_travel_format="version",  # Example, adjust accordingly
                time_travel_column="timestamp"  # Example, adjust accordingly
            )
            fv = feature_view  # Use the newly created feature view
        
        # 2) Use the SDK's training_data call (returns X, y)
        X, y = fv.training_data(target_name=target_cols[0])
        
        return X, y
    except Exception as e:
        logger.error(f"Failed to load training data from '{feature_view_name}': {e}")
        return None, None


def run_training_pipeline():
    """Run the model training pipeline."""
    logger.info("Starting training pipeline...")
    
    try:
        cities = load_cities()
        
        feature_store = FeatureStore()
        model_registry = ModelRegistry()
        
        for city in cities:
            city_name = city['name']
            logger.info(f"Training models for {city_name}...")
            
            for target_col in ['pm25', 'pm10']:
                logger.info(f"Training models for {city_name} - {target_col}")
                
                feature_view_name = f"{city_name.lower().replace(' ', '_')}_aqi_features"
                X, y = feature_store.get_training_data(feature_view_name, target_cols=[target_col])
                
                if X is None or y is None or X.empty or y.empty:
                    logger.warning(f"No training data available for {city_name} - {target_col}")
                    continue
                
                model_trainer = ModelTrainer(target_col=target_col)
                
                models, metrics, best_model_name = model_trainer.train_models(X, y)
                
                if best_model_name:
                    logger.info(f"Best model for {city_name} - {target_col}: {best_model_name}")
                    
                    best_model = models[best_model_name]
                    best_metrics = metrics[best_model_name]
                    
                    analyzer = FeatureImportanceAnalyzer(best_model.model, X=X)
                    analyzer.generate_explainer()
                    analyzer.calculate_shap_values()
                    importance_df = analyzer.get_feature_importance_df()
                    
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
                else:
                    logger.warning(f"No best model found for {city_name} - {target_col}")
        
        logger.info("Training pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_training_pipeline()
