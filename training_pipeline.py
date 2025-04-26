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
        
        # Identify and drop non-numeric columns that couldn't be converted
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Dropping non-numeric columns: {list(non_numeric_cols)}")
            X = X.drop(columns=non_numeric_cols)
        
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
                
                # Feature importance analysis (skip for linear models)
                if best_model_name in ['random_forest', 'gradient_boosting']:
                    try:
                        analyzer = FeatureImportanceAnalyzer(best_model.model, X=X)
                        analyzer.generate_explainer()
                        analyzer.calculate_shap_values()
                        importance_df = analyzer.get_feature_importance_df()
                    except Exception as e:
                        logger.error(f"Error in feature importance analysis: {e}")
                
                # Save to model registry
                try:
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
                except Exception as e:
                    logger.error(f"Failed to save model to registry: {e}")
        
        logger.info("Training pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    run_training_pipeline()
