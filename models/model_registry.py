import pandas as pd
import numpy as np
import os
import json
import logging
import hopsworks
import joblib
from datetime import datetime
from src.utils.config import Config

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Class for model registry operations."""
    
    def __init__(self, project_name=None, api_key=None):
        """Initialize the model registry.
        
        Args:
            project_name (str, optional): Hopsworks project name. Defaults to Config value.
            api_key (str, optional): Hopsworks API key. Defaults to Config value.
        """
        self.project_name = project_name or Config.HOPSWORKS_PROJECT_NAME
        self.api_key = api_key or Config.HOPSWORKS_API_KEY
        self.model_registry_name = Config.MODEL_REGISTRY_NAME
        self.project = None
        self.mr = None
        
        self._connect()
    
    def _connect(self):
        """Connect to Hopsworks project and model registry."""
        try:
            self.project = hopsworks.login(
                project=self.project_name,
                api_key_value=self.api_key
            )
            self.mr = self.project.get_model_registry()
            logger.info(f"Connected to Hopsworks model registry")
        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            self.project = None
            self.mr = None
    
    def save_model(self, model, model_name, metrics=None, city=None, target_col=None):
        """Save a model to the registry.
        
        Args:
            model: Model object to save.
            model_name (str): Name of the model.
            metrics (dict, optional): Model metrics. Defaults to None.
            city (str, optional): City name for the model. Defaults to None.
            target_col (str, optional): Target column for the model. Defaults to None.
            
        Returns:
            str: Model path in the registry or None if saving failed.
        """
        if self.mr is None:
            logger.error("Not connected to Hopsworks model registry")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        full_model_name = f"{model_name}_{target_col or 'unknown'}_{city or 'all'}_{timestamp}"
        
        try:
            temp_dir = f"/tmp/{full_model_name}"
            os.makedirs(temp_dir, exist_ok=True)
            
            if hasattr(model, 'save'):
                model_path = model.save(os.path.join(temp_dir, "model"))
            else:
                model_path = os.path.join(temp_dir, "model.joblib")
                joblib.dump(model, model_path)
            
            if metrics:
                metrics_path = os.path.join(temp_dir, "metrics.json")
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f)
            
            model_schema = {
                "name": full_model_name,
                "version": 1,
                "description": f"Model for {target_col or 'unknown'} prediction for {city or 'all cities'}",
                "model_type": model_name,
                "target_column": target_col or "unknown",
                "city": city or "all",
                "metrics": metrics or {},
                "timestamp": timestamp
            }
            
            model_dir = self.mr.python.create_model(
                name=full_model_name,
                metrics=metrics,
                model_schema=model_schema,
                description=f"Model for {target_col or 'unknown'} prediction"
            )
            
            model_dir.save(temp_dir)
            
            logger.info(f"Model saved to registry: {full_model_name}")
            return full_model_name
        except Exception as e:
            logger.error(f"Failed to save model to registry: {e}")
            return None
    
    def load_model(self, model_name, version=None):
        """Load a model from the registry.
        
        Args:
            model_name (str): Name of the model.
            version (int, optional): Model version. Defaults to latest.
            
        Returns:
            object: Loaded model or None if loading failed.
        """
        if self.mr is None:
            logger.error("Not connected to Hopsworks model registry")
            return None
        
        try:
            model = self.mr.get_model(name=model_name, version=version)
            
            model_dir = model.download()
            
            loaded_model = joblib.load(os.path.join(model_dir, "model.joblib"))
            
            logger.info(f"Model loaded from registry: {model_name}")
            return loaded_model
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            return None
    
    def get_best_model(self, city=None, target_col=None):
        """Get the best model for a city and target column.
        
        Args:
            city (str, optional): City name. Defaults to None.
            target_col (str, optional): Target column. Defaults to None.
            
        Returns:
            object: Best model or None if not found.
        """
        if self.mr is None:
            logger.error("Not connected to Hopsworks model registry")
            return None
        
        try:
            models = self.mr.get_models()
            
            filtered_models = []
            for model in models:
                model_info = model.to_dict()
                model_schema = model_info.get('model_schema', {})
                
                if model_schema:
                    schema_city = model_schema.get('city')
                    schema_target = model_schema.get('target_column')
                    
                    if (city is None or schema_city == city) and \
                       (target_col is None or schema_target == target_col):
                        filtered_models.append(model)
            
            if not filtered_models:
                logger.warning(f"No models found for city: {city}, target: {target_col}")
                return None
            
            best_model = None
            best_rmse = float('inf')
            
            for model in filtered_models:
                metrics = model.get_metrics()
                if metrics and 'rmse' in metrics and metrics['rmse'] < best_rmse:
                    best_model = model
                    best_rmse = metrics['rmse']
            
            if best_model:
                logger.info(f"Best model found: {best_model.name}, RMSE: {best_rmse}")
                return self.load_model(best_model.name)
            else:
                logger.warning("No model with valid metrics found")
                return None
        except Exception as e:
            logger.error(f"Failed to get best model: {e}")
            return None
