import numpy as np
import pandas as pd
import logging
from src.utils.config import Config
from src.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelSelector:
    """Class for selecting the best model for predictions."""
    
    def __init__(self):
        """Initialize the model selector."""
        self.model_registry = ModelRegistry()
        self.best_models = {}
    
    def get_best_model(self, city, target_col):
        """Get the best model for a city and target column.
        
        Args:
            city (str): City name.
            target_col (str): Target column.
            
        Returns:
            object: Best model or None if not found.
        """
        cache_key = f"{city}_{target_col}"
        if cache_key in self.best_models:
            return self.best_models[cache_key]
        
        best_model = self.model_registry.get_best_model(city, target_col)
        
        if best_model:
            self.best_models[cache_key] = best_model
        
        return best_model
    
    def get_best_models_for_all_cities(self, target_col):
        """Get the best models for all cities for a target column.
        
        Args:
            target_col (str): Target column.
            
        Returns:
            dict: Dictionary mapping city names to best models.
        """
        city_models = {}
        
        for city in Config.CITIES:
            city_name = city['name']
            model = self.get_best_model(city_name, target_col)
            
            if model:
                city_models[city_name] = model
            else:
                logger.warning(f"No model found for city: {city_name}, target: {target_col}")
        
        return city_models
    
    def predict_with_best_model(self, city, target_col, X):
        """Make predictions using the best model for a city.
        
        Args:
            city (str): City name.
            target_col (str): Target column.
            X (pd.DataFrame): Features.
            
        Returns:
            np.array: Predictions or None if no model found.
        """
        model = self.get_best_model(city, target_col)
        
        if model is None:
            logger.error(f"No model found for city: {city}, target: {target_col}")
            return None
        
        try:
            return model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions with best model: {e}")
            return None
