import pandas as pd
import numpy as np
import joblib
import logging
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, name=None, target_col=None):
        """Initialize the base model.
        
        Args:
            name (str, optional): Model name. Defaults to derived class name.
            target_col (str, optional): Target column. Defaults to None.
        """
        self.name = name or self.__class__.__name__
        self.target_col = target_col
        self.model = None
        self.feature_importances_ = None
    
    @abstractmethod
    def train(self, X, y):
        """Train the model.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
            
        Returns:
            object: Trained model.
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions with the model.
        
        Args:
            X (pd.DataFrame): Features.
            
        Returns:
            np.array: Predictions.
        """
        pass
    
    def save(self, path=None):
        """Save the model to disk.
        
        Args:
            path (str, optional): Path to save the model. Defaults to "{name}_{timestamp}.joblib".
            
        Returns:
            str: Path where the model was saved or None if saving failed.
        """
        if self.model is None:
            logger.error("No model to save")
            return None
        
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            path = f"{self.name}_{timestamp}.joblib"
        
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
    
    def load(self, path):
        """Load the model from disk.
        
        Args:
            path (str): Path to the model file.
            
        Returns:
            object: Loaded model or None if loading failed.
        """
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def get_feature_importances(self):
        """Get feature importances if available.
        
        Returns:
            pd.DataFrame: DataFrame with feature importances or None.
        """
        if self.feature_importances_ is None and hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        if self.feature_importances_ is not None:
            return pd.DataFrame({
                'feature': self.feature_names_,
                'importance': self.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return None
