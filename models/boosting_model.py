import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import logging
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class BoostingModel(BaseModel):
    """Gradient Boosting regression model implementation."""
    
    def __init__(self, name="GradientBoosting", target_col=None, n_estimators=100, learning_rate=0.1, max_depth=3):
        """Initialize the Gradient Boosting model.
        
        Args:
            name (str, optional): Model name. Defaults to "GradientBoosting".
            target_col (str, optional): Target column. Defaults to None.
            n_estimators (int, optional): Number of boosting stages. Defaults to 100.
            learning_rate (float, optional): Learning rate. Defaults to 0.1.
            max_depth (int, optional): Maximum tree depth. Defaults to 3.
        """
        super().__init__(name=name, target_col=target_col)
        self.feature_names_ = None
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    def train(self, X, y):
        """Train the Gradient Boosting model.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series or pd.DataFrame): Target(s).
            
        Returns:
            object: Trained model.
        """
        self.feature_names_ = X.columns.tolist()
        
        if isinstance(y, pd.DataFrame):
            if self.target_col:
                y = y[self.target_col]
            else:
                self.target_col = y.columns[0]
                y = y[self.target_col]
        
        try:
            logger.info(f"Training {self.name} model...")
            self.model.fit(X, y)
            logger.info(f"Trained {self.name} model successfully")
            
            self.feature_importances_ = self.model.feature_importances_
            
            return self.model
        except Exception as e:
            logger.error(f"Failed to train {self.name} model: {e}")
            return None
    
    def predict(self, X):
        """Make predictions with the Gradient Boosting model.
        
        Args:
            X (pd.DataFrame): Features.
            
        Returns:
            np.array: Predictions.
        """
        if self.model is None:
            logger.error("Model not trained")
            return None
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            return None
