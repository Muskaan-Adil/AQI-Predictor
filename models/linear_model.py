import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import logging
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class LinearModel(BaseModel):
    """Linear regression model implementation."""
    
    def __init__(self, name="LinearRegression", target_col=None, model_type='linear'):
        """Initialize the linear model.
        
        Args:
            name (str, optional): Model name. Defaults to "LinearRegression".
            target_col (str, optional): Target column. Defaults to None.
            model_type (str, optional): Type of linear model ('linear', 'ridge', 'lasso'). Defaults to 'linear'.
        """
        super().__init__(name=name, target_col=target_col)
        self.model_type = model_type
        self.feature_names_ = None
        
        if model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=0.1)
        else:
            self.model = LinearRegression()
    
    def train(self, X, y):
        """Train the linear model.
        
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
            
            self.feature_importances_ = np.abs(self.model.coef_) if hasattr(self.model, 'coef_') else None
            
            return self.model
        except Exception as e:
            logger.error(f"Failed to train {self.name} model: {e}")
            return None
    
    def predict(self, X):
        """Make predictions with the linear model.
        
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
