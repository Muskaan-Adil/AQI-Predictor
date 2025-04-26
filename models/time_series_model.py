import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class TimeSeriesModel(BaseModel):
    """ARIMA/SARIMA time series model implementation."""
    
    def __init__(self, name="SARIMA", target_col=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        """Initialize the time series model.
        
        Args:
            name (str, optional): Model name. Defaults to "SARIMA".
            target_col (str, optional): Target column. Defaults to None.
            order (tuple, optional): ARIMA order (p,d,q). Defaults to (1,1,1).
            seasonal_order (tuple, optional): Seasonal order (P,D,Q,s). Defaults to (0,0,0,0).
        """
        super().__init__(name=name, target_col=target_col)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.exog_columns = None
    
    def train(self, X, y):
        """Train the time series model.
        
        Args:
            X (pd.DataFrame): Features (only used for exogenous variables).
            y (pd.Series or pd.DataFrame): Target time series.
            
        Returns:
            object: Trained model.
        """
        if isinstance(y, pd.DataFrame):
            if self.target_col:
                y = y[self.target_col]
            else:
                self.target_col = y.columns[0]
                y = y[self.target_col]
        
        self.exog_columns = X.columns.tolist()
        exog = X if not X.empty else None
        
        try:
            logger.info(f"Training {self.name} model...")
            
            self.model = SARIMAX(
                y,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            
            fit_result = self.model.fit(disp=False)
            
            logger.info(f"Trained {self.name} model successfully")
            return fit_result
        except Exception as e:
            logger.error(f"Failed to train {self.name} model: {e}")
            return None
    
    def predict(self, X, steps=1):
        """Make predictions with the time series model.
        
        Args:
            X (pd.DataFrame): Features for exogenous variables.
            steps (int, optional): Number of steps to forecast. Defaults to 1.
            
        Returns:
            np.array: Predictions.
        """
        if self.model is None:
            logger.error("Model not trained")
            return None
        
        try:
            exog = X[self.exog_columns] if self.exog_columns and all(col in X for col in self.exog_columns) else None
            
            result = self.model.fit(disp=False)
            
            forecast = result.forecast(steps=steps, exog=exog)
            return forecast
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            return None
