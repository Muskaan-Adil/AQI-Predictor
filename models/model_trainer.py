import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from src.utils.config import Config
from src.models.linear_model import LinearModel
from src.models.forest_model import ForestModel
from src.models.boosting_model import BoostingModel
from src.models.time_series_model import TimeSeriesModel
from src.models.neural_net_model import NeuralNetModel
from src.evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating multiple models."""
    
    def __init__(self, target_col=None):
        """Initialize the model trainer.
        
        Args:
            target_col (str, optional): Target column for prediction. Defaults to None.
        """
        self.target_col = target_col
        self.models = {}
        self.best_model = None
        self.model_metrics = {}
        self.feature_importances = {}
    
    def _initialize_models(self, target_col):
        """Initialize all model classes.
        
        Args:
            target_col (str): Target column for prediction.
            
        Returns:
            dict: Dictionary of model objects.
        """
        models = {
            'linear': LinearModel(name="Linear", target_col=target_col),
            'ridge': LinearModel(name="Ridge", target_col=target_col, model_type='ridge'),
            'lasso': LinearModel(name="Lasso", target_col=target_col, model_type='lasso'),
            'random_forest': ForestModel(name="RandomForest", target_col=target_col),
            'gradient_boosting': BoostingModel(name="GradientBoosting", target_col=target_col),
            'sarima': TimeSeriesModel(name="SARIMA", target_col=target_col),
            'neural_net': NeuralNetModel(name="NeuralNetwork", target_col=target_col)
        }
        
        return models
    
    def train_models(self, X, y, test_size=0.2):
        """Train multiple regression models and evaluate their performance.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.DataFrame or pd.Series): Target(s).
            test_size (float, optional): Size of test set. Defaults to 0.2.
            
        Returns:
            tuple: (trained_models, metrics, best_model_name).
        """
        if self.target_col is None:
            if isinstance(y, pd.DataFrame):
                self.target_col = y.columns[0]
            else:
                self.target_col = 'target'
        
        if isinstance(y, pd.DataFrame) and self.target_col in y.columns:
            target = y[self.target_col]
        else:
            target = y
        
        self.models = self._initialize_models(self.target_col)
        
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=test_size, random_state=42)
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training model: {name}")
                
                model.train(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = calculate_metrics(y_test, y_pred)
                self.model_metrics[name] = metrics
                
                importances = model.get_feature_importances()
                if importances is not None:
                    self.feature_importances[name] = importances
                
                logger.info(f"Model {name} trained. Test RMSE: {metrics['rmse']:.4f}")
            except Exception as e:
                logger.error(f"Error training model {name}: {e}")
        
        if self.model_metrics:
            best_model_name = min(self.model_metrics, key=lambda x: self.model_metrics[x]['rmse'])
            self.best_model = best_model_name
            logger.info(f"Best model: {best_model_name} with RMSE: {self.model_metrics[best_model_name]['rmse']:.4f}")
        else:
            logger.warning("No models were successfully trained")
        
        return self.models, self.model_metrics, self.best_model
    
    def get_best_model(self):
        """Get the best performing model.
        
        Returns:
            tuple: (best_model_name, model_object) or (None, None) if no best model.
        """
        if self.best_model and self.best_model in self.models:
            return self.best_model, self.models[self.best_model]
        return None, None
    
    def predict_with_best_model(self, X):
        """Make predictions using the best model.
        
        Args:
            X (pd.DataFrame): Features.
            
        Returns:
            np.array: Predictions or None if no best model.
        """
        best_model_name, best_model = self.get_best_model()
        
        if best_model is None:
            logger.error("No best model available for prediction")
            return None
        
        try:
            return best_model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions with best model: {e}")
            return None
    
    def get_feature_importance_for_best_model(self):
        """Get feature importances for the best model.
        
        Returns:
            pd.DataFrame: Feature importances or None if not available.
        """
        if self.best_model and self.best_model in self.feature_importances:
            return self.feature_importances[self.best_model]
        return None
