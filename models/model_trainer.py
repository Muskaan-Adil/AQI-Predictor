# src/models/model_trainer.py (continued)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from utils.config import Config
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
        if isinstance(y, pd.DataFrame):
            if self.target_col and self.target_col in y.columns:
                target_col = self.target_col
            else:
                target_col = y.columns[0]
                self.target_col = target_col
        else:
            target_col = self.target_col
        
        logger.info(f"Training models for target: {target_col}")
        
        self.models = self._initialize_models(target_col)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name} model...")
                model.train(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                if isinstance(y_test, pd.DataFrame) and target_col:
                    y_true = y_test[target_col]
                else:
                    y_true = y_test
                
                metrics = calculate_metrics(y_true, y_pred)
                self.model_metrics[name] = metrics
                
                feature_importance = model.get_feature_importances()
                if feature_importance is not None:
                    self.feature_importances[name] = feature_importance
                
                logger.info(f"Trained {name} model - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
                self.model_metrics[name] = {'rmse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
        
        best_model_name = min(self.model_metrics, key=lambda k: self.model_metrics[k]['rmse'])
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with RMSE: {self.model_metrics[best_model_name]['rmse']:.4f}")
        
        return self.models, self.model_metrics, best_model_name
    
    def predict(self, X, model_name=None):
        """Make predictions using a specified model or the best model.
        
        Args:
            X (pd.DataFrame): Features for prediction.
            model_name (str, optional): Name of the model to use. Defaults to None (uses best model).
            
        Returns:
            np.array: Predictions.
        """
        if model_name and model_name in self.models:
            model = self.models[model_name]
        elif self.best_model:
            model = self.best_model
        else:
            logger.error("No model available for prediction")
            return None
        
        return model.predict(X)
    
    def get_best_model(self):
        """Get the best performing model.
        
        Returns:
            tuple: (best_model, best_model_name, metrics).
        """
        if not self.best_model:
            logger.warning("No best model available - models have not been trained yet")
            return None, None, None
        
        best_model_name = min(self.model_metrics, key=lambda k: self.model_metrics[k]['rmse'])
        best_metrics = self.model_metrics[best_model_name]
        
        return self.best_model, best_model_name, best_metrics
    
    def get_all_metrics(self):
        """Get metrics for all trained models.
        
        Returns:
            pd.DataFrame: DataFrame with metrics for all models.
        """
        if not self.model_metrics:
            logger.warning("No metrics available - models have not been trained yet")
            return pd.DataFrame()
        
        metrics_df = pd.DataFrame(self.model_metrics).T
        metrics_df.index.name = 'model'
        metrics_df = metrics_df.reset_index()
        
        return metrics_df
    
    def get_feature_importances(self, model_name=None):
        """Get feature importances for a specified model or the best model.
        
        Args:
            model_name (str, optional): Name of the model. Defaults to None (uses best model).
            
        Returns:
            pd.DataFrame: Feature importances.
        """
        if model_name and model_name in self.feature_importances:
            return self.feature_importances[model_name]
        
        best_model_name = min(self.model_metrics, key=lambda k: self.model_metrics[k]['rmse']) if self.model_metrics else None
        
        if best_model_name and best_model_name in self.feature_importances:
            return self.feature_importances[best_model_name]
        
        logger.warning("No feature importances available")
        return pd.DataFrame()
