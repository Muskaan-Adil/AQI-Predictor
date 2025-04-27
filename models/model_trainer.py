import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
from utils.config import Config
from models.linear_model import LinearModel
from models.forest_model import ForestModel
from models.boosting_model import BoostingModel
from models.time_series_model import TimeSeriesModel
from models.neural_net_model import NeuralNetModel
from evaluation.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Class for training and evaluating multiple models with cross-validation."""
    
    def __init__(self, target_col=None, n_splits=5, random_state=42):
        """Initialize the model trainer.
        
        Args:
            target_col (str, optional): Target column for prediction. Defaults to None.
            n_splits (int, optional): Number of CV splits. Defaults to 5.
            random_state (int, optional): Random seed. Defaults to 42.
        """
        self.target_col = target_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.model_metrics = {}
        self.feature_importances = {}
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _initialize_models(self, target_col):
        """Initialize all model classes."""
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
        """Train models with train-test split."""
        if isinstance(y, pd.DataFrame):
            target_col = self.target_col if self.target_col in y.columns else y.columns[0]
            self.target_col = target_col
        else:
            target_col = self.target_col
        
        logger.info(f"Training models for target: {target_col}")
        self.models = self._initialize_models(target_col)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state)
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name} model...")
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                y_true = y_test[target_col] if isinstance(y_test, pd.DataFrame) else y_test
                
                metrics = calculate_metrics(y_true, y_pred)
                self.model_metrics[name] = metrics
                
                if importance := model.get_feature_importances():
                    self.feature_importances[name] = importance
                
                logger.info(f"{name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                self.model_metrics[name] = {
                    'rmse': float('inf'), 
                    'mae': float('inf'), 
                    'r2': -float('inf')
                }
        
        best_model_name = min(self.model_metrics, key=lambda k: self.model_metrics[k]['rmse'])
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model: {best_model_name}")
        return self.models, self.model_metrics, best_model_name

    def cross_validate_models(self, X, y, models):
        """Perform cross-validation for specified models."""
        results = {}
        for name, model_type in models.items():
            try:
                logger.info(f"CV for {name}...")
                model = self._initialize_models(self.target_col)[name]
                cv_scores = {'rmse': [], 'mae': [], 'r2': []}
                
                for train_idx, test_idx in self.cv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.train(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = calculate_metrics(y_test, y_pred)
                    
                    for metric in cv_scores:
                        cv_scores[metric].append(metrics[metric])
                
                results[name] = {
                    'model': model,
                    'metrics': {k: np.mean(v) for k, v in cv_scores.items()},
                    'cv_scores': cv_scores
                }
                logger.info(f"{name} CV RMSE: {results[name]['metrics']['rmse']:.4f}")
            except Exception as e:
                logger.error(f"CV failed for {name}: {e}")
        return results

    def select_best_model(self, results):
        """Select best model from CV results."""
        if not results:
            return None
        best_name = min(results, key=lambda k: results[k]['metrics']['rmse'])
        return {
            'name': best_name,
            'model': results[best_name]['model'],
            'metrics': results[best_name]['metrics'],
            'cv_scores': results[best_name]['cv_scores']
        }

    def predict(self, X, model_name=None):
        """Make predictions using specified or best model."""
        model = (self.models[model_name] if model_name else self.best_model)
        if not model:
            logger.error("No model available")
            return None
        return model.predict(X)

    def get_best_model(self):
        """Get best model info."""
        if not self.best_model:
            return None, None, None
        best_name = min(self.model_metrics, key=lambda k: self.model_metrics[k]['rmse'])
        return self.best_model, best_name, self.model_metrics[best_name]

    def get_all_metrics(self):
        """Get metrics for all models."""
        if not self.model_metrics:
            return pd.DataFrame()
        return pd.DataFrame(self.model_metrics).T.reset_index().rename(columns={'index':'model'})

    def get_feature_importances(self, model_name=None):
        """Get feature importances."""
        if model_name in self.feature_importances:
            return self.feature_importances[model_name]
        best_name = min(self.model_metrics, key=lambda k: self.model_metrics[k]['rmse'])
        return self.feature_importances.get(best_name, pd.DataFrame())
