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
    """Enhanced model trainer with complete metrics tracking."""
    
    def __init__(self, target_col=None, n_splits=5, random_state=42):
        self.target_col = target_col
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.model_metrics = {}
        self.feature_importances = {}
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    def _initialize_models(self, target_col):
        """Initialize models with standardized naming."""
        return {
            'linear': LinearModel(name="linear", target_col=target_col),
            'ridge': LinearModel(name="ridge", target_col=target_col, model_type='ridge'),
            'lasso': LinearModel(name="lasso", target_col=target_col, model_type='lasso', max_iter=5000),
            'random_forest': ForestModel(name="random_forest", target_col=target_col),
            'gradient_boosting': BoostingModel(name="gradient_boosting", target_col=target_col),
            'sarima': TimeSeriesModel(name="sarima", target_col=target_col),
            'neural_net': NeuralNetModel(name="neural_net", target_col=target_col)
        }


    def _log_metrics(self, name, metrics):
        """Enhanced metrics logging with all key metrics."""
        logger.info(
            f"{name} - RMSE: {metrics['rmse']:.4f} ± {metrics.get('rmse_std', 0):.4f}, "
            f"MAE: {metrics['mae']:.4f} ± {metrics.get('mae_std', 0):.4f}, "
            f"R²: {metrics['r2']:.4f} ± {metrics.get('r2_std', 0):.4f}"
        )

    def cross_validate_models(self, X, y, models):
        """Enhanced CV with full metrics tracking and standard deviation."""
        results = {}
        for name, model_type in models.items():
            try:
                logger.info(f"Starting CV for {name}...")
                model = self._initialize_models(self.target_col)[name]
                cv_scores = {'rmse': [], 'mae': [], 'r2': []}
                
                for fold, (train_idx, test_idx) in enumerate(self.cv.split(X), 1):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model.train(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = calculate_metrics(y_test, y_pred)
                    
                    for metric in cv_scores:
                        cv_scores[metric].append(metrics[metric])
                    
                    logger.debug(f"Fold {fold} {name} metrics: {metrics}")

                # Calculate mean and std of metrics
                avg_metrics = {k: np.mean(v) for k, v in cv_scores.items()}
                std_metrics = {k: np.std(v) for k, v in cv_scores.items()}
                
                results[name] = {
                    'model': model,
                    'metrics': {**avg_metrics, **{f"{k}_std": v for k, v in std_metrics.items()}},
                    'cv_scores': cv_scores
                }
                self._log_metrics(name, results[name]['metrics'])
                
            except Exception as e:
                logger.error(f"CV failed for {name}: {str(e)}")
                results[name] = None
        return results

    def select_best_model(self, results):
        """Select best model with comprehensive metrics."""
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            return None
            
        best_name = min(valid_results, key=lambda k: valid_results[k]['metrics']['rmse'])
        best = valid_results[best_name]
        
        return {
            'name': best_name,
            'model': best['model'],
            'metrics': best['metrics'],
            'cv_scores': best['cv_scores']
        }

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
