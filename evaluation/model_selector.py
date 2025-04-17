import pandas as pd
import logging
from src.utils.config import Config

logger = logging.getLogger(__name__)

class ModelSelector:
    """Class for selecting the best model based on performance metrics."""
    
    def __init__(self, metrics=None, primary_metric='rmse'):
        """Initialize the model selector.
        
        Args:
            metrics (dict, optional): Dictionary of model metrics. Defaults to None.
            primary_metric (str, optional): Primary metric for selection ('rmse', 'mae', or 'r2'). Defaults to 'rmse'.
        """
        self.metrics = metrics or {}
        self.primary_metric = primary_metric
    
    def select_best_model(self, metrics=None):
        """Select the best model based on the primary metric.
        
        Args:
            metrics (dict, optional): Dictionary of model metrics. Defaults to None (uses self.metrics).
            
        Returns:
            str: Name of the best model.
        """
        metrics = metrics or self.metrics
        
        if not metrics:
            logger.warning("No metrics provided for model selection")
            return None
        
        if self.primary_metric in ['rmse', 'mae']:
            best_model = min(metrics, key=lambda k: metrics[k][self.primary_metric])
        elif self.primary_metric == 'r2':
            best_model = max(metrics, key=lambda k: metrics[k][self.primary_metric])
        else:
            logger.warning(f"Unknown metric: {self.primary_metric}. Using RMSE.")
            best_model = min(metrics, key=lambda k: metrics[k]['rmse'])
        
        logger.info(f"Selected best model: {best_model} based on {self.primary_metric}")
        return best_model
    
    def get_model_ranking(self, metrics=None, ascending=None):
        """Get a ranking of models based on the primary metric.
        
        Args:
            metrics (dict, optional): Dictionary of model metrics. Defaults to None (uses self.metrics).
            ascending (bool, optional): Sort order. Defaults to None (True for RMSE/MAE, False for R²).
            
        Returns:
            pd.DataFrame: DataFrame with model rankings.
        """
        metrics = metrics or self.metrics
        
        if not metrics:
            logger.warning("No metrics provided for model ranking")
            return pd.DataFrame()
        
        if ascending is None:
            ascending = self.primary_metric in ['rmse', 'mae']
        
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.index.name = 'model'
        metrics_df = metrics_df.reset_index()
        
        metrics_df = metrics_df.sort_values(by=self.primary_metric, ascending=ascending)
        
        return metrics_df
