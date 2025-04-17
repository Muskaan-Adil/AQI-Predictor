import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        
    Returns:
        dict: Dictionary of metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    metrics = {}
    
    try:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        metrics['mean_error'] = np.mean(y_pred - y_true)
        metrics['median_error'] = np.median(y_pred - y_true)
        metrics['max_error'] = np.max(np.abs(y_pred - y_true))
        
        nonzero_mask = y_true != 0
        if np.any(nonzero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        else:
            metrics['mape'] = np.nan
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics
