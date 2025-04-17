import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        
    Returns:
        dict: Dictionary with metrics.
    """
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}")
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            'rmse': float('inf'),
            'mae': float('inf'),
            'r2': -float('inf')
        }
