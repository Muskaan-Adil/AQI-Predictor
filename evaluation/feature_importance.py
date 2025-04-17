import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """Class for analyzing feature importance using SHAP."""
    
    def __init__(self):
        """Initialize the feature importance analyzer."""
        pass
    
    def get_model_feature_importance(self, model, X):
        """Get feature importance from model's attributes.
        
        Args:
            model: Trained model.
            X (pd.DataFrame): Feature data.
            
        Returns:
            pd.DataFrame: DataFrame with feature importances or None.
        """
        importance_df = None
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'coef_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': np.abs(model.coef_)
                }).sort_values('importance', ascending=False)
            elif hasattr(model, 'get_feature_importances'):
                importance_df = model.get_feature_importances()
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
        
        return importance_df
    
    def get_shap_values(self, model, X, sample_size=100):
        """Calculate SHAP values for feature importance.
        
        Args:
            model: Trained model.
            X (pd.DataFrame): Feature data.
            sample_size (int, optional): Number of samples to use. Defaults to 100.
            
        Returns:
            tuple: (shap_values, X_sample) or (None, None) if calculation fails.
        """
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        try:
            if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                explainer = shap.Explainer(model.predict, X_sample)
                shap_values = explainer(X_sample)
            else:
                logger.warning("Model not compatible with SHAP explainer")
                return None, None
            
            return shap_values, X_sample
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None, None
    
    def plot_shap_summary(self, shap_values, X_sample, save_path=None):
        """Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values.
            X_sample (pd.DataFrame): Sample data.
            save_path (str, optional): Path to save the plot. Defaults to None.
            
        Returns:
            matplotlib.figure.Figure: Figure object or None if plotting fails.
        """
        if shap_values is None:
            logger.error("No SHAP values to plot")
            return None
        
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                return plt.gcf()
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {e}")
            return None
    
    def get_top_features(self, shap_values, X_sample, top_n=10):
        """Get top features by SHAP importance.
        
        Args:
            shap_values: SHAP values.
            X_sample (pd.DataFrame): Sample data.
            top_n (int, optional): Number of top features to return. Defaults to 10.
            
        Returns:
            pd.DataFrame: DataFrame with top features or None if calculation fails.
        """
        if shap_values is None:
            logger.error("No SHAP values to analyze")
            return None
        
        try:
            feature_names = X_sample.columns
            importance_vals = np.abs(shap_values.values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_vals
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            return None
