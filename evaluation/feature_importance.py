import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import logging
from src.utils.config import Config

logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """Class for analyzing feature importance using SHAP."""
    
    def __init__(self, model=None, model_type=None, X=None):
        """Initialize the feature importance analyzer.
        
        Args:
            model: Trained model object.
            model_type (str, optional): Type of model ('tree', 'linear', 'neural_net'). Defaults to None (auto-detect).
            X (pd.DataFrame, optional): Features DataFrame for SHAP. Defaults to None.
        """
        self.model = model
        self.model_type = model_type
        self.X = X
        self.explainer = None
        self.shap_values = None
    
    def _detect_model_type(self):
        """Automatically detect the model type based on the model object.
        
        Returns:
            str: Detected model type.
        """
        if self.model is None:
            return None
        
        model_class = self.model.__class__.__name__
        
        if model_class in ['RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor']:
            return 'tree'
        elif model_class in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            return 'linear'
        elif 'keras' in str(type(self.model)):
            return 'neural_net'
        else:
            return 'kernel'
    
    def generate_explainer(self, X=None):
        """Generate a SHAP explainer for the model.
        
        Args:
            X (pd.DataFrame, optional): Features DataFrame. Defaults to None (uses self.X).
            
        Returns:
            object: SHAP explainer object.
        """
        if X is not None:
            self.X = X
        
        if self.model is None or self.X is None:
            logger.error("Model and features are required to generate explainer")
            return None
        
        if self.model_type is None:
            self.model_type = self._detect_model_type()
        
        try:
            logger.info(f"Generating SHAP explainer for model type: {self.model_type}")
            
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, self.X)
            elif self.model_type == 'neural_net':
                background = shap.sample(self.X, 100)
                self.explainer = shap.DeepExplainer(self.model, background)
            else:
                self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X, 100))
            
            return self.explainer
        except Exception as e:
            logger.error(f"Error generating SHAP explainer: {e}")
            return None
    
    def calculate_shap_values(self, X=None):
        """Calculate SHAP values for the features.
        
        Args:
            X (pd.DataFrame, optional): Features DataFrame. Defaults to None (uses self.X).
            
        Returns:
            array: SHAP values.
        """
        if X is not None:
            self.X = X
        
        if self.explainer is None:
            self.generate_explainer(self.X)
            
        if self.explainer is None:
            logger.error("No explainer available to calculate SHAP values")
            return None
        
        try:
            logger.info("Calculating SHAP values...")
            self.shap_values = self.explainer.shap_values(self.X)
            return self.shap_values
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return None
    
    def plot_shap_summary(self, X=None, file_path=None, max_display=20):
        """Create a SHAP summary plot.
        
        Args:
            X (pd.DataFrame, optional): Features DataFrame. Defaults to None (uses self.X).
            file_path (str, optional): Path to save the plot. Defaults to None (displays plot).
            max_display (int, optional): Maximum number of features to display. Defaults to 20.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if X is not None:
            self.X = X
        
        if self.shap_values is None:
            self.calculate_shap_values(self.X)
            
        if self.shap_values is None:
            logger.error("No SHAP values available to plot")
            return False
        
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values, 
                self.X, 
                plot_type="bar", 
                max_display=max_display,
                show=False
            )
            
            if file_path:
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP summary plot saved to {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            return False
    
    def plot_shap_dependence(self, feature, interaction_feature=None, file_path=None):
        """Create a SHAP dependence plot for a specific feature.
        
        Args:
            feature (str): Feature to plot.
            interaction_feature (str, optional): Feature to use for interaction. Defaults to None.
            file_path (str, optional): Path to save the plot. Defaults to None (displays plot).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.shap_values is None:
            self.calculate_shap_values(self.X)
            
        if self.shap_values is None:
            logger.error("No SHAP values available to plot")
            return False
        
        if feature not in self.X.columns:
            logger.error(f"Feature '{feature}' not found in dataset")
            return False
        
        try:
            plt.figure(figsize=(10, 6))
            
            if interaction_feature and interaction_feature in self.X.columns:
                shap.dependence_plot(
                    feature, 
                    self.shap_values, 
                    self.X,
                    interaction_index=interaction_feature,
                    show=False
                )
            else:
                shap.dependence_plot(
                    feature, 
                    self.shap_values, 
                    self.X,
                    show=False
                )
            
            if file_path:
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                logger.info(f"SHAP dependence plot for {feature} saved to {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error creating SHAP dependence plot: {e}")
            return False
    
    def get_feature_importance_df(self):
        """Get feature importances as a DataFrame.
        
        Returns:
            pd.DataFrame: Feature importances.
        """
        if self.shap_values is None:
            self.calculate_shap_values(self.X)
            
        if self.shap_values is None or self.X is None:
            logger.error("No SHAP values or features available")
            return pd.DataFrame()
        
        try:
            feature_names = self.X.columns
            feature_importance = np.abs(self.shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            })
            
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        except Exception as e:
            logger.error(f"Error creating feature importance DataFrame: {e}")
            return pd.DataFrame()
