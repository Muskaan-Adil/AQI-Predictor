import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import logging
import os
import tempfile
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)

class NeuralNetModel(BaseModel):
    """Neural Network model implementation using TensorFlow/Keras."""
    
    def __init__(self, name="NeuralNetwork", target_col=None, model_type='mlp', layers=None, epochs=100):
        """Initialize the Neural Network model.
        
        Args:
            name (str, optional): Model name. Defaults to "NeuralNetwork".
            target_col (str, optional): Target column. Defaults to None.
            model_type (str, optional): Type of neural network ('mlp' or 'lstm'). Defaults to 'mlp'.
            layers (list, optional): List of layer sizes. Defaults to [64, 32].
            epochs (int, optional): Number of training epochs. Defaults to 100.
        """
        super().__init__(name=name, target_col=target_col)
        self.model_type = model_type
        self.layers = layers or [64, 32]
        self.epochs = epochs
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.feature_names_ = None
    
    def _build_mlp_model(self, input_dim):
        """Build a Multi-Layer Perceptron model.
        
        Args:
            input_dim (int): Number of input features.
            
        Returns:
            keras.Model: Compiled Keras model.
        """
        model = Sequential()
        
        model.add(Dense(self.layers[0], input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.2))
        
        for units in self.layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(0.2))
        
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def _build_lstm_model(self, input_shape):
        """Build an LSTM model.
        
        Args:
            input_shape (tuple): Shape of input data (samples, time steps, features).
            
        Returns:
            keras.Model: Compiled Keras model.
        """
        model = Sequential()
        
        model.add(LSTM(self.layers[0], return_sequences=len(self.layers) > 1, 
                       input_shape=(input_shape[1], input_shape[2])))
        model.add(Dropout(0.2))
        
        for i, units in enumerate(self.layers[1:-1], 1):
            return_sequences = i < len(self.layers) - 2
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(0.2))
        
        if len(self.layers) > 1:
            model.add(Dense(self.layers[-1], activation='relu'))
        
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def _prepare_lstm_data(self, X, y=None, time_steps=10):
        """Prepare data for LSTM (reshape into sequences).
        
        Args:
            X (np.array): Feature data.
            y (np.array, optional): Target data. Defaults to None.
            time_steps (int, optional): Number of time steps. Defaults to 10.
            
        Returns:
            tuple: (X_lstm, y_lstm) reshaped data.
        """
        X_lstm, y_lstm = [], []
        
        for i in range(len(X) - time_steps):
            X_lstm.append(X[i:i + time_steps])
            if y is not None:
                y_lstm.append(y[i + time_steps])
        
        X_lstm = np.array(X_lstm)
        
        if y is not None:
            y_lstm = np.array(y_lstm)
            return X_lstm, y_lstm
        
        return X_lstm
    
    def train(self, X, y):
        """Train the Neural Network model.
        
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series or pd.DataFrame): Target(s).
            
        Returns:
            object: Trained model.
        """
        self.feature_names_ = X.columns.tolist()
        
        if isinstance(y, pd.DataFrame):
            if self.target_col:
                y = y[self.target_col]
            else:
                self.target_col = y.columns[0]
                y = y[self.target_col]
        
        try:
            logger.info(f"Training {self.name} model...")
            
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1))
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            if self.model_type == 'lstm':
                X_lstm, y_lstm = self._prepare_lstm_data(X_scaled, y_scaled)
                
                self.model = self._build_lstm_model(X_lstm.shape)
                
                self.model.fit(
                    X_lstm, y_lstm,
                    epochs=self.epochs,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
            else:
                self.model = self._build_mlp_model(X.shape[1])
                
                self.model.fit(
                    X_scaled, y_scaled,
                    epochs=self.epochs,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
            
            logger.info(f"Trained {self.name} model successfully")
            
            return self.model
        except Exception as e:
            logger.error(f"Failed to train {self.name} model: {e}")
            return None
    
    def predict(self, X):
        """Make predictions with the Neural Network model.
        
        Args:
            X (pd.DataFrame): Features.
            
        Returns:
            np.array: Predictions.
        """
        if self.model is None:
            logger.error("Model not trained")
            return None
        
        try:
            X_scaled = self.scaler_X.transform(X)
            
            if self.model_type == 'lstm':
                X_lstm = self._prepare_lstm_data(X_scaled)
                
                predictions_scaled = self.model.predict(X_lstm)
            else:
                predictions_scaled = self.model.predict(X_scaled)
            
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
            
            return predictions.flatten()
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            return None
    
    def save(self, path=None):
        """Save the Neural Network model.
        
        Args:
            path (str, optional): Directory to save the model. Defaults to a temporary directory.
            
        Returns:
            str: Path where the model was saved.
        """
        if self.model is None:
            logger.error("No model to save")
            return None
        
        if path is None:
            path = tempfile.mkdtemp()
        
        try:
            model_path = os.path.join(path, f"{self.name}_model")
            self.model.save(model_path)
            
            scaler_X_path = os.path.join(path, f"{self.name}_scaler_X.joblib")
            scaler_y_path = os.path.join(path, f"{self.name}_scaler_y.joblib")
            
            import joblib
            joblib.dump(self.scaler_X, scaler_X_path)
            joblib.dump(self.scaler_y, scaler_y_path)
            
            logger.info(f"Model saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return None
    
    def load(self, path):
        """Load the Neural Network model.
        
        Args:
            path (str): Directory containing the saved model.
            
        Returns:
            object: Loaded model.
        """
        try:
            from tensorflow.keras.models import load_model
            model_path = os.path.join(path, f"{self.name}_model")
            self.model = load_model(model_path)
            
            import joblib
            scaler_X_path = os.path.join(path, f"{self.name}_scaler_X.joblib")
            scaler_y_path = os.path.join(path, f"{self.name}_scaler_y.joblib")
            
            self.scaler_X = joblib.load(scaler_X_path)
            self.scaler_y = joblib.load(scaler_y_path)
            
            logger.info(f"Model loaded from {path}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
