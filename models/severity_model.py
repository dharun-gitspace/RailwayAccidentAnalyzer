import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import joblib
import os

class SeverityModel:
    """
    Model for predicting accident severity.
    """
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.encoders = {}
        self.features = ['Accident_Type', 'Cause', 'State/Region', 'Decade']
        self.target = 'Severity'
        self.feature_importance = None
    
    def fit(self, df):
        """
        Train the severity prediction model.
        
        Args:
            df: Preprocessed DataFrame with 'Severity' column
        """
        # Make sure the target and features exist
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame")
        
        for feature in self.features:
            if feature not in df.columns:
                raise ValueError(f"Feature column '{feature}' not found in DataFrame")
        
        # Drop rows with missing target or features
        model_df = df.dropna(subset=[self.target] + self.features)
        
        # Encode categorical features
        X = pd.DataFrame()
        
        for feature in self.features:
            encoder = LabelEncoder()
            X[feature] = encoder.fit_transform(model_df[feature])
            self.encoders[feature] = encoder
        
        # Encode target
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(model_df[self.target])
        self.encoders['target'] = y_encoder
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Train on full dataset
        self.model.fit(X, y)
        
        # Store feature importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': importance
        })
        self.feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        return self
    
    def predict(self, input_data):
        """
        Predict the severity of an accident.
        
        Args:
            input_data: Dictionary with input feature values
            
        Returns:
            Predicted severity class and probability
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        
        # Encode input data
        encoded_input = []
        
        for feature in self.features:
            if feature in input_data:
                # Handle values not seen during training
                try:
                    encoded_value = self.encoders[feature].transform([input_data[feature]])[0]
                except:
                    # Use the most frequent class if the value was not seen during training
                    encoded_value = self.encoders[feature].transform([self.encoders[feature].classes_[0]])[0]
            else:
                # Use the most frequent class if the feature is missing
                encoded_value = self.encoders[feature].transform([self.encoders[feature].classes_[0]])[0]
            
            encoded_input.append(encoded_value)
        
        # Make prediction
        encoded_input = np.array(encoded_input).reshape(1, -1)
        prediction_encoded = self.model.predict(encoded_input)[0]
        probabilities = self.model.predict_proba(encoded_input)[0]
        
        # Decode prediction
        prediction = self.encoders['target'].inverse_transform([prediction_encoded])[0]
        
        # Get probability of the predicted class
        probability = probabilities[prediction_encoded] * 100
        
        return prediction, probability
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        
        return self.feature_importance
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call 'fit' first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model and encoders
        joblib.dump({
            'model': self.model,
            'encoders': self.encoders,
            'features': self.features,
            'feature_importance': self.feature_importance
        }, path)
    
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise ValueError(f"Model file '{path}' not found")
        
        # Load model and encoders
        saved_data = joblib.load(path)
        
        self.model = saved_data['model']
        self.encoders = saved_data['encoders']
        self.features = saved_data['features']
        self.feature_importance = saved_data['feature_importance']
        
        return self
