import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    """
    Anomaly detection model for identifying unusual railway accidents.
    """
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.scaler = None
        self.features = ['Fatalities', 'Injuries', 'Year']
    
    def fit(self, df):
        """
        Train an anomaly detection model.
        
        Args:
            df: Preprocessed DataFrame
        """
        # Features for anomaly detection
        numeric_features = ['Fatalities', 'Injuries']
        
        # Get only the needed columns and drop rows with missing values
        model_df = df[numeric_features].dropna()
        
        # Add Year as a feature if it exists
        if 'Year' in df.columns:
            model_df['Year'] = df.loc[model_df.index, 'Year']
            self.features = numeric_features + ['Year']
        else:
            self.features = numeric_features
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(model_df[self.features])
        
        # Train Isolation Forest model
        self.model = IsolationForest(
            contamination=0.05,  # 5% of the data will be considered anomalies
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X)
        
        return self
    
    def detect_anomalies(self, df, contamination=0.05):
        """
        Detect anomalies in the dataset.
        
        Args:
            df: DataFrame to analyze
            contamination: Proportion of anomalies expected (0 to 0.5)
            
        Returns:
            DataFrame containing anomalies
        """
        if self.model is None:
            self.fit(df)
            
        # If contamination has changed, retrain the model
        elif self.model.contamination != contamination:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            
            # Extract features and scale
            numeric_features = ['Fatalities', 'Injuries']
            model_df = df[numeric_features].dropna()
            
            # Add Year as a feature if it exists
            if 'Year' in df.columns:
                model_df['Year'] = df.loc[model_df.index, 'Year']
                self.features = numeric_features + ['Year']
            else:
                self.features = numeric_features
            
            # Scale and fit
            X = self.scaler.transform(model_df[self.features])
            self.model.fit(X)
        
        # Extract features for prediction
        numeric_features = ['Fatalities', 'Injuries']
        pred_df = df[numeric_features].dropna()
        
        # Add Year if it's used as a feature
        if 'Year' in self.features and 'Year' in df.columns:
            pred_df['Year'] = df.loc[pred_df.index, 'Year']
        
        # Scale features
        X = self.scaler.transform(pred_df[self.features])
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        anomaly_predictions = self.model.predict(X)
        anomaly_scores = self.model.decision_function(X)
        
        # Normalize scores to 0-1 range for better interpretation
        # Lower scores indicate more anomalous points
        normalized_scores = 1 - (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        
        # Get anomalies
        anomaly_indices = pred_df.index[anomaly_predictions == -1]
        anomalies = df.loc[anomaly_indices].copy()
        
        # Add anomaly scores
        anomalies['anomaly_score'] = normalized_scores[anomaly_predictions == -1]
        
        return anomalies.sort_values('anomaly_score', ascending=False)
    
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
        
        # Save model and scaler
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, path)
    
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model
        """
        if not os.path.exists(path):
            raise ValueError(f"Model file '{path}' not found")
        
        # Load model and scaler
        saved_data = joblib.load(path)
        
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.features = saved_data['features']
        
        return self
