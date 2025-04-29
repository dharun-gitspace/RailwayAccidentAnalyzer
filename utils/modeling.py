import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import xgboost as xgb
import joblib
import os

def train_severity_model(df):
    """
    Train a model to predict accident severity.
    
    Args:
        df: Preprocessed DataFrame with 'Severity' column
        
    Returns:
        Trained model, label encoders, and feature names
    """
    # Features and target
    features = ['Accident_Type', 'Cause', 'State/Region', 'Decade']
    target = 'Severity'
    
    # Drop rows with missing target or features
    model_df = df.dropna(subset=[target] + features)
    
    # Encode categorical features
    encoders = {}
    X = pd.DataFrame()
    
    for feature in features:
        encoder = LabelEncoder()
        X[feature] = encoder.fit_transform(model_df[feature])
        encoders[feature] = encoder
    
    # Encode target
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(model_df[target])
    encoders['target'] = y_encoder
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Use stratified cross-validation to evaluate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean F1 score: {cv_scores.mean()}")
    
    # Train on full dataset
    model.fit(X, y)
    
    return model, encoders, features

def predict_severity(model, encoders, features, input_data):
    """
    Predict the severity of an accident.
    
    Args:
        model: Trained model
        encoders: Dictionary of label encoders for each feature
        features: List of feature names
        input_data: Dictionary with input feature values
        
    Returns:
        Predicted severity class and probability
    """
    # Encode input data
    encoded_input = []
    
    for feature in features:
        if feature in input_data:
            # Handle values not seen during training
            try:
                encoded_value = encoders[feature].transform([input_data[feature]])[0]
            except:
                # Use the most frequent class if the value was not seen during training
                encoded_value = encoders[feature].transform([encoders[feature].classes_[0]])[0]
        else:
            # Use the most frequent class if the feature is missing
            encoded_value = encoders[feature].transform([encoders[feature].classes_[0]])[0]
        
        encoded_input.append(encoded_value)
    
    # Make prediction
    encoded_input = np.array(encoded_input).reshape(1, -1)
    prediction_encoded = model.predict(encoded_input)[0]
    probabilities = model.predict_proba(encoded_input)[0]
    
    # Decode prediction
    prediction = encoders['target'].inverse_transform([prediction_encoded])[0]
    
    # Get probability of the predicted class
    probability = probabilities[prediction_encoded] * 100
    
    return prediction, probability

def train_anomaly_detector(df):
    """
    Train an anomaly detection model.
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        Trained anomaly detection model
    """
    # Features for anomaly detection
    features = ['Fatalities', 'Injuries']
    
    # Drop rows with missing features
    model_df = df.dropna(subset=features)
    
    # Scale numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(model_df[features])
    
    # Train Isolation Forest model
    model = IsolationForest(
        contamination=0.05,  # 5% of the data will be considered anomalies
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X)
    
    return model, scaler, features
