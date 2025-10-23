
import pickle
import pandas as pd
import numpy as np

def load_models_and_preprocessing():
    """Load all trained models and preprocessing objects"""
    
    # Load models
    with open('trained_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    # Load encoders
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    # Load scaler
    with open('feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load training data
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    return models, encoders, scaler, data

def predict_with_model(model_name, X_new, models, encoders, scaler):
    """Make predictions with a specific trained model"""
    
    # Get model info
    model_info = models[model_name]
    model = model_info['model']
    use_scaled = model_info['use_scaled_data']
    
    # Encode categorical variables
    X_encoded = X_new.copy()
    for feature, encoder in encoders.items():
        if feature in X_encoded.columns:
            X_encoded[feature] = encoder.transform(X_encoded[feature].astype(str))
    
    # Scale if needed
    if use_scaled:
        numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        X_encoded[numerical_features] = scaler.transform(X_encoded[numerical_features])
    
    # Make prediction
    prediction = model.predict(X_encoded)
    return prediction

def get_all_predictions(X_new, models, encoders, scaler):
    """Get predictions from all models"""
    predictions = {}
    for model_name in models.keys():
        predictions[model_name] = predict_with_model(model_name, X_new, models, encoders, scaler)
    return predictions

# Example usage:
# models, encoders, scaler, data = load_models_and_preprocessing()
# predictions = get_all_predictions(X_test, models, encoders, scaler)
