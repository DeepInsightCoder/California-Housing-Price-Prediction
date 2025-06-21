"""
Utility functions for the California Housing Price Prediction project
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def load_california_housing_data():
    """Load the California housing dataset"""
    housing_data = fetch_california_housing()
    df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
    df['MedHouseVal'] = housing_data.target
    return df, housing_data

def create_engineered_features(df):
    """Create engineered features for the dataset"""
    df_processed = df.copy()
    
    # Create combined features
    df_processed['RoomsPerHousehold'] = df_processed['AveRooms'] / df_processed['AveOccup']
    df_processed['BedroomsPerRoom'] = df_processed['AveBedrms'] / df_processed['AveRooms']
    df_processed['PopulationPerHousehold'] = df_processed['Population'] / df_processed['AveOccup']
    
    # Handle any infinite or NaN values
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.fillna(df_processed.median(), inplace=True)
    
    return df_processed

def load_trained_models():
    """Load all trained models and preprocessing objects"""
    models = {}
    
    # Try to load all model files
    model_files = {
        'Linear Regression': 'model_linear_regression.pkl',
        'Decision Tree': 'model_decision_tree.pkl',
        'Random Forest': 'model_random_forest.pkl'
    }
    
    for name, filename in model_files.items():
        if os.path.exists(filename):
            try:
                models[name] = joblib.load(filename)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
    
    # Load best model
    best_model = None
    if os.path.exists('best_model.pkl'):
        try:
            best_model = joblib.load('best_model.pkl')
        except Exception as e:
            print(f"Warning: Could not load best model: {e}")
    
    # Load preprocessing objects
    scaler = None
    imputer = None
    
    if os.path.exists('scaler.pkl'):
        try:
            scaler = joblib.load('scaler.pkl')
        except Exception as e:
            print(f"Warning: Could not load scaler: {e}")
    
    if os.path.exists('imputer.pkl'):
        try:
            imputer = joblib.load('imputer.pkl')
        except Exception as e:
            print(f"Warning: Could not load imputer: {e}")
    
    # Load feature names
    feature_names = None
    if os.path.exists('feature_names.pkl'):
        try:
            feature_names = joblib.load('feature_names.pkl')
        except Exception as e:
            print(f"Warning: Could not load feature names: {e}")
    
    # Load model scores
    model_scores = None
    if os.path.exists('model_scores.pkl'):
        try:
            model_scores = joblib.load('model_scores.pkl')
        except Exception as e:
            print(f"Warning: Could not load model scores: {e}")
    
    return models, best_model, scaler, imputer, feature_names, model_scores

def preprocess_input_data(input_data, imputer=None, scaler=None, feature_names=None):
    """Preprocess input data for prediction"""
    # Convert input to DataFrame if it's not already
    if not isinstance(input_data, pd.DataFrame):
        if feature_names:
            input_df = pd.DataFrame([input_data], columns=feature_names)
        else:
            # Default feature names
            default_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                              'Population', 'AveOccup', 'Latitude', 'Longitude',
                              'RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold']
            input_df = pd.DataFrame([input_data], columns=default_features)
    else:
        input_df = input_data.copy()
    
    # Apply imputation if imputer is available
    if imputer:
        input_processed = imputer.transform(input_df)
        input_df = pd.DataFrame(input_processed, columns=input_df.columns)
    
    # Apply scaling if scaler is available
    if scaler:
        input_scaled = scaler.transform(input_df)
        input_df = pd.DataFrame(input_scaled, columns=input_df.columns)
    
    return input_df

def validate_input_ranges(input_data):
    """Validate that input data is within reasonable ranges"""
    validation_rules = {
        'MedInc': (0, 20),  # Median income should be positive and reasonable
        'HouseAge': (0, 100),  # House age should be between 0 and 100 years
        'AveRooms': (1, 20),  # Average rooms should be reasonable
        'AveBedrms': (0, 10),  # Average bedrooms should be reasonable
        'Population': (1, 50000),  # Population should be positive
        'AveOccup': (1, 20),  # Average occupancy should be reasonable
        'Latitude': (32, 42),  # California latitude range
        'Longitude': (-125, -114),  # California longitude range
    }
    
    warnings = []
    
    for feature, (min_val, max_val) in validation_rules.items():
        if feature in input_data:
            value = input_data[feature]
            if value < min_val or value > max_val:
                warnings.append(f"{feature}: {value} is outside typical range [{min_val}, {max_val}]")
    
    return warnings

def format_prediction_output(prediction, model_name=None):
    """Format prediction output for display"""
    prediction_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
    
    # Convert to actual dollar amount (California housing prices are in units of $100,000)
    dollar_amount = prediction_value * 100000
    
    output = {
        'prediction_raw': float(prediction_value),
        'prediction_dollars': f"${dollar_amount:,.2f}",
        'prediction_formatted': f"${dollar_amount/1000:.1f}K" if dollar_amount < 1000000 else f"${dollar_amount/1000000:.2f}M",
        'model_used': model_name or 'Unknown'
    }
    
    return output

def get_feature_descriptions():
    """Get descriptions for all features"""
    descriptions = {
        'MedInc': 'Median income in block group (in tens of thousands of dollars)',
        'HouseAge': 'Median house age in block group (in years)',
        'AveRooms': 'Average number of rooms per household',
        'AveBedrms': 'Average number of bedrooms per household',
        'Population': 'Block group population',
        'AveOccup': 'Average number of household members',
        'Latitude': 'Block group latitude',
        'Longitude': 'Block group longitude',
        'RoomsPerHousehold': 'Engineered: Rooms per household member',
        'BedroomsPerRoom': 'Engineered: Ratio of bedrooms to total rooms',
        'PopulationPerHousehold': 'Engineered: Population density per household'
    }
    return descriptions

def get_sample_inputs():
    """Get sample input values for testing"""
    samples = {
        'Low Income Area': {
            'MedInc': 2.5,
            'HouseAge': 30.0,
            'AveRooms': 4.5,
            'AveBedrms': 1.2,
            'Population': 3000,
            'AveOccup': 3.5,
            'Latitude': 34.0,
            'Longitude': -118.0
        },
        'High Income Area': {
            'MedInc': 8.0,
            'HouseAge': 15.0,
            'AveRooms': 6.5,
            'AveBedrms': 1.3,
            'Population': 2500,
            'AveOccup': 2.8,
            'Latitude': 37.5,
            'Longitude': -122.0
        },
        'Average Area': {
            'MedInc': 4.5,
            'HouseAge': 25.0,
            'AveRooms': 5.2,
            'AveBedrms': 1.1,
            'Population': 3500,
            'AveOccup': 3.2,
            'Latitude': 35.5,
            'Longitude': -120.0
        }
    }
    return samples

def calculate_engineered_features_from_input(input_data):
    """Calculate engineered features from basic input"""
    engineered = input_data.copy()
    
    # Calculate engineered features
    engineered['RoomsPerHousehold'] = engineered['AveRooms'] / engineered['AveOccup']
    engineered['BedroomsPerRoom'] = engineered['AveBedrms'] / engineered['AveRooms']
    engineered['PopulationPerHousehold'] = engineered['Population'] / engineered['AveOccup']
    
    # Handle any potential division by zero or infinite values
    engineered = engineered.replace([np.inf, -np.inf], np.nan)
    
    return engineered
