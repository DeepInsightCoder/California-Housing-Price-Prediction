"""
Quick training script for California Housing models
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# Feature engineering
print("Creating features...")
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']

# Handle infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Prepare data
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
print("Preprocessing...")
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize models
print("Training models...")
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10)
}

model_scores = {}
trained_models = {}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    model_scores[name] = {
        'test_rmse': rmse,
        'test_mae': mae,
        'test_predictions': y_pred
    }
    
    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Save everything
print("Saving models...")

# Save individual models
for name, model in trained_models.items():
    filename = f"model_{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)

# Save best model (lowest RMSE)
best_model_name = min(model_scores.keys(), key=lambda x: model_scores[x]['test_rmse'])
best_model = trained_models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')

# Save preprocessing objects
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names and scores
joblib.dump(list(X.columns), 'feature_names.pkl')
joblib.dump(model_scores, 'model_scores.pkl')

print(f"Training completed! Best model: {best_model_name}")
print("All models and preprocessing objects saved.")