"""
Model Training Module for California Housing Price Prediction
This module handles data preprocessing, model training, evaluation, and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class HousingPricePredictor:
    """Class to handle the complete ML pipeline for housing price prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.housing_data = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading California housing dataset...")
        self.housing_data = fetch_california_housing()
        self.df = pd.DataFrame(self.housing_data.data, columns=self.housing_data.feature_names)
        self.df['MedHouseVal'] = self.housing_data.target
        
        print(f"Dataset loaded with shape: {self.df.shape}")
        return self.df
    
    def create_income_categories(self):
        """Create income categories for stratified sampling"""
        print("Creating income categories for stratified sampling...")
        
        # Create income categories based on median income
        self.df['IncCat'] = pd.cut(
            self.df['MedInc'],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1, 2, 3, 4, 5]
        )
        
        print("Income category distribution:")
        print(self.df['IncCat'].value_counts().sort_index())
        
        return self.df['IncCat']
    
    def feature_engineering(self):
        """Create additional features"""
        print("Performing feature engineering...")
        
        # Create combined features
        self.df['RoomsPerHousehold'] = self.df['AveRooms'] / self.df['AveOccup']
        self.df['BedroomsPerRoom'] = self.df['AveBedrms'] / self.df['AveRooms']
        self.df['PopulationPerHousehold'] = self.df['Population'] / self.df['AveOccup']
        
        print("New features created:")
        print("- RoomsPerHousehold")
        print("- BedroomsPerRoom") 
        print("- PopulationPerHousehold")
        
        # Handle any infinite or NaN values
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return self.df
    
    def split_data(self):
        """Split data using stratified sampling"""
        print("Splitting data with stratified sampling...")
        
        # Create income categories for stratification
        income_cat = self.create_income_categories()
        
        # Prepare features and target
        X = self.df.drop(['MedHouseVal', 'IncCat'], axis=1)
        y = self.df['MedHouseVal']
        
        # Stratified split
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
        
        for train_index, test_index in split.split(X, income_cat):
            self.X_train = X.iloc[train_index]
            self.X_test = X.iloc[test_index]
            self.y_train = y.iloc[train_index]
            self.y_test = y.iloc[test_index]
        
        # Remove income category column as it's no longer needed
        if 'IncCat' in self.X_train.columns:
            self.X_train = self.X_train.drop('IncCat', axis=1)
            self.X_test = self.X_test.drop('IncCat', axis=1)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Verify stratification worked
        print("\nIncome category proportions:")
        print("Overall:", (income_cat.value_counts() / len(income_cat)).sort_index())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_data(self):
        """Preprocess the data (imputation and scaling)"""
        print("Preprocessing data...")
        
        # Handle missing values with median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(self.X_train)
        X_test_imputed = imputer.transform(self.X_test)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        # Convert back to DataFrame to maintain feature names
        self.X_train_processed = pd.DataFrame(
            X_train_scaled, 
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_processed = pd.DataFrame(
            X_test_scaled, 
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        # Save preprocessing objects
        joblib.dump(imputer, 'imputer.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        
        print("Data preprocessing completed")
        return self.X_train_processed, self.X_test_processed
    
    def initialize_models(self):
        """Initialize ML models"""
        print("Initializing models...")
        
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=self.random_state),
            "Random Forest": RandomForestRegressor(random_state=self.random_state, n_estimators=100)
        }
        
        print(f"Initialized {len(self.models)} models")
        return self.models
    
    def train_models(self):
        """Train all models"""
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(self.X_train_processed, self.y_train)
            self.trained_models[name] = model
            
            # Make predictions
            train_pred = model.predict(self.X_train_processed)
            test_pred = model.predict(self.X_test_processed)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
            train_mae = mean_absolute_error(self.y_train, train_pred)
            test_mae = mean_absolute_error(self.y_test, test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train_processed, self.y_train,
                cv=5, scoring='neg_mean_squared_error'
            )
            cv_rmse = np.sqrt(-cv_scores)
            
            # Store results
            self.model_scores[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std(),
                'test_predictions': test_pred
            }
            
            print(f"{name} - Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
        
        print("Model training completed")
        return self.trained_models, self.model_scores
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Create results DataFrame
        results_data = []
        for name, scores in self.model_scores.items():
            results_data.append({
                'Model': name,
                'Train RMSE': f"{scores['train_rmse']:.4f}",
                'Test RMSE': f"{scores['test_rmse']:.4f}",
                'Train MAE': f"{scores['train_mae']:.4f}",
                'Test MAE': f"{scores['test_mae']:.4f}",
                'CV RMSE (mean±std)': f"{scores['cv_rmse_mean']:.4f}±{scores['cv_rmse_std']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        # Find best model based on test RMSE
        best_model_name = min(self.model_scores.keys(), 
                             key=lambda x: self.model_scores[x]['test_rmse'])
        self.best_model = self.trained_models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Test RMSE: {self.model_scores[best_model_name]['test_rmse']:.4f}")
        
        return results_df, best_model_name
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning on the best model"""
        print(f"\nPerforming hyperparameter tuning on {self.best_model_name}...")
        
        if self.best_model_name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
        elif self.best_model_name == "Decision Tree":
            param_grid = {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            
            grid_search = GridSearchCV(
                DecisionTreeRegressor(random_state=self.random_state),
                param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        else:
            print("No hyperparameter tuning for Linear Regression")
            return self.best_model
        
        # Fit grid search
        grid_search.fit(self.X_train_processed, self.y_train)
        
        # Get best model
        tuned_model = grid_search.best_estimator_
        
        # Evaluate tuned model
        tuned_pred = tuned_model.predict(self.X_test_processed)
        tuned_rmse = np.sqrt(mean_squared_error(self.y_test, tuned_pred))
        tuned_mae = mean_absolute_error(self.y_test, tuned_pred)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Tuned Model Test RMSE: {tuned_rmse:.4f}")
        print(f"Tuned Model Test MAE: {tuned_mae:.4f}")
        print(f"Original Model Test RMSE: {self.model_scores[self.best_model_name]['test_rmse']:.4f}")
        
        # Update best model if tuned version is better
        if tuned_rmse < self.model_scores[self.best_model_name]['test_rmse']:
            self.best_model = tuned_model
            print("Tuned model is better - using tuned version")
        else:
            print("Original model is better - keeping original")
        
        return self.best_model
    
    def create_visualizations(self):
        """Create evaluation visualizations"""
        print("Creating evaluation visualizations...")
        
        # 1. Actual vs Predicted plot for all models
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (name, scores) in enumerate(self.model_scores.items()):
            ax = axes[idx]
            
            # Plot actual vs predicted
            ax.scatter(self.y_test, scores['test_predictions'], alpha=0.6, s=30)
            
            # Plot perfect prediction line
            min_val = min(self.y_test.min(), scores['test_predictions'].min())
            max_val = max(self.y_test.max(), scores['test_predictions'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{name}\nRMSE: {scores["test_rmse"]:.4f}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Actual vs Predicted Values Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # 2. Residual plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (name, scores) in enumerate(self.model_scores.items()):
            ax = axes[idx]
            
            residuals = self.y_test - scores['test_predictions']
            ax.scatter(scores['test_predictions'], residuals, alpha=0.6, s=30)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'{name} - Residual Plot')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Residual Analysis', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # 3. Feature importance for tree-based models
        tree_models = ['Decision Tree', 'Random Forest']
        tree_models_available = [name for name in tree_models if name in self.trained_models]
        
        if tree_models_available:
            fig, axes = plt.subplots(1, len(tree_models_available), figsize=(6*len(tree_models_available), 8))
            if len(tree_models_available) == 1:
                axes = [axes]
            
            for idx, name in enumerate(tree_models_available):
                model = self.trained_models[name]
                feature_importance = model.feature_importances_
                feature_names = self.X_train_processed.columns
                
                # Sort features by importance
                indices = np.argsort(feature_importance)[::-1]
                
                ax = axes[idx]
                ax.bar(range(len(feature_importance)), feature_importance[indices])
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title(f'{name} - Feature Importance')
                ax.set_xticks(range(len(feature_importance)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("Saving models and preprocessing objects...")
        
        # Save all trained models
        for name, model in self.trained_models.items():
            filename = f"model_{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} as {filename}")
        
        # Save the best model separately
        joblib.dump(self.best_model, 'best_model.pkl')
        print(f"Saved best model ({self.best_model_name}) as best_model.pkl")
        
        # Save feature names
        feature_names = list(self.X_train_processed.columns)
        joblib.dump(feature_names, 'feature_names.pkl')
        print("Saved feature names as feature_names.pkl")
        
        # Save model performance scores
        joblib.dump(self.model_scores, 'model_scores.pkl')
        print("Saved model scores as model_scores.pkl")
        
        print("All models and objects saved successfully")

def main():
    """Main function to run the complete training pipeline"""
    print("="*60)
    print("CALIFORNIA HOUSING PRICE PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Initialize predictor
    predictor = HousingPricePredictor(random_state=42)
    
    # Load and prepare data
    df = predictor.load_and_prepare_data()
    
    # Feature engineering
    df = predictor.feature_engineering()
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data()
    
    # Preprocess data
    X_train_processed, X_test_processed = predictor.preprocess_data()
    
    # Initialize and train models
    models = predictor.initialize_models()
    trained_models, model_scores = predictor.train_models()
    
    # Evaluate models
    results_df, best_model_name = predictor.evaluate_models()
    
    # Hyperparameter tuning
    best_model = predictor.hyperparameter_tuning()
    
    # Create visualizations
    predictor.create_visualizations()
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print("All models and preprocessing objects have been saved")

if __name__ == "__main__":
    main()
