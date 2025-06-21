"""
Streamlit Web Application for California Housing Price Prediction
Interactive interface for making housing price predictions using trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from utils import (
    load_california_housing_data, 
    create_engineered_features,
    load_trained_models,
    preprocess_input_data,
    validate_input_ranges,
    format_prediction_output,
    get_feature_descriptions,
    get_sample_inputs,
    calculate_engineered_features_from_input
)
from explore import HousingDataExplorer
from train import HousingPricePredictor

# Configure the page
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HousingPredictionApp:
    """Main application class for the housing prediction app"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.model_scores = None
        self.sample_data = None
        
        # Initialize session state
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'sample_data_loaded' not in st.session_state:
            st.session_state.sample_data_loaded = False
    
    def load_models_and_data(self):
        """Load trained models and sample data"""
        try:
            # Load models
            self.models, self.best_model, self.scaler, self.imputer, self.feature_names, self.model_scores = load_trained_models()
            
            # Load sample data for visualizations
            df, _ = load_california_housing_data()
            self.sample_data = create_engineered_features(df)
            
            st.session_state.models_loaded = bool(self.models)
            st.session_state.sample_data_loaded = True
            
            return True
        except Exception as e:
            st.error(f"Error loading models or data: {str(e)}")
            return False
    
    def show_header(self):
        """Display the application header"""
        st.markdown('<h1 class="main-header">🏠 California Housing Price Predictor</h1>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <p><strong>Welcome to the California Housing Price Predictor!</strong></p>
        <p>This application uses machine learning models trained on California housing data to predict median house values. 
        Enter the housing characteristics below to get a price prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_sidebar_controls(self):
        """Display sidebar controls for input parameters"""
        st.sidebar.header("🏠 Housing Characteristics")
        st.sidebar.markdown("Enter the details of the area you want to evaluate:")
        
        # Get sample inputs for quick selection
        sample_inputs = get_sample_inputs()
        
        # Sample selection dropdown
        selected_sample = st.sidebar.selectbox(
            "Quick Start - Select a Sample:",
            ["Custom Input"] + list(sample_inputs.keys()),
            help="Choose a predefined sample or enter custom values"
        )
        
        # Initialize default values
        default_values = sample_inputs.get(selected_sample, {
            'MedInc': 4.5,
            'HouseAge': 25.0,
            'AveRooms': 5.2,
            'AveBedrms': 1.1,
            'Population': 3500,
            'AveOccup': 3.2,
            'Latitude': 35.5,
            'Longitude': -120.0
        }) if selected_sample != "Custom Input" else {
            'MedInc': 4.5,
            'HouseAge': 25.0,
            'AveRooms': 5.2,
            'AveBedrms': 1.1,
            'Population': 3500,
            'AveOccup': 3.2,
            'Latitude': 35.5,
            'Longitude': -120.0
        }
        
        st.sidebar.markdown("---")
        
        # Input fields
        input_data = {}
        
        input_data['MedInc'] = st.sidebar.number_input(
            "Median Income (in $10,000s)",
            min_value=0.0, max_value=20.0, 
            value=float(default_values['MedInc']),
            step=0.1,
            help="Median income in the block group (in tens of thousands of dollars)"
        )
        
        input_data['HouseAge'] = st.sidebar.number_input(
            "House Age (years)",
            min_value=0.0, max_value=100.0,
            value=float(default_values['HouseAge']),
            step=1.0,
            help="Median age of houses in the block group"
        )
        
        input_data['AveRooms'] = st.sidebar.number_input(
            "Average Rooms per Household",
            min_value=1.0, max_value=20.0,
            value=float(default_values['AveRooms']),
            step=0.1,
            help="Average number of rooms per household"
        )
        
        input_data['AveBedrms'] = st.sidebar.number_input(
            "Average Bedrooms per Household",
            min_value=0.0, max_value=10.0,
            value=float(default_values['AveBedrms']),
            step=0.1,
            help="Average number of bedrooms per household"
        )
        
        input_data['Population'] = st.sidebar.number_input(
            "Population",
            min_value=1, max_value=50000,
            value=int(default_values['Population']),
            step=100,
            help="Total population in the block group"
        )
        
        input_data['AveOccup'] = st.sidebar.number_input(
            "Average Occupancy",
            min_value=1.0, max_value=20.0,
            value=float(default_values['AveOccup']),
            step=0.1,
            help="Average number of household members"
        )
        
        input_data['Latitude'] = st.sidebar.number_input(
            "Latitude",
            min_value=32.0, max_value=42.0,
            value=float(default_values['Latitude']),
            step=0.01,
            help="Geographic latitude of the block group"
        )
        
        input_data['Longitude'] = st.sidebar.number_input(
            "Longitude",
            min_value=-125.0, max_value=-114.0,
            value=float(default_values['Longitude']),
            step=0.01,
            help="Geographic longitude of the block group"
        )
        
        return input_data
    
    def make_predictions(self, input_data):
        """Make predictions using all available models"""
        try:
            # Add engineered features
            engineered_input = calculate_engineered_features_from_input(input_data)
            
            # Validate input ranges
            warnings = validate_input_ranges(input_data)
            if warnings:
                st.warning("⚠️ Input Validation Warnings:")
                for warning in warnings:
                    st.warning(f"• {warning}")
            
            # Preprocess input data
            processed_input = preprocess_input_data(
                engineered_input, 
                self.imputer, 
                self.scaler, 
                self.feature_names
            )
            
            # Make predictions with all models
            predictions = {}
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(processed_input)
                    predictions[name] = format_prediction_output(pred, name)
                except Exception as e:
                    st.error(f"Error making prediction with {name}: {str(e)}")
            
            return predictions, engineered_input
            
        except Exception as e:
            st.error(f"Error in prediction process: {str(e)}")
            return {}, {}
    
    def display_predictions(self, predictions):
        """Display prediction results"""
        if not predictions:
            st.error("No predictions available. Please check if models are loaded correctly.")
            return
        
        st.markdown('<h2 class="section-header">🎯 Price Predictions</h2>', unsafe_allow_html=True)
        
        # Create columns for predictions
        cols = st.columns(len(predictions))
        
        for idx, (model_name, pred_info) in enumerate(predictions.items()):
            with cols[idx]:
                # Get model performance if available
                performance_info = ""
                if self.model_scores and model_name in self.model_scores:
                    rmse = self.model_scores[model_name].get('test_rmse', 'N/A')
                    mae = self.model_scores[model_name].get('test_mae', 'N/A')
                    performance_info = f"RMSE: {rmse:.4f} | MAE: {mae:.4f}" if rmse != 'N/A' else ""
                
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="margin-top: 0;">{model_name}</h3>
                    <h2 style="color: #1f77b4; margin: 0.5rem 0;">{pred_info['prediction_formatted']}</h2>
                    <p style="margin: 0; color: #666;">{pred_info['prediction_dollars']}</p>
                    <small style="color: #888;">{performance_info}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Show best model prediction prominently
        if self.best_model and predictions:
            best_pred = None
            for name, pred in predictions.items():
                if any(model_name.lower() in name.lower() for model_name in ['random forest', 'decision tree', 'linear']):
                    best_pred = pred
                    break
            
            if best_pred:
                st.markdown("---")
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #1f77b4, #17a2b8); color: white; padding: 1.5rem; border-radius: 0.5rem; text-align: center; margin: 1rem 0;">
                    <h3 style="margin: 0; color: white;">🏆 Best Model Prediction</h3>
                    <h1 style="margin: 0.5rem 0; color: white;">{best_pred['prediction_formatted']}</h1>
                    <p style="margin: 0; opacity: 0.9;">{best_pred['prediction_dollars']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def show_input_summary(self, input_data, engineered_data):
        """Display input data summary"""
        st.markdown('<h2 class="section-header">📊 Input Summary</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Features")
            basic_df = pd.DataFrame(list(input_data.items()), columns=['Feature', 'Value'])
            st.dataframe(basic_df, use_container_width=True)
        
        with col2:
            st.subheader("Engineered Features")
            if len(engineered_data) > len(input_data):
                engineered_features = {k: v for k, v in engineered_data.items() if k not in input_data}
                engineered_df = pd.DataFrame(list(engineered_features.items()), columns=['Feature', 'Value'])
                st.dataframe(engineered_df, use_container_width=True)
    
    def show_model_performance(self):
        """Display model performance comparison"""
        if not self.model_scores:
            st.info("Model performance data not available. Please run the training script first.")
            return
        
        st.markdown('<h2 class="section-header">📈 Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        # Create performance DataFrame
        performance_data = []
        for name, scores in self.model_scores.items():
            performance_data.append({
                'Model': name,
                'Test RMSE': round(scores.get('test_rmse', 0), 4),
                'Test MAE': round(scores.get('test_mae', 0), 4),
                'CV RMSE Mean': round(scores.get('cv_rmse_mean', 0), 4),
                'CV RMSE Std': round(scores.get('cv_rmse_std', 0), 4)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Display performance table
        st.dataframe(performance_df, use_container_width=True)
        
        # Create performance visualization
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Test RMSE',
            x=performance_df['Model'],
            y=performance_df['Test RMSE'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Test MAE',
            x=performance_df['Model'],
            y=performance_df['Test MAE'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Error Metrics',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_data_exploration(self):
        """Show data exploration section"""
        st.markdown('<h2 class="section-header">🔍 Data Exploration</h2>', unsafe_allow_html=True)
        
        if not st.session_state.sample_data_loaded or self.sample_data is None:
            st.info("Loading sample data for exploration...")
            try:
                df, _ = load_california_housing_data()
                self.sample_data = create_engineered_features(df)
                st.session_state.sample_data_loaded = True
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
                return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", f"{len(self.sample_data):,}")
        with col2:
            st.metric("Features", len(self.sample_data.columns) - 1)
        with col3:
            st.metric("Avg House Value", f"${self.sample_data['MedHouseVal'].mean()*100000:,.0f}")
        with col4:
            st.metric("Max House Value", f"${self.sample_data['MedHouseVal'].max()*100000:,.0f}")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Select feature to visualize
        feature_options = [col for col in self.sample_data.columns if col != 'MedHouseVal']
        selected_feature = st.selectbox("Select feature to visualize:", feature_options)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                self.sample_data, 
                x=selected_feature,
                nbins=50,
                title=f'Distribution of {selected_feature}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot vs target
            fig = px.scatter(
                self.sample_data.sample(1000),  # Sample for performance
                x=selected_feature,
                y='MedHouseVal',
                opacity=0.6,
                title=f'{selected_feature} vs House Value'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        
        # Calculate correlation matrix
        correlation_matrix = self.sample_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_about(self):
        """Show about section"""
        st.markdown('<h2 class="section-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Purpose
        This application demonstrates an end-to-end machine learning workflow for predicting California housing prices. 
        It implements concepts from "Hands-On Machine Learning with Scikit-Learn" and showcases:
        
        - **Data Loading & Exploration**: Comprehensive analysis of the California housing dataset
        - **Feature Engineering**: Creation of meaningful combined features
        - **Model Training**: Implementation of multiple ML algorithms
        - **Model Evaluation**: Rigorous performance comparison using cross-validation
        - **Interactive Prediction**: User-friendly interface for making predictions
        
        ### 🤖 Models Used
        - **Linear Regression**: Simple baseline model
        - **Decision Tree**: Non-linear model with interpretable rules
        - **Random Forest**: Ensemble method for improved accuracy
        
        ### 📊 Dataset Features
        - **MedInc**: Median income in block group (in $10,000s)
        - **HouseAge**: Median house age in block group
        - **AveRooms**: Average number of rooms per household
        - **AveBedrms**: Average number of bedrooms per household
        - **Population**: Block group population
        - **AveOccup**: Average number of household members
        - **Latitude/Longitude**: Geographic coordinates
        
        ### 🔧 Engineered Features
        - **RoomsPerHousehold**: Rooms divided by occupancy
        - **BedroomsPerRoom**: Bedroom ratio
        - **PopulationPerHousehold**: Population density measure
        
        ### 🚀 Technology Stack
        - **Streamlit**: Interactive web application framework
        - **Scikit-learn**: Machine learning algorithms and utilities
        - **Pandas**: Data manipulation and analysis
        - **Plotly**: Interactive visualizations
        - **NumPy**: Numerical computations
        """)
    
    def run(self):
        """Run the main application"""
        # Show header
        self.show_header()
        
        # Try to load models and data
        if not st.session_state.models_loaded:
            with st.spinner("Loading trained models and data..."):
                success = self.load_models_and_data()
                if not success:
                    st.error("""
                    ❌ **Models not found!** 
                    
                    Please run the training script first to generate the required model files:
                    ```
                    python train.py
                    ```
                    
                    This will create the necessary model files and preprocessing objects.
                    """)
                    st.stop()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📈 Model Performance", "🔍 Data Exploration", "ℹ️ About"])
        
        with tab1:
            # Main prediction interface
            input_data = self.show_sidebar_controls()
            
            if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
                with st.spinner("Making predictions..."):
                    predictions, engineered_input = self.make_predictions(input_data)
                    
                    if predictions:
                        self.display_predictions(predictions)
                        st.markdown("---")
                        self.show_input_summary(input_data, engineered_input)
            
            # Show feature descriptions
            st.markdown('<h2 class="section-header">📋 Feature Descriptions</h2>', unsafe_allow_html=True)
            descriptions = get_feature_descriptions()
            
            for feature, description in descriptions.items():
                if feature in list(input_data.keys()) + ['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold']:
                    st.markdown(f"**{feature}**: {description}")
        
        with tab2:
            self.show_model_performance()
        
        with tab3:
            self.show_data_exploration()
        
        with tab4:
            self.show_about()

def main():
    """Main function to run the Streamlit app"""
    app = HousingPredictionApp()
    app.run()

if __name__ == "__main__":
    main()
