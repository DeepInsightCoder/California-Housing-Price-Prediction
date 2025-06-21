"""
Data Exploration Module for California Housing Dataset
This module handles data loading, exploration, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings('ignore')

class HousingDataExplorer:
    """Class to handle data exploration and visualization for California housing dataset"""
    
    def __init__(self):
        self.housing_data = None
        self.df = None
        
    def load_data(self):
        """Load the California housing dataset"""
        print("Loading California housing dataset...")
        self.housing_data = fetch_california_housing()
        self.df = pd.DataFrame(self.housing_data.data, columns=self.housing_data.feature_names)
        self.df['MedHouseVal'] = self.housing_data.target
        print(f"Dataset loaded successfully with shape: {self.df.shape}")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*50)
        print("BASIC DATASET INFORMATION")
        print("="*50)
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nFeature Descriptions:")
        feature_descriptions = {
            'MedInc': 'Median income in block group',
            'HouseAge': 'Median house age in block group',
            'AveRooms': 'Average number of rooms per household',
            'AveBedrms': 'Average number of bedrooms per household',
            'Population': 'Block group population',
            'AveOccup': 'Average number of household members',
            'Latitude': 'Block group latitude',
            'Longitude': 'Block group longitude',
            'MedHouseVal': 'Median house value (target variable)'
        }
        
        for feature, description in feature_descriptions.items():
            print(f"{feature}: {description}")
    
    def create_histograms(self, save_plots=False):
        """Create histograms for all features"""
        print("\nGenerating histograms...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Distribution of Housing Features', fontsize=16, y=0.98)
        
        features = self.df.columns
        axes_flat = axes.flatten()
        
        for i, feature in enumerate(features):
            axes_flat[i].hist(self.df[feature], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes_flat[i].set_title(f'{feature}')
            axes_flat[i].set_xlabel(feature)
            axes_flat[i].set_ylabel('Frequency')
            axes_flat[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('housing_histograms.png', dpi=300, bbox_inches='tight')
            print("Histograms saved as 'housing_histograms.png'")
        
        plt.show()
    
    def create_scatter_matrix(self, save_plots=False):
        """Create scatter matrix for key attributes"""
        print("\nGenerating scatter matrix...")
        
        # Select key attributes for scatter matrix
        key_attributes = ['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']
        
        fig, axes = scatter_matrix(
            self.df[key_attributes], 
            figsize=(12, 10), 
            alpha=0.6,
            diagonal='hist',
            hist_kwds={'bins': 50, 'alpha': 0.7}
        )
        
        # Improve the plot appearance
        for ax in axes.flatten():
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Scatter Matrix of Key Housing Attributes', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('scatter_matrix.png', dpi=300, bbox_inches='tight')
            print("Scatter matrix saved as 'scatter_matrix.png'")
        
        plt.show()
    
    def create_correlation_heatmap(self, save_plots=False):
        """Create correlation heatmap"""
        print("\nGenerating correlation heatmap...")
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            mask=mask,
            square=True,
            fmt='.2f',
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title('Correlation Matrix of Housing Features', fontsize=14, pad=20)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("Correlation heatmap saved as 'correlation_heatmap.png'")
        
        plt.show()
    
    def geographical_visualization(self, save_plots=False):
        """Create geographical visualization of housing data"""
        print("\nGenerating geographical visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with latitude and longitude
        scatter = plt.scatter(
            self.df['Longitude'], 
            self.df['Latitude'],
            c=self.df['MedHouseVal'],
            cmap='viridis',
            alpha=0.6,
            s=self.df['Population']/100,  # Size based on population
            edgecolors='black',
            linewidth=0.5
        )
        
        plt.colorbar(scatter, label='Median House Value')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('California Housing Prices by Location\n(Size = Population, Color = Median House Value)')
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            plt.savefig('geographical_plot.png', dpi=300, bbox_inches='tight')
            print("Geographical plot saved as 'geographical_plot.png'")
        
        plt.show()
    
    def analyze_outliers(self):
        """Analyze outliers in the dataset"""
        print("\n" + "="*50)
        print("OUTLIER ANALYSIS")
        print("="*50)
        
        for column in self.df.columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            outlier_percentage = (len(outliers) / len(self.df)) * 100
            
            print(f"{column}: {len(outliers)} outliers ({outlier_percentage:.2f}%)")
    
    def feature_insights(self):
        """Provide insights about the features"""
        print("\n" + "="*50)
        print("FEATURE INSIGHTS")
        print("="*50)
        
        # Income vs House Value correlation
        income_corr = self.df['MedInc'].corr(self.df['MedHouseVal'])
        print(f"Correlation between Median Income and House Value: {income_corr:.3f}")
        
        # Average rooms analysis
        avg_rooms_corr = self.df['AveRooms'].corr(self.df['MedHouseVal'])
        print(f"Correlation between Average Rooms and House Value: {avg_rooms_corr:.3f}")
        
        # House age analysis
        age_corr = self.df['HouseAge'].corr(self.df['MedHouseVal'])
        print(f"Correlation between House Age and House Value: {age_corr:.3f}")
        
        # Population density insights
        print(f"\nPopulation Statistics:")
        print(f"Mean Population: {self.df['Population'].mean():.0f}")
        print(f"Max Population: {self.df['Population'].max():.0f}")
        print(f"Min Population: {self.df['Population'].min():.0f}")

def main():
    """Main function to run data exploration"""
    explorer = HousingDataExplorer()
    
    # Load data
    df = explorer.load_data()
    
    # Basic information
    explorer.basic_info()
    
    # Visualizations
    explorer.create_histograms(save_plots=False)
    explorer.create_scatter_matrix(save_plots=False)
    explorer.create_correlation_heatmap(save_plots=False)
    explorer.geographical_visualization(save_plots=False)
    
    # Analysis
    explorer.analyze_outliers()
    explorer.feature_insights()
    
    print("\n" + "="*50)
    print("DATA EXPLORATION COMPLETED")
    print("="*50)

if __name__ == "__main__":
    main()
