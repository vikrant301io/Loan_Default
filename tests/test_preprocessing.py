"""Unit tests for preprocessing modules."""
import pytest
import pandas as pd
import numpy as np

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.scaler import FeatureScaler


class TestDataCleaner:
    """Test data cleaning functionality."""
    
    def test_remove_id_columns(self):
        """Test ID column removal."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Client_Income': [50000, 60000, 70000],
            'Default': [0, 1, 0]
        })
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.remove_id_columns(df)
        
        assert 'ID' not in cleaned_df.columns
        assert 'Client_Income' in cleaned_df.columns
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'Client_Income': [50000, 60000, 60000],
            'Default': [0, 1, 1]
        })
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.remove_duplicates(df)
        
        assert len(cleaned_df) == 2
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = pd.DataFrame({
            'Client_Income': [50000, np.nan, 70000],
            'Category': ['A', 'B', np.nan],
            'Default': [0, 1, 0]
        })
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.handle_missing_values(df)
        
        assert cleaned_df['Client_Income'].isnull().sum() == 0
        assert cleaned_df['Category'].isnull().sum() == 0
    
    def test_handle_outliers(self):
        """Test outlier handling."""
        df = pd.DataFrame({
            'Client_Income': [50000, 60000, 1000000],  # 1000000 is outlier
            'Default': [0, 1, 0]
        })
        
        cleaner = DataCleaner()
        cleaned_df = cleaner.handle_outliers(df, method='cap')
        
        # Outlier should be capped
        assert cleaned_df['Client_Income'].max() < 1000000


class TestCategoricalEncoder:
    """Test categorical encoding."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        df = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'A'],
            'Default': [0, 1, 0, 1]
        })
        
        encoder = CategoricalEncoder()
        encoded_df = encoder.fit_transform(df)
        
        assert encoded_df['Category'].dtype in [np.int64, int]
        assert encoded_df['Category'].nunique() == 3
    
    def test_transform_unseen_categories(self):
        """Test handling of unseen categories."""
        train_df = pd.DataFrame({
            'Category': ['A', 'B', 'C'],
            'Default': [0, 1, 0]
        })
        
        test_df = pd.DataFrame({
            'Category': ['A', 'D'],  # D is unseen
            'Default': [0, 1]
        })
        
        encoder = CategoricalEncoder()
        encoder.fit(train_df)
        encoded_df = encoder.transform(test_df)
        
        # Unseen category should be encoded as -1
        assert -1 in encoded_df['Category'].values


class TestFeatureEngineer:
    """Test feature engineering."""
    
    def test_create_features(self):
        """Test feature creation."""
        df = pd.DataFrame({
            'Age_Days': [-7300, -10950, -14600],
            'Client_Income': [50000, 60000, 70000],
            'Credit_Amount': [100000, 120000, 140000],
            'Default': [0, 1, 0]
        })
        
        engineer = FeatureEngineer()
        engineered_df = engineer.create_features(df)
        
        assert 'Age_Years' in engineered_df.columns
        assert 'Credit_Income_Ratio' in engineered_df.columns
    
    def test_age_features(self):
        """Test age-related features."""
        df = pd.DataFrame({
            'Age_Days': [-7300, -10950],
            'Default': [0, 1]
        })
        
        engineer = FeatureEngineer()
        engineered_df = engineer.create_features(df)
        
        assert 'Age_Years' in engineered_df.columns
        assert engineered_df['Age_Years'].min() > 0


class TestFeatureScaler:
    """Test feature scaling."""
    
    def test_robust_scaler(self):
        """Test robust scaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = FeatureScaler(scaler_type='robust')
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.shape == X.shape
        assert not np.isnan(X_scaled).any()
    
    def test_standard_scaler(self):
        """Test standard scaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = FeatureScaler(scaler_type='standard')
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.shape == X.shape
        assert not np.isnan(X_scaled).any()
    
    def test_invalid_scaler_type(self):
        """Test invalid scaler type."""
        with pytest.raises(ValueError):
            FeatureScaler(scaler_type='invalid')
