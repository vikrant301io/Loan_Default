"""Unit tests for model training."""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.models.trainer import ModelTrainer


class TestModelTrainer:
    """Test model training functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return X, y
    
    def test_handle_imbalance_smote(self, sample_data):
        """Test SMOTE imbalance handling."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        X_balanced, y_balanced = trainer.handle_imbalance(X, y, method='smote')
        
        assert len(X_balanced) >= len(X)
        assert len(y_balanced) >= len(y)
    
    def test_handle_imbalance_undersample(self, sample_data):
        """Test undersampling."""
        X, y = sample_data
        trainer = ModelTrainer()
        
        X_balanced, y_balanced = trainer.handle_imbalance(X, y, method='undersample')
        
        assert len(X_balanced) <= len(X)
        assert len(y_balanced) <= len(y)
    
    def test_get_models(self):
        """Test model dictionary creation."""
        trainer = ModelTrainer()
        models = trainer.get_models()
        
        assert 'Logistic Regression' in models
        assert 'Random Forest' in models
        assert 'XGBoost' in models
        assert 'LightGBM' in models
    
    def test_train_models(self, sample_data):
        """Test model training."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        trainer = ModelTrainer()
        results = trainer.train(
            X_train, y_train, X_test, y_test,
            handle_imbalance_method=None
        )
        
        assert len(results) > 0
        assert 'Logistic Regression' in results
        assert 'roc_auc' in results['Logistic Regression']
    
    def test_get_best_model(self, sample_data):
        """Test best model selection."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test, handle_imbalance_method=None)
        
        best_name, best_model = trainer.get_best_model(metric='roc_auc')
        
        assert best_name in trainer.results
        assert best_model is not None
    
    def test_get_comparison_df(self, sample_data):
        """Test comparison DataFrame generation."""
        X, y = sample_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]
        
        trainer = ModelTrainer()
        trainer.train(X_train, y_train, X_test, y_test, handle_imbalance_method=None)
        
        comparison_df = trainer.get_comparison_df()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert 'Model' in comparison_df.columns
        assert 'ROC-AUC' in comparison_df.columns
        assert len(comparison_df) > 0
