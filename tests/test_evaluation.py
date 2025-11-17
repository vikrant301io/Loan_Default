"""Unit tests for model evaluation."""
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.evaluation.metrics import ModelEvaluator


class TestModelEvaluator:
    """Test model evaluation functionality."""
    
    @pytest.fixture
    def sample_model_and_data(self):
        """Create sample model and data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        return model, X, y, y_pred, y_pred_proba
    
    def test_calculate_all_metrics(self, sample_model_and_data):
        """Test metric calculation."""
        model, X, y, y_pred, y_pred_proba = sample_model_and_data
        
        evaluator = ModelEvaluator(model, X, y, y_pred, y_pred_proba)
        metrics = evaluator.calculate_all_metrics()
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
    
    def test_evaluate(self, sample_model_and_data):
        """Test comprehensive evaluation."""
        model, X, y, y_pred, y_pred_proba = sample_model_and_data
        
        evaluator = ModelEvaluator(model, X, y, y_pred, y_pred_proba)
        evaluation = evaluator.evaluate()
        
        assert 'confusion_matrix' in evaluation
        assert 'classification_report' in evaluation
        assert 'roc_auc' in evaluation
        assert 'optimal_threshold' in evaluation
    
    def test_get_feature_importance(self, sample_model_and_data):
        """Test feature importance extraction."""
        model, X, y, y_pred, y_pred_proba = sample_model_and_data
        
        evaluator = ModelEvaluator(model, X, y, y_pred, y_pred_proba)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        importance = evaluator.get_feature_importance(feature_names)
        
        assert importance is not None
        assert len(importance) == X.shape[1]
    
    def test_threshold_analysis(self, sample_model_and_data):
        """Test threshold analysis."""
        model, X, y, y_pred, y_pred_proba = sample_model_and_data
        
        evaluator = ModelEvaluator(model, X, y, y_pred, y_pred_proba)
        threshold_df = evaluator.threshold_analysis()
        
        assert 'threshold' in threshold_df.columns
        assert 'precision' in threshold_df.columns
        assert 'recall' in threshold_df.columns
        assert 'f1' in threshold_df.columns
        assert len(threshold_df) > 0
