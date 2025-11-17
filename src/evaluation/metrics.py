"""Model evaluation metrics module."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = 'Model'
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            y_pred: Predictions
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name
        logger.info(f"Initialized ModelEvaluator for {model_name}")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.
        
        Returns:
            Dictionary of evaluation metrics
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Calculate metrics
        metrics = {
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                self.y_test, self.y_pred,
                target_names=['No Default', 'Default'],
                output_dict=True
            ),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba)
        }
        
        # Additional metrics from confusion matrix
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
            })
        
        # ROC curve data
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        metrics['optimal_threshold'] = float(thresholds[optimal_idx])
        metrics['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_pr_idx = np.argmax(f1_scores)
        metrics['optimal_pr_threshold'] = float(pr_thresholds[optimal_pr_idx])
        metrics['precision_recall_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
        
        logger.info(f"Evaluation completed for {self.model_name}")
        
        return metrics
    
    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Calculate all standard metrics.
        
        Returns:
            Dictionary of metric -> value
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_test, self.y_pred, zero_division=0),
            'f1': f1_score(self.y_test, self.y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba)
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: list) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance or None
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            return importance_dict
        return None
    
    def threshold_analysis(self, thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Analyze performance at different thresholds.
        
        Args:
            thresholds: Array of thresholds to test. If None, uses default range
            
        Returns:
            DataFrame with metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            
            results.append({
                'threshold': threshold,
                'precision': precision_score(self.y_test, y_pred_thresh, zero_division=0),
                'recall': recall_score(self.y_test, y_pred_thresh, zero_division=0),
                'f1': f1_score(self.y_test, y_pred_thresh, zero_division=0)
            })
        
        df = pd.DataFrame(results)
        return df

