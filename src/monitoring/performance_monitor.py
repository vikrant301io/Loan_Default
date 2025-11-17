"""Model performance monitoring module."""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(
        self,
        metrics: List[str] = None,
        alert_thresholds: Dict[str, float] = None
    ):
        """
        Initialize performance monitor.
        
        Args:
            metrics: List of metrics to monitor
            alert_thresholds: Dictionary of metric -> threshold for alerts
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        if alert_thresholds is None:
            alert_thresholds = {
                'accuracy': 0.7,
                'precision': 0.6,
                'recall': 0.5,
                'f1': 0.6,
                'roc_auc': 0.7
            }
        
        self.metrics = metrics
        self.alert_thresholds = alert_thresholds
        self.performance_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized PerformanceMonitor with metrics: {metrics}")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metric -> value
        """
        metrics_dict = {}
        
        if 'accuracy' in self.metrics:
            metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision' in self.metrics:
            metrics_dict['precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        if 'recall' in self.metrics:
            metrics_dict['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        if 'f1' in self.metrics:
            metrics_dict['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        if 'roc_auc' in self.metrics and y_pred_proba is not None:
            try:
                metrics_dict['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                logger.warning("Could not calculate ROC-AUC (possibly only one class present)")
                metrics_dict['roc_auc'] = 0.0
        
        return metrics_dict
    
    def check_alerts(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if any metrics are below thresholds.
        
        Args:
            metrics: Dictionary of metric -> value
            
        Returns:
            Dictionary of metric -> alert_status
        """
        alerts = {}
        
        for metric, value in metrics.items():
            if metric in self.alert_thresholds:
                threshold = self.alert_thresholds[metric]
                alerts[metric] = value < threshold
        
        return alerts
    
    def monitor(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Monitor model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            metadata: Additional metadata (timestamp, model_version, etc.)
            
        Returns:
            Dictionary with monitoring results
        """
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        alerts = self.check_alerts(metrics)
        
        monitoring_result = {
            'metrics': metrics,
            'alerts': alerts,
            'has_alerts': any(alerts.values()),
            'metadata': metadata or {}
        }
        
        # Store in history
        self.performance_history.append(monitoring_result)
        
        if monitoring_result['has_alerts']:
            logger.warning(f"Performance alerts triggered: {alerts}")
        else:
            logger.info("Performance monitoring: All metrics within thresholds")
        
        return monitoring_result
    
    def get_performance_history(self) -> pd.DataFrame:
        """
        Get performance history as DataFrame.
        
        Returns:
            DataFrame with performance history
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        history_data = []
        for record in self.performance_history:
            row = {**record['metrics'], **record['metadata']}
            row['has_alerts'] = record['has_alerts']
            history_data.append(row)
        
        return pd.DataFrame(history_data)
    
    def detect_degradation(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        degradation_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect performance degradation compared to baseline.
        
        Args:
            baseline_metrics: Baseline performance metrics
            current_metrics: Current performance metrics
            degradation_threshold: Relative degradation threshold (e.g., 0.05 = 5%)
            
        Returns:
            Dictionary with degradation analysis
        """
        degradation = {}
        
        for metric in baseline_metrics.keys():
            if metric in current_metrics:
                baseline_val = baseline_metrics[metric]
                current_val = current_metrics[metric]
                
                if baseline_val > 0:
                    relative_change = (baseline_val - current_val) / baseline_val
                    degradation[metric] = {
                        'baseline': baseline_val,
                        'current': current_val,
                        'absolute_change': current_val - baseline_val,
                        'relative_change': relative_change,
                        'is_degraded': relative_change > degradation_threshold
                    }
        
        has_degradation = any(d.get('is_degraded', False) for d in degradation.values())
        
        result = {
            'has_degradation': has_degradation,
            'degradation_details': degradation
        }
        
        if has_degradation:
            logger.warning(f"Performance degradation detected: {degradation}")
        
        return result

