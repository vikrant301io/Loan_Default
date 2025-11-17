"""Model training module."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Initialized ModelTrainer with random_state={random_state}")
    
    def handle_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'smote'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance.
        
        Args:
            X: Features
            y: Target
            method: Sampling method ('smote', 'undersample', 'combined')
            
        Returns:
            Balanced X and y
        """
        original_dist = pd.Series(y).value_counts()
        logger.info(f"Original class distribution: {original_dist.to_dict()}")
        
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=self.random_state)
            X_balanced, y_balanced = rus.fit_resample(X, y)
        elif method == 'combined':
            over = SMOTE(sampling_strategy=0.5, random_state=self.random_state)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=self.random_state)
            X_balanced, y_balanced = over.fit_resample(X, y)
            X_balanced, y_balanced = under.fit_resample(X_balanced, y_balanced)
        else:
            logger.warning(f"Unknown method {method}, returning original data")
            return X, y
        
        new_dist = pd.Series(y_balanced).value_counts()
        logger.info(f"New class distribution: {new_dist.to_dict()}")
        
        return X_balanced, y_balanced
    
    def get_models(self) -> Dict[str, Any]:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model name -> model instance
        """
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced',
                C=0.1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1,
                class_weight='balanced'
            )
        }
        
        return models
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        models: Optional[Dict[str, Any]] = None,
        handle_imbalance_method: Optional[str] = 'smote'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            models: Optional dictionary of models. If None, uses default models
            handle_imbalance_method: Method to handle class imbalance
            
        Returns:
            Dictionary of model results
        """
        if models is None:
            models = self.get_models()
        
        # Handle imbalance if requested
        if handle_imbalance_method:
            X_train, y_train = self.handle_imbalance(
                X_train, y_train, method=handle_imbalance_method
            )
        
        self.models = {}
        self.results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Evaluate
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                self.models[name] = model
                self.results[name] = {
                    'model': model,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    **metrics
                }
                
                logger.info(f"✓ {name} - ROC-AUC: {metrics['roc_auc']:.4f}, "
                           f"F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"✗ Error training {name}: {e}")
                continue
        
        return self.results
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any]:
        """
        Get the best model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_name = max(self.results.keys(), key=lambda k: self.results[k][metric])
        best_model = self.results[best_name]['model']
        
        logger.info(f"Best model: {best_name} ({metric}={self.results[best_name][metric]:.4f})")
        
        return best_name, best_model
    
    def get_comparison_df(self) -> pd.DataFrame:
        """
        Get comparison DataFrame of all models.
        
        Returns:
            DataFrame with model metrics
        """
        if not self.results:
            raise ValueError("No models trained yet")
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'ROC-AUC': result['roc_auc']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('ROC-AUC', ascending=False)
        
        return df

