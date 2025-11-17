"""Feature scaling module."""
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureScaler:
    """Handles feature scaling."""
    
    def __init__(self, scaler_type: str = 'robust'):
        """
        Initialize feature scaler.
        
        Args:
            scaler_type: Type of scaler ('robust', 'standard')
        """
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler_type = scaler_type
        logger.info(f"Initialized FeatureScaler with {scaler_type} scaler")
    
    def fit(self, X):
        """
        Fit scaler on training data.
        
        Args:
            X: Training features
        """
        self.scaler.fit(X)
        logger.info("Fitted scaler on training data")
        return self
    
    def transform(self, X):
        """
        Transform features using fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features
        """
        return self.scaler.transform(X)
    
    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Args:
            X: Features to scale
            
        Returns:
            Scaled features
        """
        return self.scaler.fit_transform(X)
    
    def save(self, filepath: str):
        """
        Save scaler to disk.
        
        Args:
            filepath: Path to save scaler
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, filepath)
        logger.info(f"Saved scaler to {filepath}")
    
    def load(self, filepath: str) -> 'FeatureScaler':
        """
        Load scaler from disk.
        
        Args:
            filepath: Path to load scaler from
            
        Returns:
            Self
        """
        self.scaler = joblib.load(filepath)
        logger.info(f"Loaded scaler from {filepath}")
        return self

