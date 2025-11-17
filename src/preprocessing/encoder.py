"""Categorical encoding module."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """Handles categorical variable encoding."""
    
    def __init__(self):
        """Initialize categorical encoder."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        logger.info("Initialized CategoricalEncoder")
    
    def fit(self, df: pd.DataFrame, categorical_cols: Optional[list] = None) -> 'CategoricalEncoder':
        """
        Fit encoders on training data.
        
        Args:
            df: Training DataFrame
            categorical_cols: List of categorical columns. If None, auto-detects object columns
            
        Returns:
            Self
        """
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Fitted encoder for {col} ({len(le.classes_)} categories)")
        
        logger.info(f"Fitted encoders for {len(self.label_encoders)} categorical columns")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using fitted encoders.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                mask = df[col].isin(encoder.classes_)
                df.loc[mask, col] = encoder.transform(df.loc[mask, col])
                df.loc[~mask, col] = -1  # Encode unseen as -1
                df[col] = df[col].astype(int)
                logger.debug(f"Encoded {col}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, categorical_cols: Optional[list] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to encode
            categorical_cols: List of categorical columns
            
        Returns:
            Encoded DataFrame
        """
        return self.fit(df, categorical_cols).transform(df)
    
    def save(self, filepath: str):
        """
        Save encoders to disk.
        
        Args:
            filepath: Path to save encoders
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoders, filepath)
        logger.info(f"Saved encoders to {filepath}")
    
    def load(self, filepath: str) -> 'CategoricalEncoder':
        """
        Load encoders from disk.
        
        Args:
            filepath: Path to load encoders from
            
        Returns:
            Self
        """
        self.label_encoders = joblib.load(filepath)
        logger.info(f"Loaded encoders from {filepath}")
        return self

