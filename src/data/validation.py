"""Data validation module."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and structure."""
    
    def __init__(self, target_col: str = 'Default'):
        """
        Initialize data validator.
        
        Args:
            target_col: Name of target column
        """
        self.target_col = target_col
        logger.info(f"Initialized DataValidator with target: {target_col}")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Basic checks
        if df.empty:
            results['is_valid'] = False
            results['errors'].append("DataFrame is empty")
            return results
        
        # Check target column
        if self.target_col not in df.columns:
            results['is_valid'] = False
            results['errors'].append(f"Target column '{self.target_col}' not found")
        else:
            # Check target distribution
            target_dist = df[self.target_col].value_counts()
            if len(target_dist) < 2:
                results['warnings'].append("Target has less than 2 classes")
            
            # Check for extreme imbalance
            if len(target_dist) == 2:
                ratio = target_dist.max() / target_dist.min()
                if ratio > 10:
                    results['warnings'].append(f"Extreme class imbalance detected (ratio: {ratio:.2f})")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            high_missing = missing[missing > len(df) * 0.5]
            if len(high_missing) > 0:
                results['warnings'].append(f"Columns with >50% missing: {high_missing.index.tolist()}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate rows")
        
        # Statistics
        results['stats'] = {
            'shape': df.shape,
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': missing.to_dict(),
            'duplicates': duplicates,
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        
        logger.info(f"Validation completed. Valid: {results['is_valid']}, "
                   f"Errors: {len(results['errors'])}, Warnings: {len(results['warnings'])}")
        
        return results
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, str]) -> bool:
        """
        Validate DataFrame schema against expected schema.
        
        Args:
            df: DataFrame to validate
            expected_schema: Dictionary mapping column names to expected dtypes
            
        Returns:
            True if schema matches
        """
        for col, expected_dtype in expected_schema.items():
            if col not in df.columns:
                logger.error(f"Column '{col}' not found in DataFrame")
                return False
            
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype:
                logger.warning(f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}")
        
        return True

