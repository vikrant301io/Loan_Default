"""Data ingestion module for loading CSV data."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles data loading from CSV files."""
    
    def __init__(self, filepath: str):
        """
        Initialize data ingestion.
        
        Args:
            filepath: Path to CSV file or directory containing CSV files
        """
        self.filepath = Path(filepath)
        
        # If it's a directory, find the largest CSV file
        if self.filepath.is_dir():
            csv_files = list(self.filepath.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in directory: {filepath}")
            # Use the largest file (likely the main dataset)
            self.filepath = max(csv_files, key=lambda f: f.stat().st_size)
            logger.info(f"Found CSV file in directory: {self.filepath.name}")
        elif not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Initialized DataIngestion for: {self.filepath}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            Exception: If loading fails
        """
        try:
            logger.info(f"Loading data from: {self.filepath}")
            df = pd.read_csv(self.filepath)
            logger.info(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}")
            raise
    
    def load_data_with_validation(
        self, 
        expected_columns: Optional[list] = None,
        min_rows: int = 1
    ) -> pd.DataFrame:
        """
        Load data with basic validation.
        
        Args:
            expected_columns: Optional list of expected column names
            min_rows: Minimum number of rows required
            
        Returns:
            Loaded and validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        df = self.load_data()
        
        # Validate shape
        if len(df) < min_rows:
            raise ValueError(f"Data has {len(df)} rows, minimum required: {min_rows}")
        
        # Validate columns
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing expected columns: {missing_cols}")
        
        logger.info("✓ Data validation passed")
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data information
        """
        info = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'columns': df.columns.tolist()
        }
        
        logger.info(f"Dataset info: {info['shape'][0]} rows, {info['shape'][1]} columns")
        return info

