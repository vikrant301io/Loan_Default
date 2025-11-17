"""Data cleaning module."""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning operations."""
    
    def __init__(self, target_col: str = 'Default'):
        """
        Initialize data cleaner.
        
        Args:
            target_col: Name of target column
        """
        self.target_col = target_col
        logger.info("Initialized DataCleaner")
    
    def remove_id_columns(self, df: pd.DataFrame, id_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove ID columns.
        
        Args:
            df: DataFrame to clean
            id_cols: List of ID column names. If None, auto-detects 'ID' column
            
        Returns:
            Cleaned DataFrame
        """
        if id_cols is None:
            id_cols = ['ID'] if 'ID' in df.columns else []
        
        cols_to_remove = [col for col in id_cols if col in df.columns]
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            logger.info(f"Removed ID columns: {cols_to_remove}")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame without duplicates
        """
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            logger.info(f"Removed {duplicates} duplicate rows")
        
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'median',
        categorical_strategy: str = 'mode'
    ) -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: DataFrame to clean
            strategy: Strategy for numerical columns ('median', 'mean', 'zero')
            categorical_strategy: Strategy for categorical columns ('mode', 'unknown')
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            logger.info("No missing values found")
            return df
        
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)
        
        for col in numerical_cols:
            if df[col].isnull().any():
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'zero':
                    fill_value = 0
                else:
                    fill_value = df[col].median()
                
                df[col].fillna(fill_value, inplace=True)
                logger.debug(f"Filled {col} with {strategy}: {fill_value:.2f}")
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if df[col].isnull().any():
                if categorical_strategy == 'mode':
                    fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                else:
                    fill_value = 'Unknown'
                
                df[col].fillna(fill_value, inplace=True)
                logger.debug(f"Filled {col} with {categorical_strategy}: {fill_value}")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return df
    
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'cap',
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle outliers using IQR method.
        
        Args:
            df: DataFrame to clean
            method: Method to handle outliers ('cap', 'remove')
            exclude_cols: Columns to exclude from outlier handling
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        if exclude_cols is None:
            exclude_cols = ['Age_Days', 'Employed_Days', 'Registration_Days', 'ID_Days']
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)
        
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        outliers_count = 0
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0:
                outliers_count += outliers
                if method == 'cap':
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    logger.debug(f"Capped {outliers} outliers in {col}")
                elif method == 'remove':
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    logger.debug(f"Removed {outliers} outliers in {col}")
        
        if outliers_count > 0:
            logger.info(f"Handled {outliers_count} outliers using {method} method")
        
        return df
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform all cleaning operations.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning")
        df = self.remove_id_columns(df)
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.handle_outliers(df)
        logger.info("Data cleaning completed")
        
        return df

