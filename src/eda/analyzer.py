"""Comprehensive Exploratory Data Analysis for Loan Default."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Comprehensive Exploratory Data Analysis for Loan Default."""
    
    def __init__(self, df: pd.DataFrame, target_col: str = 'Default'):
        """
        Initialize EDA Analyzer.
        
        Args:
            df: DataFrame to analyze
            target_col: Name of target column
        """
        self.df = df.copy()
        self.target_col = target_col
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        logger.info(f"Initialized EDAAnalyzer with {len(df)} rows, {len(df.columns)} columns")
    
    def analyze_target(self) -> Tuple[pd.Series, pd.Series]:
        """
        Analyze target variable distribution.
        
        Returns:
            Tuple of (value_counts, normalized_counts)
        """
        logger.info("Analyzing target variable")
        
        if self.target_col not in self.df.columns:
            logger.warning(f"Target column '{self.target_col}' not found")
            return pd.Series(), pd.Series()
        
        target_counts = self.df[self.target_col].value_counts()
        target_props = self.df[self.target_col].value_counts(normalize=True)
        
        logger.info(f"Target distribution: {target_counts.to_dict()}")
        logger.info(f"Target proportions: {target_props.to_dict()}")
        
        return target_counts, target_props
    
    def identify_feature_types(self) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical features.
        
        Returns:
            Tuple of (numerical_cols, categorical_cols)
        """
        # Numerical columns
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'ID' in self.numerical_cols:
            self.numerical_cols.remove('ID')
        if self.target_col in self.numerical_cols:
            self.numerical_cols.remove(self.target_col)
        
        # Categorical columns
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Also include binary numerical columns as categorical
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col not in ['ID', self.target_col] and col not in self.numerical_cols:
                unique_vals = self.df[col].nunique()
                if unique_vals <= 5:
                    self.categorical_cols.append(col)
        
        logger.info(f"Identified {len(self.numerical_cols)} numerical and "
                   f"{len(self.categorical_cols)} categorical features")
        
        return self.numerical_cols, self.categorical_cols
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for numerical features.
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.numerical_cols:
            self.identify_feature_types()
        
        if not self.numerical_cols:
            logger.warning("No numerical columns found")
            return pd.DataFrame()
        
        summary = self.df[self.numerical_cols].describe()
        logger.info("Generated summary statistics")
        return summary.T
    
    def detect_skewness(self, threshold: float = 1.0) -> pd.Series:
        """
        Detect skewed features.
        
        Args:
            threshold: Absolute skewness threshold
            
        Returns:
            Series of skewed features
        """
        if not self.numerical_cols:
            self.identify_feature_types()
        
        if not self.numerical_cols:
            return pd.Series()
        
        skewness = self.df[self.numerical_cols].skew().sort_values(ascending=False)
        high_skew = skewness[abs(skewness) > threshold]
        
        if len(high_skew) > 0:
            logger.info(f"Found {len(high_skew)} highly skewed features (|skew| > {threshold})")
        else:
            logger.info(f"No highly skewed features found (threshold: {threshold})")
        
        return high_skew
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Returns:
            DataFrame with missing value information
        """
        missing = self.df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing[missing > 0].index,
            'Missing_Count': missing[missing > 0].values,
            'Percentage': (missing[missing > 0].values / len(self.df) * 100).round(2)
        })
        
        if len(missing_df) > 0:
            logger.info(f"Found missing values in {len(missing_df)} columns")
        else:
            logger.info("No missing values found")
        
        return missing_df
    
    def analyze_correlations(self, top_n: int = 15) -> pd.DataFrame:
        """
        Analyze correlations with target variable.
        
        Args:
            top_n: Number of top correlations to return
            
        Returns:
            DataFrame with correlations
        """
        if not self.numerical_cols:
            self.identify_feature_types()
        
        corr_features = [col for col in self.numerical_cols if col in self.df.columns]
        if self.target_col in self.df.columns:
            corr_features.append(self.target_col)
        
        if len(corr_features) < 2:
            logger.warning("Not enough numerical features for correlation analysis")
            return pd.DataFrame()
        
        corr_matrix = self.df[corr_features].corr()
        
        if self.target_col in corr_matrix.columns:
            target_corr = corr_matrix[self.target_col].abs().sort_values(ascending=False)
            top_corr = target_corr[1:top_n+1]  # Exclude target itself
            logger.info(f"Top {len(top_corr)} correlations with target identified")
            return top_corr
        
        return pd.DataFrame()
    
    def detect_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            DataFrame with outlier information
        """
        if not self.numerical_cols:
            self.identify_feature_types()
        
        outlier_summary = []
        
        for col in self.numerical_cols[:20]:  # Analyze first 20
            if col not in self.df.columns:
                continue
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            else:  # zscore
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = (z_scores > 3).sum()
                lower_bound = None
                upper_bound = None
            
            outlier_pct = (outliers / len(self.df)) * 100
            
            outlier_summary.append({
                'Feature': col,
                'Outliers': outliers,
                'Percentage': f"{outlier_pct:.2f}%",
                'Lower_Bound': f"{lower_bound:.2f}" if lower_bound is not None else "N/A",
                'Upper_Bound': f"{upper_bound:.2f}" if upper_bound is not None else "N/A"
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        logger.info(f"Outlier analysis completed for {len(outlier_df)} features")
        
        return outlier_df
    
    def get_feature_target_relationships(self) -> pd.DataFrame:
        """
        Analyze relationships between features and target.
        
        Returns:
            DataFrame with feature-target relationships
        """
        if self.target_col not in self.df.columns:
            logger.warning(f"Target column '{self.target_col}' not found")
            return pd.DataFrame()
        
        relationships = []
        
        # Numerical features
        if not self.numerical_cols:
            self.identify_feature_types()
        
        for col in self.numerical_cols[:10]:  # First 10
            if col in self.df.columns:
                default_mean = self.df.groupby(self.target_col)[col].mean()
                if len(default_mean) == 2:
                    relationships.append({
                        'Feature': col,
                        'Type': 'Numerical',
                        'No_Default_Mean': default_mean.get(0, 0),
                        'Default_Mean': default_mean.get(1, 0),
                        'Difference': abs(default_mean.get(1, 0) - default_mean.get(0, 0))
                    })
        
        # Categorical features
        for col in self.categorical_cols[:10]:  # First 10
            if col in self.df.columns:
                default_rate = self.df.groupby(col)[self.target_col].mean()
                relationships.append({
                    'Feature': col,
                    'Type': 'Categorical',
                    'Default_Rate_Range': f"{default_rate.min():.3f} - {default_rate.max():.3f}",
                    'Categories': len(default_rate)
                })
        
        relationships_df = pd.DataFrame(relationships)
        logger.info(f"Analyzed relationships for {len(relationships_df)} features")
        
        return relationships_df
    
    def generate_report(self) -> dict:
        """
        Generate comprehensive EDA report.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Generating comprehensive EDA report")
        
        report = {
            'dataset_shape': self.df.shape,
            'target_distribution': self.analyze_target()[0].to_dict(),
            'missing_values': self.analyze_missing_values().to_dict('records'),
            'numerical_features': len(self.numerical_cols) if self.numerical_cols else 0,
            'categorical_features': len(self.categorical_cols) if self.categorical_cols else 0,
            'skewed_features': self.detect_skewness().to_dict(),
            'top_correlations': self.analyze_correlations().to_dict(),
            'outliers': self.detect_outliers().to_dict('records')
        }
        
        logger.info("EDA report generated successfully")
        return report

