"""Data drift detection module."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift between reference and current data."""
    
    def __init__(self, threshold: float = 0.05, method: str = 'ks_test'):
        """
        Initialize drift detector.
        
        Args:
            threshold: P-value threshold for drift detection
            method: Method to use ('ks_test', 'psi', 'mmd')
        """
        self.threshold = threshold
        self.method = method
        logger.info(f"Initialized DriftDetector with method={method}, threshold={threshold}")
    
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            reference: Reference data distribution
            current: Current data distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins based on reference data
        _, bin_edges = pd.cut(reference, bins=bins, retbins=True, duplicates='drop')
        
        # Handle edge case where all values are the same
        if len(bin_edges) < 2:
            return 0.0
        
        # Discretize both distributions
        reference_binned = pd.cut(reference, bins=bin_edges, include_lowest=True, duplicates='drop')
        current_binned = pd.cut(current, bins=bin_edges, include_lowest=True, duplicates='drop')
        
        # Calculate proportions
        ref_props = reference_binned.value_counts(normalize=True).sort_index()
        curr_props = current_binned.value_counts(normalize=True).sort_index()
        
        # Align indices
        all_bins = ref_props.index.union(curr_props.index)
        ref_props = ref_props.reindex(all_bins, fill_value=0.0001)
        curr_props = curr_props.reindex(all_bins, fill_value=0.0001)
        
        # Calculate PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return psi
    
    def detect_drift_ks(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Tuple[bool, float, float]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            reference: Reference data
            current: Current data
            
        Returns:
            Tuple of (is_drift, p_value, statistic)
        """
        try:
            statistic, p_value = stats.ks_2samp(reference, current)
            is_drift = p_value < self.threshold
            return is_drift, p_value, statistic
        except Exception as e:
            logger.error(f"Error in KS test: {e}")
            return False, 1.0, 0.0
    
    def detect_drift_psi(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Tuple[bool, float]:
        """
        Detect drift using PSI.
        
        Args:
            reference: Reference data
            current: Current data
            
        Returns:
            Tuple of (is_drift, psi_value)
        """
        try:
            psi = self.calculate_psi(reference, current)
            # PSI thresholds: < 0.1 (no drift), 0.1-0.25 (minor), > 0.25 (major)
            is_drift = psi > 0.25
            return is_drift, psi
        except Exception as e:
            logger.error(f"Error in PSI calculation: {e}")
            return False, 0.0
    
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        columns: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Detect drift for multiple columns.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            columns: Columns to check (if None, checks all numerical columns)
            
        Returns:
            Dictionary with drift detection results
        """
        if columns is None:
            columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            'drift_detected': False,
            'drifted_features': [],
            'details': {}
        }
        
        for col in columns:
            if col not in reference_data.columns or col not in current_data.columns:
                continue
            
            ref_series = reference_data[col].dropna()
            curr_series = current_data[col].dropna()
            
            if len(ref_series) == 0 or len(curr_series) == 0:
                continue
            
            if self.method == 'ks_test':
                is_drift, p_value, statistic = self.detect_drift_ks(ref_series, curr_series)
                results['details'][col] = {
                    'drift_detected': is_drift,
                    'p_value': p_value,
                    'statistic': statistic,
                    'method': 'ks_test'
                }
            elif self.method == 'psi':
                is_drift, psi_value = self.detect_drift_psi(ref_series, curr_series)
                results['details'][col] = {
                    'drift_detected': is_drift,
                    'psi': psi_value,
                    'method': 'psi'
                }
            
            if results['details'][col].get('drift_detected', False):
                results['drift_detected'] = True
                results['drifted_features'].append(col)
        
        logger.info(f"Drift detection completed: {len(results['drifted_features'])} features drifted")
        
        return results
    
    def generate_drift_report(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Generate drift detection report.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            columns: Columns to check
            
        Returns:
            DataFrame with drift report
        """
        results = self.detect_drift(reference_data, current_data, columns)
        
        report_data = []
        for col, details in results['details'].items():
            report_data.append({
                'Feature': col,
                'Drift_Detected': details['drift_detected'],
                'Method': details['method'],
                **{k: v for k, v in details.items() if k not in ['drift_detected', 'method']}
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Drift_Detected', ascending=False)
        
        return report_df

