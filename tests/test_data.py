"""Unit tests for data ingestion and validation."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator


class TestDataIngestion:
    """Test data ingestion functionality."""
    
    def test_load_data_success(self):
        """Test successful data loading."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'ID': [1, 2, 3],
                'Client_Income': [50000, 60000, 70000],
                'Default': [0, 1, 0]
            })
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataIngestion(temp_path)
            loaded_df = loader.load_data()
            
            assert isinstance(loaded_df, pd.DataFrame)
            assert len(loaded_df) == 3
            assert 'Client_Income' in loaded_df.columns
        finally:
            os.unlink(temp_path)
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            DataIngestion('nonexistent_file.csv')
    
    def test_load_data_with_validation(self):
        """Test data loading with validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'ID': [1, 2, 3],
                'Client_Income': [50000, 60000, 70000],
                'Default': [0, 1, 0]
            })
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataIngestion(temp_path)
            loaded_df = loader.load_data_with_validation(
                expected_columns=['ID', 'Client_Income', 'Default'],
                min_rows=2
            )
            
            assert len(loaded_df) == 3
        finally:
            os.unlink(temp_path)
    
    def test_get_data_info(self):
        """Test data info extraction."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'ID': [1, 2, 3],
                'Client_Income': [50000, 60000, 70000],
                'Default': [0, 1, 0]
            })
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loader = DataIngestion(temp_path)
            df = loader.load_data()
            info = loader.get_data_info(df)
            
            assert 'shape' in info
            assert 'memory_usage_mb' in info
            assert 'columns' in info
            assert info['shape'] == (3, 3)
        finally:
            os.unlink(temp_path)


class TestDataValidator:
    """Test data validation functionality."""
    
    def test_validate_data(self):
        """Test DataFrame validation."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Client_Income': [50000, 60000, 70000],
            'Default': [0, 1, 0]
        })
        
        validator = DataValidator()
        result = validator.validate_data(df)
        
        assert result['is_valid'] is True
        assert 'stats' in result
    
    def test_validate_missing_values(self):
        """Test missing value detection."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Client_Income': [50000, np.nan, 70000],
            'Default': [0, 1, 0]
        })
        
        validator = DataValidator()
        result = validator.validate_data(df)
        
        assert result['stats']['missing_values']['Client_Income'] == 1
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        
        validator = DataValidator()
        result = validator.validate_data(df)
        
        assert result['is_valid'] is False
        assert len(result['errors']) > 0

