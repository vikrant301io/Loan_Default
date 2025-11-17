"""Unit tests for ML pipeline."""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.pipeline import MLPipeline


class TestMLPipeline:
    """Test ML pipeline functionality."""
    
    @pytest.fixture
    def sample_data_file(self):
        """Create sample data file."""
        df = pd.DataFrame({
            'ID': range(1, 101),
            'Client_Income': np.random.uniform(30000, 100000, 100),
            'Credit_Amount': np.random.uniform(50000, 200000, 100),
            'Loan_Annuity': np.random.uniform(1000, 5000, 100),
            'Age_Days': -np.random.randint(7000, 25000, 100),
            'Default': np.random.choice([0, 1], 100, p=[0.85, 0.15])
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        # Create minimal config
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        if config_path.exists():
            pipeline = MLPipeline(config_path=str(config_path))
            assert pipeline.config is not None
    
    def test_preprocess_data(self, sample_data_file):
        """Test data preprocessing."""
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        pipeline = MLPipeline(config_path=str(config_path))
        
        # Load data
        from src.data.ingestion import DataIngestion
        loader = DataIngestion(sample_data_file)
        df = loader.load_data()
        
        # Preprocess
        df_processed, preprocessors = pipeline.preprocess_data(df, fit=True)
        
        assert df_processed.shape[0] == df.shape[0]
        assert 'cleaner' in preprocessors
        assert 'encoder' in preprocessors
