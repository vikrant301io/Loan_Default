"""Main script to run the complete ML pipeline."""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MLPipeline
from src.utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


def find_dataset_file(dataset_dir: str) -> str:
    """Find CSV file in dataset directory or as direct file."""
    dataset_path = Path(dataset_dir)
    
    # If it's already a CSV file, return it
    if dataset_path.is_file() and dataset_path.suffix == '.csv':
        return str(dataset_path)
    
    # If it's a directory, look for CSV files
    if dataset_path.is_dir():
        csv_files = list(dataset_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
        
        if len(csv_files) == 1:
            return str(csv_files[0])
        
        # If multiple, use the largest one (likely the main dataset)
        largest_file = max(csv_files, key=lambda f: f.stat().st_size)
        logger.info(f"Found {len(csv_files)} CSV files. Using: {largest_file.name}")
        
        return str(largest_file)
    
    # Try to find Dataset.csv on Desktop
    desktop_dataset = Path("C:/Users/ASUS/Desktop/Dataset.csv")
    if desktop_dataset.exists():
        logger.info(f"Found dataset file: {desktop_dataset}")
        return str(desktop_dataset)
    
    raise FileNotFoundError(f"Dataset not found: {dataset_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run Loan Default Prediction Pipeline')
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default=r'C:\Users\ASUS\Desktop\Dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (only run EDA)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level='INFO', log_file='logs/pipeline.log')
    
    try:
        # Find dataset file
        logger.info(f"Looking for dataset in: {args.dataset_dir}")
        dataset_file = find_dataset_file(args.dataset_dir)
        logger.info(f"Using dataset: {dataset_file}")
        
        # Initialize pipeline
        pipeline = MLPipeline(config_path=args.config)
        
        # Update config with dataset path
        pipeline.config.config['data']['raw_data_path'] = dataset_file
        pipeline.config.config['data']['train_path'] = dataset_file
        
        # Load data
        logger.info("Loading data...")
        from src.data.ingestion import DataIngestion
        data_loader = DataIngestion(dataset_file)
        df = data_loader.load_data()
        
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Run EDA
        logger.info("Running Exploratory Data Analysis...")
        eda_report = pipeline.run_eda(df)
        logger.info("EDA completed")
        
        if args.skip_training:
            logger.info("Skipping training as requested")
            return
        
        # Train models
        logger.info("Starting model training...")
        results = pipeline.train(data_path=dataset_file)
        
        # Print results summary
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"\nBest Model: {results['best_model_name']}")
        print("\nModel Performance Metrics:")
        for metric, value in results['evaluation_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nModel Comparison:")
        print(results['comparison_df'].to_string(index=False))
        
        print("\n[SUCCESS] Pipeline completed successfully!")
        print(f"\nModel saved to: models/{pipeline.config.get('model_registry.best_model_name')}")
        print(f"Preprocessor saved to: models/{pipeline.config.get('model_registry.preprocessor_name')}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

