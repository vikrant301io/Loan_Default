"""Prediction script for the ML pipeline."""
import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MLPipeline
from src.utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make predictions on new data')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data file for prediction'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file (overrides config)'
    )
    parser.add_argument(
        '--preprocessor',
        type=str,
        default=None,
        help='Path to preprocessor file (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level='INFO', log_file='logs/predict.log')
    
    try:
        # Initialize pipeline
        pipeline = MLPipeline(config_path=args.config)
        
        # Make predictions
        predictions = pipeline.predict(
            data_path=args.data,
            model_path=args.model,
            preprocessor_path=args.preprocessor
        )
        
        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Predictions saved to {output_path}")
        logger.info(f"\nPrediction Summary:")
        logger.info(f"  Total samples: {len(predictions)}")
        logger.info(f"  Default predictions: {predictions['prediction'].sum()}")
        logger.info(f"  High risk: {(predictions['risk_level'] == 'HIGH').sum()}")
        logger.info(f"  Medium risk: {(predictions['risk_level'] == 'MEDIUM').sum()}")
        logger.info(f"  Low risk: {(predictions['risk_level'] == 'LOW').sum()}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

