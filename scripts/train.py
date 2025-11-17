"""Training script for the ML pipeline."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MLPipeline
from src.utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train loan default prediction models')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to training data (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level='INFO', log_file='logs/train.log')
    
    try:
        # Initialize pipeline
        pipeline = MLPipeline(config_path=args.config)
        
        # Train
        results = pipeline.train(data_path=args.data)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Best Model: {results['best_model_name']}")
        logger.info("\nMetrics:")
        for metric, value in results['evaluation_metrics'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        logger.info("\nModel Comparison:")
        print(results['comparison_df'].to_string(index=False))
        
        logger.info("\n✓ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

