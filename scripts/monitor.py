"""Monitoring script for model performance and drift detection."""
import argparse
import sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestion
from src.monitoring.drift_detection import DriftDetector
from src.monitoring.performance_monitor import PerformanceMonitor
from src.utils.config import Config
from src.utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


def main():
    """Main monitoring function."""
    parser = argparse.ArgumentParser(description='Monitor model performance and data drift')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--reference-data',
        type=str,
        required=True,
        help='Path to reference data'
    )
    parser.add_argument(
        '--current-data',
        type=str,
        required=True,
        help='Path to current data'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='monitoring_report.csv',
        help='Path to save monitoring report'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(level='INFO', log_file='logs/monitor.log')
    
    try:
        config = Config(args.config)
        
        # Load data
        logger.info("Loading reference and current data...")
        ref_loader = DataIngestion(args.reference_data)
        curr_loader = DataIngestion(args.current_data)
        
        ref_df = ref_loader.load_data()
        curr_df = curr_loader.load_data()
        
        # Drift detection
        logger.info("Running drift detection...")
        drift_detector = DriftDetector(
            threshold=config.get('monitoring.drift_detection.threshold', 0.05),
            method=config.get('monitoring.drift_detection.method', 'ks_test')
        )
        
        drift_report = drift_detector.generate_drift_report(ref_df, curr_df)
        
        logger.info("\nDrift Detection Results:")
        print(drift_report.to_string(index=False))
        
        # Performance monitoring (if model provided)
        if args.model:
            logger.info("Running performance monitoring...")
            
            model = joblib.load(args.model)
            
            # Get predictions
            target_col = config.get('data.target_column', 'Default')
            if target_col in curr_df.columns:
                X_curr = curr_df.drop(columns=[target_col])
                y_curr = curr_df[target_col].values
                
                y_pred = model.predict(X_curr.values)
                y_pred_proba = model.predict_proba(X_curr.values)[:, 1]
                
                # Monitor performance
                monitor = PerformanceMonitor(
                    metrics=config.get('monitoring.performance_monitoring.metrics', 
                                     ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']),
                    alert_thresholds=config.get('monitoring.performance_monitoring.alert_thresholds', {})
                )
                
                monitoring_result = monitor.monitor(y_curr, y_pred, y_pred_proba)
                
                logger.info("\nPerformance Monitoring Results:")
                logger.info(f"Metrics: {monitoring_result['metrics']}")
                logger.info(f"Alerts: {monitoring_result['alerts']}")
                logger.info(f"Has Alerts: {monitoring_result['has_alerts']}")
        
        # Save report
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drift_report.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Monitoring report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Monitoring failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

