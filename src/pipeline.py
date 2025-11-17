"""Main ML pipeline orchestrator."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import logging

from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.scaler import FeatureScaler
from src.eda.analyzer import EDAAnalyzer
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.mlflow_tracker import MLflowTracker
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MLPipeline:
    """Main ML pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ML pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        setup_logger(
            level=self.config.get('logging.level', 'INFO'),
            log_file=self.config.get('logging.file', 'logs/pipeline.log')
        )
        
        self.mlflow_tracker = None
        if self.config.get('mlflow.enabled', True):
            self.mlflow_tracker = MLflowTracker(
                tracking_uri=self.config.get('mlflow.tracking_uri', 'sqlite:///mlruns.db'),
                experiment_name=self.config.get('mlflow.experiment_name', 'loan_default_prediction')
            )
        
        logger.info("Initialized MLPipeline")
    
    def run_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run exploratory data analysis.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            EDA report dictionary
        """
        logger.info("Starting EDA")
        eda = EDAAnalyzer(df, target_col=self.config.get('data.target_column', 'Default'))
        report = eda.generate_report()
        logger.info("EDA completed")
        return report
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> tuple:
        """
        Preprocess data.
        
        Args:
            df: DataFrame to preprocess
            fit: Whether to fit preprocessors (True for training, False for inference)
            
        Returns:
            Tuple of (processed_df, preprocessor_dict)
        """
        logger.info("Starting data preprocessing")
        
        df = df.copy()
        target_col = self.config.get('data.target_column', 'Default')
        
        # Initialize preprocessors
        cleaner = DataCleaner(target_col=target_col)
        encoder = CategoricalEncoder()
        feature_engineer = FeatureEngineer()
        scaler = FeatureScaler(
            scaler_type=self.config.get('preprocessing.scaler_type', 'robust')
        )
        
        # Clean data
        if self.config.get('preprocessing.handle_missing_values', True):
            df = cleaner.handle_missing_values(df)
        
        if self.config.get('preprocessing.handle_outliers', True):
            df = cleaner.handle_outliers(
                df,
                method=self.config.get('preprocessing.outlier_method', 'cap')
            )
        
        # Encode categorical
        if self.config.get('preprocessing.encode_categorical', True):
            if fit:
                df = encoder.fit_transform(df)
            else:
                df = encoder.transform(df)
        
        # Feature engineering
        if self.config.get('feature_engineering.enabled', True):
            df = feature_engineer.create_features(df)
        
        # Store preprocessors
        preprocessors = {
            'cleaner': cleaner,
            'encoder': encoder,
            'feature_engineer': feature_engineer,
            'scaler': scaler
        }
        
        logger.info(f"Preprocessing completed. Shape: {df.shape}")
        
        return df, preprocessors
    
    def train(
        self,
        data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train models.
        
        Args:
            data_path: Optional path to training data (overrides config)
            
        Returns:
            Dictionary with training results
        """
        logger.info("="*80)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*80)
        
        # Load data
        if data_path is None:
            data_path = self.config.get('data.train_path', 'data/train.csv')
        
        data_loader = DataIngestion(data_path)
        df = data_loader.load_data()
        
        # EDA
        if self.config.get('eda.enabled', True):
            eda_report = self.run_eda(df)
        
        # Preprocess
        df_processed, preprocessors = self.preprocess_data(df, fit=True)
        
        # Prepare features and target
        target_col = self.config.get('data.target_column', 'Default')
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Train-test split
        test_size = self.config.get('model.test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.config.get('model.random_state', 42),
            stratify=y
        )
        
        # Scale features
        if self.config.get('preprocessing.scale_features', True):
            X_train_scaled = preprocessors['scaler'].fit_transform(X_train)
            X_test_scaled = preprocessors['scaler'].transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Train models
        trainer = ModelTrainer(random_state=self.config.get('model.random_state', 42))
        
        imbalance_method = None
        if self.config.get('model.handle_imbalance', True):
            imbalance_method = self.config.get('model.imbalance_method', 'smote')
        
        results = trainer.train(
            X_train_scaled,
            y_train.values,
            X_test_scaled,
            y_test.values,
            handle_imbalance_method=imbalance_method
        )
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model(
            metric=self.config.get('training.scoring_metric', 'roc_auc')
        )
        
        # Evaluate best model
        evaluator = ModelEvaluator(
            best_model,
            X_test_scaled,
            y_test.values,
            results[best_model_name]['y_pred'],
            results[best_model_name]['y_pred_proba'],
            best_model_name
        )
        
        evaluation_metrics = evaluator.calculate_all_metrics()
        
        # Log to MLflow
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run(run_name=f"train_{best_model_name}")
            
            # Log parameters
            params = {
                'model': best_model_name,
                'test_size': test_size,
                'imbalance_method': imbalance_method or 'none',
                'scaler_type': self.config.get('preprocessing.scaler_type', 'robust')
            }
            self.mlflow_tracker.log_params(params)
            
            # Log metrics
            self.mlflow_tracker.log_metrics(evaluation_metrics)
            
            # Log model
            if self.config.get('mlflow.log_models', True):
                self.mlflow_tracker.log_model(best_model, artifact_path="model")
            
            self.mlflow_tracker.end_run()
        
        # Save models and preprocessors
        model_dir = Path(self.config.get('model_registry.registry_path', 'models'))
        model_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = model_dir / self.config.get('model_registry.best_model_name', 'best_model.pkl')
        joblib.dump(best_model, best_model_path)
        logger.info(f"Saved best model to {best_model_path}")
        
        preprocessor_path = model_dir / self.config.get('model_registry.preprocessor_name', 'preprocessor.pkl')
        joblib.dump(preprocessors, preprocessor_path)
        logger.info(f"Saved preprocessors to {preprocessor_path}")
        
        # Prepare results
        training_results = {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'preprocessors': preprocessors,
            'evaluation_metrics': evaluation_metrics,
            'all_results': results,
            'comparison_df': trainer.get_comparison_df(),
            'feature_names': X.columns.tolist(),
            'X_test': X_test_scaled,
            'y_test': y_test.values
        }
        
        logger.info("="*80)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("="*80)
        
        return training_results
    
    def predict(
        self,
        data_path: str,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            data_path: Path to data file
            model_path: Path to saved model (overrides config)
            preprocessor_path: Path to saved preprocessors (overrides config)
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Starting prediction pipeline")
        
        # Load model and preprocessors
        if model_path is None:
            model_path = Path(self.config.get('model_registry.registry_path', 'models')) / \
                        self.config.get('model_registry.best_model_name', 'best_model.pkl')
        
        if preprocessor_path is None:
            preprocessor_path = Path(self.config.get('model_registry.registry_path', 'models')) / \
                              self.config.get('model_registry.preprocessor_name', 'preprocessor.pkl')
        
        model = joblib.load(model_path)
        preprocessors = joblib.load(preprocessor_path)
        
        # Load and preprocess data
        data_loader = DataIngestion(data_path)
        df = data_loader.load_data()
        
        df_processed, _ = self.preprocess_data(df, fit=False)
        
        # Scale features
        if self.config.get('preprocessing.scale_features', True):
            X_scaled = preprocessors['scaler'].transform(df_processed)
        else:
            X_scaled = df_processed.values
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'risk_level': pd.cut(
                probabilities,
                bins=[0, 0.4, 0.7, 1.0],
                labels=['LOW', 'MEDIUM', 'HIGH']
            )
        })
        
        logger.info(f"Predictions completed: {len(results_df)} samples")
        
        return results_df

