"""MLflow experiment tracking integration."""
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking wrapper."""
    
    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "loan_default_prediction"
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"Initialized MLflowTracker: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
        logger.debug(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics: {list(metrics.keys())}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log model to MLflow.
        
        Args:
            model: Model to log
            artifact_path: Path within the run's artifact directory
            registered_model_name: Optional name for model registry
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"Logged model to: {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact.
        
        Args:
            local_path: Local path to artifact
            artifact_path: Optional path within artifact directory
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log a directory of artifacts.
        
        Args:
            local_dir: Local directory path
            artifact_path: Optional path within artifact directory
        """
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.debug(f"Logged artifacts from: {local_dir}")
    
    def log_data_version(self, data_path: str, dvc_hash: Optional[str] = None):
        """
        Log data version information.
        
        Args:
            data_path: Path to data file
            dvc_hash: Optional DVC hash
        """
        data_info = {
            'data_path': str(data_path),
            'dvc_hash': dvc_hash
        }
        mlflow.log_params(data_info)
        logger.info(f"Logged data version info: {data_path}")
    
    def get_best_run(
        self,
        metric: str = "roc_auc",
        ascending: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to use for comparison
            ascending: Whether to sort ascending
            
        Returns:
            Dictionary with best run information
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
            )
            
            if len(runs) == 0:
                return None
            
            best_run = runs.iloc[0]
            return {
                'run_id': best_run['run_id'],
                'metric': metric,
                'metric_value': best_run[f'metrics.{metric}'],
                'params': {k.replace('params.', ''): v for k, v in best_run.items() if k.startswith('params.')}
            }
        except Exception as e:
            logger.error(f"Error getting best run: {e}")
            return None

