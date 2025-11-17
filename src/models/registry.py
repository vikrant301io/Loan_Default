"""Model registry for saving and loading models."""
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model storage and versioning."""
    
    def __init__(self, registry_path: str = "models"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to model registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ModelRegistry at {self.registry_path}")
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Save model to registry.
        
        Args:
            model: Model instance to save
            model_name: Name of the model
            metadata: Optional metadata dictionary
            version: Optional version string. If None, uses timestamp
            
        Returns:
            Path to saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metadata
        if metadata:
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata to {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """
        Load model from registry.
        
        Args:
            model_name: Name of the model
            version: Version to load. If None, loads latest
            
        Returns:
            Loaded model instance
        """
        model_dir = self.registry_path / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        if version is None:
            # Load latest version
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if not versions:
                raise FileNotFoundError(f"No versions found for {model_name}")
            version = versions[-1]
        
        model_path = model_dir / version / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        return model
    
    def load_metadata(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model metadata.
        
        Args:
            model_name: Name of the model
            version: Version to load. If None, loads latest
            
        Returns:
            Metadata dictionary
        """
        model_dir = self.registry_path / model_name
        
        if version is None:
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if not versions:
                raise FileNotFoundError(f"No versions found for {model_name}")
            version = versions[-1]
        
        metadata_path = model_dir / version / "metadata.json"
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self) -> list:
        """
        List all models in registry.
        
        Returns:
            List of model names
        """
        models = [d.name for d in self.registry_path.iterdir() if d.is_dir()]
        return models
    
    def list_versions(self, model_name: str) -> list:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        model_dir = self.registry_path / model_name
        
        if not model_dir.exists():
            return []
        
        versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
        return versions

