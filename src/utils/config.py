"""Configuration management for the loan default prediction pipeline."""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the ML pipeline."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'data.train_path')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access."""
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return self.get(key) is not None

