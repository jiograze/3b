import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        required_keys = ["model", "data", "ui"]
        return all(key in config for key in required_keys)
    
    def load_config(self):
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self.config = yaml.safe_load(f)
                if not self.validate_config(self.config):
                    raise ValueError("Invalid configuration structure")
            else:
                self.config = self.default_config()
                self.save_config()
        except Exception as e:
            raise ConfigError(f"Configuration error: {str(e)}")
    
    def default_config(self):
        """Default configuration settings"""
        return {
            "model": {
                "checkpoint_dir": "models/checkpoints",
                "batch_size": 32,
                "learning_rate": 1e-4
            },
            "data": {
                "database_path": "data/database.sqlite",
                "model_dir": "data/3d_models",
                "image_dir": "data/images"
            },
            "ui": {
                "theme": "dark",
                "viewer_width": 800,
                "viewer_height": 600
            }
        }
    
    def save_config(self):
        """Save configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

class ConfigError(Exception):
    pass
