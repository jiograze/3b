import yaml
from pathlib import Path

class Config:
    def __init__(self, config_dict):
        self.model_path = config_dict.get('model_path', 'models/checkpoints')
        self.data_path = config_dict.get('data_path', 'data')
        self.device = config_dict.get('device', 'cuda')
        self.batch_size = config_dict.get('batch_size', 1)

def load_config(config_path='config/config.yaml'):
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    return Config(config_dict)
