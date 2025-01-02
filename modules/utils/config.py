import os
import yaml
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Yapılandırma dosyasını yükle
    
    Args:
        config_path (str, optional): Yapılandırma dosyası yolu. 
                                   Belirtilmezse varsayılan yapılandırma kullanılır.
    
    Returns:
        Dict[str, Any]: Yapılandırma sözlüğü
    """
    if config_path is None:
        # Varsayılan yapılandırma
        config = {
            "model": {
                "hidden_size": 512,
                "num_points": 2048,
                "use_attention": True,
                "num_layers": 6,
                "dropout": 0.1
            },
            "training": {
                "batch_size": 32,
                "num_workers": 4,
                "learning_rate": 1e-4,
                "num_epochs": 100,
                "scheduler_step_size": 20,
                "scheduler_gamma": 0.5,
                "normal_weight": 0.1,
                "save_interval": 10,
                "checkpoint_dir": "models/checkpoints",
                "generate_samples": True,
                "sample_dir": "outputs/samples"
            },
            "data": {
                "dataset_path": "data/datasets/sample",
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "augmentation": {
                    "rotation": True,
                    "scaling": True,
                    "jittering": True
                }
            },
            "logging": {
                "use_wandb": True,
                "project_name": "text2shape",
                "run_name": "baseline",
                "log_interval": 100
            }
        }
    else:
        # Yapılandırma dosyasını yükle
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"UYARI: Yapılandırma dosyası bulunamadı: {config_path}")
            print("Varsayılan yapılandırma kullanılıyor...")
            return load_config()
        except yaml.YAMLError as e:
            print(f"UYARI: Yapılandırma dosyası okunamadı: {str(e)}")
            print("Varsayılan yapılandırma kullanılıyor...")
            return load_config()
    
    # Gerekli dizinleri oluştur
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["training"]["sample_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["data"]["dataset_path"]), exist_ok=True)
    
    return config 