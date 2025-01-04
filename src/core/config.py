"""
Konfigürasyon Yönetimi
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Konfigürasyon dosyasını yükler"""
    
    # Varsayılan konfigürasyon
    default_config = {
        "app": {
            "name": "Otuken3D",
            "version": "0.1.0",
            "description": "3D Model İşleme ve Dönüştürme API'si"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "timeout": 60
        },
        "storage": {
            "temp_dir": "/tmp/otuken3d",
            "max_upload_size": 100 * 1024 * 1024,  # 100MB
            "allowed_extensions": [
                ".obj", ".stl", ".ply", ".gltf", ".glb",
                ".fbx", ".dae", ".3ds", ".blend"
            ]
        },
        "processing": {
            "max_vertices": 1000000,
            "max_faces": 500000,
            "default_simplify_ratio": 0.5,
            "texture_size_limit": 4096
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/otuken3d.log"
        },
        "security": {
            "api_key_required": False,
            "allowed_origins": ["*"],
            "rate_limit": {
                "requests": 100,
                "period": 60  # saniye
            }
        }
    }
    
    # Konfigürasyon dosyası yolu
    if config_path is None:
        config_path = os.environ.get(
            "OTUKEN3D_CONFIG",
            str(Path(__file__).parent.parent / "config.yml")
        )
    
    # Konfigürasyon dosyasını oku
    config = default_config.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    _deep_update(config, user_config)
        except Exception as e:
            print(f"Konfigürasyon dosyası okuma hatası: {str(e)}")
            print("Varsayılan konfigürasyon kullanılıyor.")
    
    # Çevre değişkenlerinden güncelle
    _update_from_env(config)
    
    return config

def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """İç içe sözlükleri günceller"""
    for key, value in update_dict.items():
        if (
            key in base_dict and 
            isinstance(base_dict[key], dict) and 
            isinstance(value, dict)
        ):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def _update_from_env(config: Dict) -> None:
    """Çevre değişkenlerinden konfigürasyonu günceller"""
    env_mapping = {
        "OTUKEN3D_HOST": ("server", "host"),
        "OTUKEN3D_PORT": ("server", "port"),
        "OTUKEN3D_WORKERS": ("server", "workers"),
        "OTUKEN3D_TEMP_DIR": ("storage", "temp_dir"),
        "OTUKEN3D_MAX_UPLOAD": ("storage", "max_upload_size"),
        "OTUKEN3D_LOG_LEVEL": ("logging", "level"),
        "OTUKEN3D_LOG_FILE": ("logging", "file"),
        "OTUKEN3D_API_KEY_REQUIRED": ("security", "api_key_required"),
        "OTUKEN3D_ALLOWED_ORIGINS": ("security", "allowed_origins")
    }
    
    for env_var, config_path in env_mapping.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Değeri doğru tipe dönüştür
            current = config
            for part in config_path[:-1]:
                current = current[part]
            
            original_value = current[config_path[-1]]
            if isinstance(original_value, bool):
                value = value.lower() in ('true', '1', 'yes')
            elif isinstance(original_value, int):
                value = int(value)
            elif isinstance(original_value, float):
                value = float(value)
            elif isinstance(original_value, list):
                value = value.split(',')
                
            current[config_path[-1]] = value

def get_config() -> Dict[str, Any]:
    """Singleton konfigürasyon nesnesini döndürür"""
    if not hasattr(get_config, "_config"):
        get_config._config = load_config()
    return get_config._config

def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Konfigürasyonu yeniden yükler"""
    if hasattr(get_config, "_config"):
        del get_config._config
    return get_config() 