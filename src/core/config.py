"""
Yapılandırma Modülü
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy
import yaml
import logging

from src.core.types import PathLike

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Yapılandırma ile ilgili hatalar için özel istisna sınıfı"""
    pass

def load_config(config_path: Optional[PathLike] = None) -> Dict[str, Any]:
    """Yapılandırma dosyasını yükler
    
    Args:
        config_path: Yapılandırma dosyası yolu
        
    Returns:
        Yapılandırma sözlüğü
        
    Raises:
        ConfigError: Yapılandırma yükleme veya doğrulama hatası
        FileNotFoundError: Yapılandırma dosyası bulunamadığında
    """
    try:
        # Varsayılan yapılandırma
        config = {
            'app': {
                'name': 'Otuken3D',
                'version': '0.1.0',
                'description': '3B Model İşleme API'
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout': 60
            },
            'storage': {
                'upload_dir': 'uploads',
                'output_dir': 'outputs', 
                'temp_dir': 'temp',
                'max_file_size': 100 * 1024 * 1024  # 100MB
            },
            'processing': {
                'max_vertices': 100000,
                'max_faces': 50000,
                'texture_size': 2048,
                'texture_quality': 90
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/otuken3d.log'
            },
            'security': {
                'allowed_formats': ['.obj', '.stl', '.ply', '.glb', '.gltf', '.fbx', '.dae'],
                'allowed_origins': ['*'],
                'max_batch_size': 10
            }
        }

        if config_path:
            config_path = Path(config_path)
            
            if not config_path.is_file():
                raise FileNotFoundError(f"Yapılandırma dosyası bulunamadı: {config_path}")
                
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    
                if not isinstance(user_config, dict):
                    raise ConfigError("Yapılandırma dosyası geçerli bir YAML sözlüğü değil")
                    
                # Kullanıcı yapılandırmasını birleştir
                _merge_config(config, user_config)
                
            except yaml.YAMLError as e:
                raise ConfigError(f"YAML ayrıştırma hatası: {str(e)}")
            except Exception as e:
                raise ConfigError(f"Yapılandırma yükleme hatası: {str(e)}")
                
        # Çevre değişkenlerinden yükle
        _load_env_config(config)
        
        # Yapılandırmayı doğrula
        _validate_config(config)
        
        return config
        
    except Exception as e:
        logger.error(f"Yapılandırma yükleme hatası: {str(e)}")
        raise
    
def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """İki yapılandırmayı birleştirir
    
    Args:
        base: Temel yapılandırma
        override: Üzerine yazılacak yapılandırma
        
    Note:
        Override'daki yeni anahtarlar base'e eklenir.
        Değerler derin kopyalama ile eklenir.
    """
    for key, value in override.items():
        if key in base:
            if isinstance(base[key], dict) and isinstance(value, dict):
                _merge_config(base[key], value)
            else:
                base[key] = deepcopy(value)
        else:
            base[key] = deepcopy(value)
            
def _load_env_config(config: Dict[str, Any]) -> None:
    """Çevre değişkenlerinden yapılandırma yükler
    
    Args:
        config: Yapılandırma sözlüğü
        
    Raises:
        ConfigError: Çevre değişkeni dönüştürme hatası
    """
    env_map = {
        'OTUKEN3D_HOST': ('server', 'host', str),
        'OTUKEN3D_PORT': ('server', 'port', int),
        'OTUKEN3D_WORKERS': ('server', 'workers', int),
        'OTUKEN3D_TIMEOUT': ('server', 'timeout', int),
        'OTUKEN3D_UPLOAD_DIR': ('storage', 'upload_dir', str),
        'OTUKEN3D_OUTPUT_DIR': ('storage', 'output_dir', str),
        'OTUKEN3D_TEMP_DIR': ('storage', 'temp_dir', str),
        'OTUKEN3D_MAX_FILE_SIZE': ('storage', 'max_file_size', int),
        'OTUKEN3D_LOG_LEVEL': ('logging', 'level', str),
        'OTUKEN3D_LOG_FILE': ('logging', 'file', str)
    }
    
    for env_key, (section, key, type_func) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            try:
                # Yapılandırma yolunu izle
                current = config
                for part in section.split('.'):
                    current = current[part]
                
                # Değeri dönüştür ve ata
                try:
                    current[key] = type_func(value)
                except ValueError as e:
                    raise ConfigError(f"Çevre değişkeni dönüştürme hatası - {env_key}: {str(e)}")
                    
            except KeyError as e:
                logger.warning(f"Geçersiz yapılandırma yolu: {section}.{key}")
                
def _validate_config(config: Dict[str, Any]) -> None:
    """Yapılandırmayı doğrular
    
    Args:
        config: Yapılandırma sözlüğü
        
    Raises:
        ConfigError: Doğrulama hatası
    """
    required_fields = {
        'app': ['name', 'version'],
        'server': ['host', 'port'],
        'storage': ['upload_dir', 'output_dir', 'temp_dir'],
        'logging': ['level', 'format', 'file']
    }
    
    for section, fields in required_fields.items():
        if section not in config:
            raise ConfigError(f"Eksik yapılandırma bölümü: {section}")
            
        for field in fields:
            if field not in config[section]:
                raise ConfigError(f"Eksik yapılandırma alanı: {section}.{field}")
                
    # Port numarası kontrolü
    if not (1024 <= config['server']['port'] <= 65535):
        raise ConfigError("Port numarası 1024-65535 aralığında olmalıdır")
        
    # Dizin yolları kontrolü
    for dir_key in ['upload_dir', 'output_dir', 'temp_dir']:
        path = Path(config['storage'][dir_key])
        if not path.is_absolute():
            config['storage'][dir_key] = str(Path.cwd() / path) 