from pathlib import Path
import yaml
from typing import Dict, Any

class BaseConfig:
    """Temel yapılandırma sınıfı"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/config.yaml"
        self._config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Yapılandırma dosyasını yükle"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ConfigError(f"Yapılandırma dosyası yüklenemedi: {str(e)}")
            
    def get(self, key: str, default: Any = None) -> Any:
        """Yapılandırma değerini getir"""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """Yapılandırma değerini güncelle"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
        
    def save(self) -> None:
        """Yapılandırmayı kaydet"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, allow_unicode=True)
        except Exception as e:
            raise ConfigError(f"Yapılandırma kaydedilemedi: {str(e)}")
            
    @property
    def config(self) -> Dict[str, Any]:
        """Tüm yapılandırmayı getir"""
        return self._config.copy()

class ConfigError(Exception):
    """Yapılandırma hataları için özel istisna sınıfı"""
    pass 