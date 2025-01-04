import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self, base_config_path: Optional[str] = None):
        """
        Konfigürasyon yöneticisi.
        Args:
            base_config_path: Temel konfigürasyon dosyasının yolu
        """
        if base_config_path is None:
            base_config_path = "modules/training/config.yaml"
            
        self.base_config = self.load_config(base_config_path)
        
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        """Konfigürasyon dosyasını yükle"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def merge_configs(self, specific_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Base config ile spesifik config'i birleştir.
        Derin birleştirme yapar (nested dictionary'ler için).
        """
        def deep_merge(base: Dict[str, Any], specific: Dict[str, Any]) -> Dict[str, Any]:
            merged = base.copy()
            for key, value in specific.items():
                if (
                    key in merged and 
                    isinstance(merged[key], dict) and 
                    isinstance(value, dict)
                ):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged
            
        return deep_merge(self.base_config, specific_config)
        
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Belirli bir model tipi için konfigürasyon al"""
        model_config_paths = {
            "text2shape": "models/text2shape/model_config.yaml",
            "image2shape": "models/image2shape/model_config.yaml",
            "otuken3d": "models/otuken3d/configs/model_config.yaml"
        }
        
        if model_type not in model_config_paths:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
            
        specific_config = self.load_config(model_config_paths[model_type])
        return self.merge_configs(specific_config)
        
    def save_config(self, config: Dict[str, Any], path: str):
        """Konfigürasyonu kaydet"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
    def update_base_config(self, updates: Dict[str, Any]):
        """Base config'i güncelle"""
        self.base_config = self.merge_configs(updates) 