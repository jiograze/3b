"""
Dağıtım Yapılandırması
"""

from typing import Optional, Dict, Any
from pathlib import Path

class DeploymentConfig:
    """Dağıtım yapılandırmasını yöneten sınıf"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Yapılandırma sözlüğü
        """
        self.config = config or {}
        self.validate_config()
        
    def validate_config(self) -> None:
        """Yapılandırmayı doğrular"""
        required_keys = [
            'target_platform',
            'deployment_type',
            'model_format'
        ]
        
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Eksik yapılandırma anahtarları: {missing_keys}")
            
    def get_target_platform(self) -> str:
        """Hedef platformu döndürür"""
        return self.config['target_platform']
        
    def get_deployment_type(self) -> str:
        """Dağıtım tipini döndürür"""
        return self.config['deployment_type']
        
    def get_model_format(self) -> str:
        """Model formatını döndürür"""
        return self.config['model_format']
        
    def get_optimization_level(self) -> int:
        """Optimizasyon seviyesini döndürür"""
        return self.config.get('optimization_level', 0)
        
    def get_compression_enabled(self) -> bool:
        """Sıkıştırmanın etkin olup olmadığını döndürür"""
        return self.config.get('compression_enabled', False)
        
    def get_backup_enabled(self) -> bool:
        """Yedeğin etkin olup olmadığını döndürür"""
        return self.config.get('backup_enabled', True)
        
    def get_target_dir(self) -> Optional[Path]:
        """Hedef dizini döndürür"""
        target_dir = self.config.get('target_dir')
        return Path(target_dir) if target_dir else None
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Yapılandırmayı günceller
        
        Args:
            updates: Güncellenecek değerler
        """
        self.config.update(updates)
        self.validate_config() 