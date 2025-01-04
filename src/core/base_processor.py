"""
Temel İşlemci Sınıfı
"""

from typing import Dict, Any, Optional
from .logger import LoggerMixin

class BaseProcessor(LoggerMixin):
    """Tüm işlemci sınıfları için temel sınıf"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: İşlemci yapılandırması
        """
        self.config = config or {}
        
    def validate_config(self) -> None:
        """Yapılandırmayı doğrular"""
        pass
        
    def initialize(self) -> None:
        """İşlemciyi başlatır"""
        self.validate_config()
        
    def cleanup(self) -> None:
        """Kaynakları temizler"""
        pass
        
    def __enter__(self):
        """Context manager girişi"""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkışı"""
        self.cleanup()
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Yapılandırma değeri döndürür
        
        Args:
            key: Yapılandırma anahtarı
            default: Varsayılan değer
            
        Returns:
            Yapılandırma değeri
        """
        return self.config.get(key, default)
        
    def set_config(self, key: str, value: Any) -> None:
        """Yapılandırma değeri atar
        
        Args:
            key: Yapılandırma anahtarı
            value: Yeni değer
        """
        self.config[key] = value
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Yapılandırmayı günceller
        
        Args:
            updates: Güncellenecek değerler
        """
        self.config.update(updates)
        
    def reset_config(self) -> None:
        """Yapılandırmayı sıfırlar"""
        self.config = {} 