"""
Temel İşlemci Sınıfı
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.core.types import PathLike

class BaseProcessor:
    """Tüm işlemciler için temel sınıf"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: İşlemci yapılandırması
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.required_keys = []
        
        # Yapılandırmayı doğrula
        self.validate_config()
        
    def validate_config(self) -> None:
        """Yapılandırmayı doğrular"""
        if not self.config:
            return
            
        for key in self.required_keys:
            if key not in self.config:
                raise ValueError(f"Eksik yapılandırma anahtarı: {key}")
                
    def validate_file(self, path: PathLike) -> None:
        """Dosyanın varlığını kontrol eder
        
        Args:
            path: Dosya yolu
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
            
    def validate_dir(self, path: PathLike) -> None:
        """Klasörün varlığını kontrol eder
        
        Args:
            path: Klasör yolu
        """
        path = Path(path)
        if not path.is_dir():
            raise NotADirectoryError(f"Klasör bulunamadı: {path}")
            
    def ensure_dir(self, path: PathLike) -> None:
        """Klasörün varlığını garanti eder
        
        Args:
            path: Klasör yolu
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Yapılandırma değerini döndürür
        
        Args:
            key: Yapılandırma anahtarı
            default: Varsayılan değer
            
        Returns:
            Yapılandırma değeri
        """
        return self.config.get(key, default) 