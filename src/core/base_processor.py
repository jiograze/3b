"""
Temel İşlemci Sınıfı
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from src.core.logger import LoggerMixin

class BaseProcessor(LoggerMixin):
    """Tüm işlemciler için temel sınıf"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: İşlemci yapılandırması
        """
        super().__init__()
        self.config = config or {}
        
    def validate_config(self, required_keys: list[str]) -> None:
        """Yapılandırma anahtarlarını doğrular
        
        Args:
            required_keys: Gerekli yapılandırma anahtarları
            
        Raises:
            ValueError: Eksik yapılandırma anahtarı varsa
        """
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Eksik yapılandırma anahtarları: {missing_keys}")
            
    def get_config(self, key: str, default: Any = None) -> Any:
        """Yapılandırma değerini alır
        
        Args:
            key: Yapılandırma anahtarı
            default: Varsayılan değer
            
        Returns:
            Yapılandırma değeri
        """
        return self.config.get(key, default)
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Yapılandırmayı günceller
        
        Args:
            updates: Güncellenecek değerler
        """
        self.config.update(updates)
        self.logger.info(f"Yapılandırma güncellendi: {updates}")
        
    def reset_config(self) -> None:
        """Yapılandırmayı sıfırlar"""
        self.config = {}
        self.logger.info("Yapılandırma sıfırlandı")
        
    def validate_file(self, file_path: Path) -> None:
        """Dosya yolunu doğrular
        
        Args:
            file_path: Dosya yolu
            
        Raises:
            FileNotFoundError: Dosya bulunamazsa
            ValueError: Dosya geçersizse
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            
        if not file_path.is_file():
            raise ValueError(f"Geçersiz dosya: {file_path}")
            
    def validate_dir(self, dir_path: Path) -> None:
        """Klasör yolunu doğrular
        
        Args:
            dir_path: Klasör yolu
            
        Raises:
            NotADirectoryError: Klasör bulunamazsa
        """
        if not dir_path.exists():
            raise NotADirectoryError(f"Klasör bulunamadı: {dir_path}")
            
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Geçersiz klasör: {dir_path}")
            
    def ensure_dir(self, dir_path: Path) -> None:
        """Klasörün var olduğundan emin olur
        
        Args:
            dir_path: Klasör yolu
        """
        dir_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Klasör oluşturuldu: {dir_path}")
        
    def cleanup(self) -> None:
        """Geçici dosyaları ve kaynakları temizler"""
        pass  # Alt sınıflar bu metodu override edebilir 