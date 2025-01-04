"""
Loglama Modülü
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Loglama sistemini yapılandırır"""
    if config is None:
        config = {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/otuken3d.log"
        }
        
    # Log seviyesini ayarla
    log_level = getattr(logging, config.get("level", "INFO").upper())
    
    # Formatlayıcıyı oluştur
    formatter = logging.Formatter(config.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    
    # Ana logger'ı yapılandır
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Mevcut handler'ları temizle
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Konsol handler'ı ekle
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Dosya handler'ı ekle
    if "file" in config:
        log_file = Path(config["file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
def get_logger(name: str) -> logging.Logger:
    """İsimlendirilmiş logger döndürür"""
    return logging.getLogger(name)

class LoggerMixin:
    """Sınıflara loglama yeteneği ekleyen mixin"""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger 