"""
Loglama Modülü
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Logger nesnesi oluşturur ve yapılandırır
    
    Args:
        name: Logger adı
        
    Returns:
        Logger nesnesi
    """
    logger = logging.getLogger(name)
    
    # Logger zaten yapılandırılmışsa, mevcut logger'ı döndür
    if logger.handlers:
        return logger
    
    # Log seviyesini ayarla
    logger.setLevel(logging.INFO)
    
    # Formatlayıcı oluştur
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Konsol handler'ı ekle
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Logs dizinini oluştur
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Dosya handler'ı ekle
    file_handler = logging.FileHandler(
        log_dir / f"{name if name else 'app'}.log"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def setup_logging(config: Dict[str, Any]) -> None:
    """Loglama sistemini yapılandırır
    
    Args:
        config: Loglama yapılandırması
    """
    # Varsayılan değerler
    log_level = config.get('level', 'INFO').upper()
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('file')
    
    # Kök logger'ı yapılandır
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # Formatlayıcı oluştur
    formatter = logging.Formatter(log_format)
    
    # Konsol handler'ı ekle
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Dosya handler'ı ekle
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Logger nesnesi döndürür
    
    Args:
        name: Logger adı
        
    Returns:
        Logger nesnesi
    """
    return logging.getLogger(name) 