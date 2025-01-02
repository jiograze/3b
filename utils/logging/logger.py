import logging
import sys
from pathlib import Path

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        """Logger'ı yapılandır"""
        # Log dizinini oluştur
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Ana logger'ı yapılandır
        self.logger = logging.getLogger('otuken3d')
        self.logger.setLevel(logging.INFO)
        
        # Dosya handler'ı
        file_handler = logging.FileHandler('logs/otuken3d.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Konsol handler'ı
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatı ayarla
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Handler'ları ekle
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    @classmethod
    def get_logger(cls, name=None):
        """Logger instance'ı döndür"""
        instance = cls()
        if name:
            return logging.getLogger(f'otuken3d.{name}')
        return instance.logger

def get_logger(name=None):
    """Kolay erişim için yardımcı fonksiyon"""
    return Logger.get_logger(name) 