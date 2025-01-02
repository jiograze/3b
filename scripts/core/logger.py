import os
import logging
import logging.handlers
import yaml
from pathlib import Path

class OtukenLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OtukenLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """Yapılandırma dosyasını yükle"""
        config_path = Path('scripts/config.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)['logging']
        except Exception as e:
            # Varsayılan yapılandırma
            return {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'handlers': [
                    {
                        'type': 'file',
                        'filename': 'logs/otuken3d.log',
                        'max_bytes': 10485760,
                        'backup_count': 5
                    },
                    {
                        'type': 'console',
                        'level': 'INFO'
                    }
                ]
            }
    
    def _setup_logging(self):
        """Logging sistemini yapılandır"""
        # Log dizinini oluştur
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Ana logger'ı yapılandır
        logger = logging.getLogger('otuken3d')
        logger.setLevel(getattr(logging, self.config['level']))
        
        # Formatter oluştur
        formatter = logging.Formatter(self.config['format'])
        
        # Handler'ları ekle
        for handler_config in self.config['handlers']:
            if handler_config['type'] == 'file':
                handler = logging.handlers.RotatingFileHandler(
                    handler_config['filename'],
                    maxBytes=handler_config['max_bytes'],
                    backupCount=handler_config['backup_count']
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                
            elif handler_config['type'] == 'console':
                handler = logging.StreamHandler()
                handler.setLevel(getattr(logging, handler_config['level']))
                handler.setFormatter(formatter)
                logger.addHandler(handler)
    
    @staticmethod
    def get_logger(name=None):
        """Logger örneği al"""
        if name:
            return logging.getLogger(f'otuken3d.{name}')
        return logging.getLogger('otuken3d')

# Singleton örneği oluştur
logger = OtukenLogger()

def get_logger(name=None):
    """Logger örneği almak için yardımcı fonksiyon"""
    return logger.get_logger(name) 