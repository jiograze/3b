"""
Ana Uygulama Modülü
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.endpoints import app as api_app
from src.core.config import load_config
from src.core.logger import setup_logger
from modules.training.model_factory import ModelFactory
from modules.training.config_manager import ConfigManager

# Logger'ı yapılandır
logger = setup_logger(__name__)

# Yapılandırmayı yükle
config = load_config()

# Config manager ve model factory oluştur
config_manager = ConfigManager()
model_factory = ModelFactory(config_manager)

# Modelleri yükle
try:
    models = model_factory.create_all_models(pretrained=True)
    logger.info("Tüm modeller başarıyla yüklendi")
except Exception as e:
    logger.error(f"Model yükleme hatası: {str(e)}")
    models = {}

# API uygulamasını al
app = api_app

# Modelleri app state'e ekle
app.state.models = models
app.state.model_factory = model_factory
app.state.config_manager = config_manager

if __name__ == "__main__":
    # Sunucuyu başlat
    uvicorn.run(
        "src.main:app",
        host=config['server']['host'],
        port=config['server']['port'],
        workers=config['server']['workers'],
        reload=True
    ) 