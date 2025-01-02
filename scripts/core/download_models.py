import logging
import os
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline
import torch

logging.basicConfig(
    filename='logs/setup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ModelDownloader:
    def __init__(self, cache_dir='models/pretrained'):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
    def download_point_e(self):
        """Point-E modelini indir"""
        try:
            self.logger.info("Point-E modeli indiriliyor...")
            pipeline = DiffusionPipeline.from_pretrained(
                'openai/point-e-base',
                cache_dir=self.cache_dir
            )
            self.logger.info("Point-E modeli başarıyla indirildi")
            return True
        except Exception as e:
            self.logger.error(f"Point-E indirme hatası: {str(e)}")
            return False
            
    def download_clip(self):
        """CLIP modelini indir"""
        try:
            self.logger.info("CLIP modeli indiriliyor...")
            CLIPTextModel.from_pretrained(
                'openai/clip-vit-base-patch32',
                cache_dir=self.cache_dir
            )
            CLIPTokenizer.from_pretrained(
                'openai/clip-vit-base-patch32',
                cache_dir=self.cache_dir
            )
            self.logger.info("CLIP modeli başarıyla indirildi")
            return True
        except Exception as e:
            self.logger.error(f"CLIP indirme hatası: {str(e)}")
            return False
            
    def verify_models(self):
        """İndirilen modellerin doğruluğunu kontrol et"""
        models = {
            'point-e': 'openai/point-e-base',
            'clip': 'openai/clip-vit-base-patch32'
        }
        
        for model_name, model_path in models.items():
            path = Path(self.cache_dir) / model_path
            if not path.exists():
                self.logger.warning(f"{model_name} model dosyaları eksik")
                return False
        return True

def main():
    downloader = ModelDownloader()
    
    # Modelleri indir
    models_success = all([
        downloader.download_point_e(),
        downloader.download_clip()
    ])
    
    # Doğrulama yap
    if models_success and downloader.verify_models():
        logging.info("Tüm modeller başarıyla indirildi ve doğrulandı")
        return 0
    else:
        logging.error("Model indirme veya doğrulama hatası")
        return 1

if __name__ == "__main__":
    exit(main()) 