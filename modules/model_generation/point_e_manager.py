import os
import torch
import logging
from pathlib import Path
from ..point_e.models.download import load_checkpoint
from ..point_e.util.plotting import plot_point_cloud
from ..point_e.models.configs import MODEL_CONFIGS

class PointEManager:
    def __init__(self, cache_dir='models/pretrained'):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        
    def load_model(self, model_name='base40M-textvec'):
        """Point-E modelini yükle"""
        try:
            self.logger.info(f"Point-E {model_name} modeli yükleniyor...")
            
            # Model konfigürasyonunu al
            self.config = MODEL_CONFIGS[model_name]
            
            # Modeli yükle
            self.model = load_checkpoint(model_name, self.device, cache_dir=self.cache_dir)
            self.model.eval()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Point-E model yükleme hatası: {str(e)}")
            return False
            
    def generate_point_cloud(self, text_prompt, num_points=1024, output_path=None):
        """Metin açıklamasından 3D nokta bulutu oluştur"""
        try:
            if self.model is None:
                self.load_model()
                
            # Text-to-3D dönüşümü yap
            with torch.no_grad():
                point_cloud = self.model.generate(
                    text_prompt,
                    num_points=num_points,
                    device=self.device
                )
            
            if output_path:
                # Nokta bulutunu görselleştir ve kaydet
                fig = plot_point_cloud(point_cloud, output_path)
                
                # Ayrıca ham veriyi de kaydet
                torch.save(point_cloud, output_path.replace('.png', '.pt'))
                
            return point_cloud
            
        except Exception as e:
            self.logger.error(f"Nokta bulutu oluşturma hatası: {str(e)}")
            return None
            
    def validate_model(self):
        """Model dosyalarının varlığını ve bütünlüğünü kontrol et"""
        try:
            # Test amaçlı basit bir nokta bulutu oluştur
            test_cloud = self.generate_point_cloud("a simple cube")
            return test_cloud is not None
        except Exception:
            return False 