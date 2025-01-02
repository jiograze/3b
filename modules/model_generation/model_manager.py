import os
from typing import Optional, Dict
import torch
from .model_generator import ModelGenerator

class ModelManager:
    def __init__(self, config: Dict = None):
        """
        Model yöneticisi
        
        Args:
            config (Dict): Konfigürasyon ayarları
        """
        self.config = config or {}
        self.generator = ModelGenerator(config)
        self.models_dir = self.config.get("models_dir", "models/generated")
        
        # Çıktı dizinini oluştur
        os.makedirs(self.models_dir, exist_ok=True)
    
    def generate_model(self, 
                      text_prompt: Optional[str] = None,
                      image: Optional[torch.Tensor] = None) -> Optional[str]:
        """
        Metin veya görüntüden 3D model oluşturur ve kaydeder
        
        Args:
            text_prompt (Optional[str]): Metin prompt'u
            image (Optional[torch.Tensor]): Görüntü tensoru
            
        Returns:
            Optional[str]: Kaydedilen model dosyasının yolu
        """
        try:
            if text_prompt:
                mesh = self.generator.generate_from_text(text_prompt)
                filename = f"text_{hash(text_prompt)}.obj"
            elif image is not None:
                mesh = self.generator.generate_from_image(image)
                filename = f"image_{hash(str(image))}.obj"
            else:
                raise ValueError("Text prompt veya görüntü gerekli")
            
            if mesh is None:
                return None
                
            filepath = os.path.join(self.models_dir, filename)
            self.generator.save_model(mesh, filepath)
            
            return filepath
            
        except Exception as e:
            print(f"Model oluşturma ve kaydetme hatası: {str(e)}")
            return None 