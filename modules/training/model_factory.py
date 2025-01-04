from typing import Dict, Any, Optional
import torch
from pathlib import Path

from models.text2shape.text2shape_model import Text2ShapeModel
from models.image2shape.image2shape_model import Image2ShapeModel
from modules.training.otuken3d_model import Otuken3DModel
from .config_manager import ConfigManager

class ModelFactory:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Model factory sınıfı.
        Args:
            config_manager: Konfigürasyon yöneticisi
        """
        self.config_manager = config_manager or ConfigManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_model(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Model oluştur.
        Args:
            model_type: Model tipi ("text2shape", "image2shape", "otuken3d")
            config: Model konfigürasyonu (None ise config manager'dan alınır)
            pretrained: Önceden eğitilmiş model kullanılsın mı
            checkpoint_path: Checkpoint dosyasının yolu
        """
        # Konfigürasyonu al
        if config is None:
            config = self.config_manager.get_model_config(model_type)
            
        # Model oluştur
        if model_type == "text2shape":
            model = Text2ShapeModel(config=config, device=self.device)
        elif model_type == "image2shape":
            model = Image2ShapeModel(config=config, device=self.device)
        elif model_type == "otuken3d":
            model = Otuken3DModel(config=config, device=self.device)
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
            
        # Pretrained model yükle
        if pretrained:
            if checkpoint_path is None:
                checkpoint_path = self._get_default_checkpoint_path(model_type)
            model.load_model(checkpoint_path)
            
        return model.to(self.device)
        
    @staticmethod
    def _get_default_checkpoint_path(model_type: str) -> str:
        """Varsayılan checkpoint yolunu al"""
        checkpoint_paths = {
            "text2shape": "models/text2shape/text2shape_model.pt",
            "image2shape": "models/image2shape/image2shape_model.pt",
            "otuken3d": "models/otuken3d/text2shape_model.pt"
        }
        return checkpoint_paths[model_type]
        
    def create_all_models(
        self,
        pretrained: bool = False
    ) -> Dict[str, torch.nn.Module]:
        """Tüm modelleri oluştur"""
        return {
            model_type: self.create_model(model_type, pretrained=pretrained)
            for model_type in ["text2shape", "image2shape", "otuken3d"]
        } 