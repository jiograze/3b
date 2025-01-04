import torch
import os
from typing import Dict, Any, Optional
from .model_generator import TextToModelGenerator
from utils.logging import setup_logger

logger = setup_logger(__name__)

class ModelManager:
    """Model yönetim sınıfı"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Args:
            config: Konfigürasyon
            device: Cihaz
        """
        self.config = config
        self.device = device
        
        # Model üreticiyi yükle
        self.generator = TextToModelGenerator(config, device)
        
    def save_model(
        self,
        model: torch.nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None
    ):
        """Modeli kaydet
        
        Args:
            model: Model
            path: Kayıt yolu
            optimizer: Optimizer
            epoch: Epoch numarası
        """
        # Kayıt dizinini oluştur
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Checkpoint oluştur
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': self.config
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        # Kaydet
        torch.save(checkpoint, path)
        logger.info(f"Model kaydedildi: {path}")
        
    def load_model(
        self,
        model: torch.nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Modeli yükle
        
        Args:
            model: Model
            path: Yükleme yolu
            optimizer: Optimizer
            
        Returns:
            Checkpoint bilgileri
        """
        # Checkpoint'i yükle
        checkpoint = torch.load(path, map_location=self.device)
        
        # Model durumunu yükle
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optimizer durumunu yükle
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        logger.info(f"Model yüklendi: {path}")
        
        return checkpoint
        
    def generate_model(
        self,
        text: str,
        num_points: Optional[int] = None,
        temperature: Optional[float] = None,
        save_path: Optional[str] = None
    ):
        """Metinden model üret
        
        Args:
            text: Metin
            num_points: Nokta sayısı
            temperature: Sıcaklık parametresi
            save_path: Kayıt yolu
        """
        # Model üret
        points = self.generator.generate(
            text,
            num_points=num_points,
            temperature=temperature
        )
        
        # Kaydet
        if save_path is not None:
            self.generator.save_points(points, save_path)
            logger.info(f"Nokta bulutu kaydedildi: {save_path}")
            
        return points 