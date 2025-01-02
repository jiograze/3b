from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from utils.logging.logger import get_logger
from utils.helpers.exceptions import ModelError

logger = get_logger('model')

class BaseModel(nn.Module, ABC):
    """Temel model sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._initialize_model()
        
    @abstractmethod
    def _initialize_model(self) -> None:
        """Model mimarisini başlat"""
        pass
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """İleri geçiş"""
        pass
        
    def save_checkpoint(self, path: str, extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Model checkpoint'ini kaydet"""
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'config': self.config
            }
            
            if extra_data:
                checkpoint.update(extra_data)
                
            torch.save(checkpoint, path)
            logger.info(f"Model kaydedildi: {path}")
            
        except Exception as e:
            raise ModelError(f"Model kaydedilemedi: {str(e)}")
            
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Model checkpoint'ini yükle"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model yüklendi: {path}")
            return checkpoint
            
        except Exception as e:
            raise ModelError(f"Model yüklenemedi: {str(e)}")
            
    @property
    def device(self) -> torch.device:
        """Model cihazını getir"""
        return next(self.parameters()).device
        
    def to_device(self, device: Optional[torch.device] = None) -> 'BaseModel':
        """Modeli belirtilen cihaza taşı"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        logger.debug(f"Model taşındı: {device}")
        return self
        
    def get_parameter_count(self) -> Dict[str, int]:
        """Model parametre sayılarını getir"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
        
    def print_model_summary(self) -> None:
        """Model özetini yazdır"""
        param_counts = self.get_parameter_count()
        logger.info("Model Özeti:")
        logger.info(f"Toplam parametre: {param_counts['total']:,}")
        logger.info(f"Eğitilebilir parametre: {param_counts['trainable']:,}")
        logger.info(f"Dondurulmuş parametre: {param_counts['frozen']:,}")
        logger.info(f"Cihaz: {self.device}")
        
    def freeze(self) -> None:
        """Tüm model parametrelerini dondur"""
        for param in self.parameters():
            param.requires_grad = False
        logger.debug("Model parametreleri donduruldu")
        
    def unfreeze(self) -> None:
        """Tüm model parametrelerinin dondurmasını kaldır"""
        for param in self.parameters():
            param.requires_grad = True
        logger.debug("Model parametrelerinin dondurması kaldırıldı") 