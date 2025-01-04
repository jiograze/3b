import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from utils.helpers.exceptions import ModelError

class BaseModel(nn.Module):
    """Temel model sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model konfigürasyonu
        """
        super().__init__()
        self.config = config
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """İleri geçiş
        
        Args:
            batch: Girdi verisi
            
        Returns:
            Çıktı sözlüğü
        """
        raise NotImplementedError("Alt sınıflar forward metodunu uygulamalı")
        
    def save(self, path: str):
        """Modeli kaydet
        
        Args:
            path: Kayıt yolu
        """
        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            raise ModelError(f"Model kaydedilemedi: {str(e)}")
            
    def load(self, path: str):
        """Modeli yükle
        
        Args:
            path: Model yolu
        """
        try:
            self.load_state_dict(torch.load(path))
        except Exception as e:
            raise ModelError(f"Model yüklenemedi: {str(e)}")
            
    def get_num_parameters(self) -> int:
        """Toplam parametre sayısını döndür"""
        return sum(p.numel() for p in self.parameters())
        
    def get_parameter_groups(self) -> Dict[str, Any]:
        """Parametre gruplarını döndür"""
        return {
            'all': self.parameters(),
            'trainable': filter(lambda p: p.requires_grad, self.parameters()),
            'frozen': filter(lambda p: not p.requires_grad, self.parameters())
        }
        
    def freeze(self):
        """Tüm parametreleri dondur"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Tüm parametreleri çöz"""
        for param in self.parameters():
            param.requires_grad = True
            
    def to_device(self, device: str):
        """Modeli belirtilen cihaza taşı
        
        Args:
            device: Hedef cihaz
        """
        self.to(device)
        
    def get_config(self) -> Dict[str, Any]:
        """Model konfigürasyonunu döndür"""
        return self.config 