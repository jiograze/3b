import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import numpy as np

class PointCloudGenerator(nn.Module):
    """Nokta bulutu üretici sınıfı"""
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Args:
            config: Konfigürasyon
        """
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = self._build_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> nn.Module:
        """Encoder oluştur"""
        hidden_size = self.config['model']['hidden_size']
        
        return nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def _build_decoder(self) -> nn.Module:
        """Decoder oluştur"""
        hidden_size = self.config['model']['hidden_size']
        
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
        
    def forward(
        self,
        points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """İleri geçiş
        
        Args:
            points: Nokta bulutu (B x N x 3)
            
        Returns:
            Çıktılar
        """
        # Encoder
        features = self.encoder(points)
        
        # Decoder
        reconstructed = self.decoder(features)
        
        return {
            'reconstructed': reconstructed,
            'features': features
        }
        
    def generate(
        self,
        num_points: int,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Nokta bulutu üret
        
        Args:
            num_points: Nokta sayısı
            device: Cihaz
            
        Returns:
            Nokta bulutu (N x 3)
        """
        # Rastgele gürültü
        z = torch.randn(num_points, self.config['model']['hidden_size'])
        z = z.to(device)
        
        # Decoder
        with torch.no_grad():
            points = self.decoder(z)
            
        return points
        
    def save_points(
        self,
        points: Union[np.ndarray, torch.Tensor],
        path: str
    ):
        """Nokta bulutunu kaydet
        
        Args:
            points: Nokta bulutu
            path: Kayıt yolu
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
            
        np.save(path, points) 