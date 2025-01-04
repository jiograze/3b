import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from torchvision.models import resnet50
import numpy as np
from PIL import Image
import torchvision.transforms as T

class ImageToPointCloud(nn.Module):
    """Görüntü tabanlı nokta bulutu üretici sınıfı"""
    
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
        
        # Image encoder
        self.image_encoder = self._build_image_encoder()
        
        # Point cloud generator
        self.point_generator = self._build_point_generator()
        
        # Görüntü dönüşümleri
        self.transforms = self._setup_transforms()
        
    def _build_image_encoder(self) -> nn.Module:
        """Image encoder oluştur"""
        # ResNet50 modelini yükle
        image_encoder = resnet50(pretrained=True)
        
        # Son katmanı kaldır
        image_encoder = nn.Sequential(*list(image_encoder.children())[:-1])
        
        # Parametreleri dondur
        for param in image_encoder.parameters():
            param.requires_grad = False
            
        return image_encoder
        
    def _build_point_generator(self) -> nn.Module:
        """Point cloud generator oluştur"""
        from .point_cloud import PointCloudGenerator
        return PointCloudGenerator(self.config)
        
    def _setup_transforms(self) -> T.Compose:
        """Görüntü dönüşümlerini ayarla"""
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """İleri geçiş
        
        Args:
            batch: Veri batch'i
            
        Returns:
            Çıktılar
        """
        # Image encoder
        image_features = self.image_encoder(batch['image'])
        image_features = image_features.squeeze()
        
        # Point cloud generator
        outputs = self.point_generator(image_features)
        
        return outputs
        
    def generate(
        self,
        image: Union[str, Image.Image, np.ndarray],
        num_points: Optional[int] = None,
        temperature: Optional[float] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Görüntüden nokta bulutu üret
        
        Args:
            image: Görüntü
            num_points: Nokta sayısı
            temperature: Sıcaklık parametresi
            device: Cihaz
            
        Returns:
            Nokta bulutu (N x 3)
        """
        # Parametreleri ayarla
        if num_points is None:
            num_points = self.config['generation']['num_points']
        if temperature is None:
            temperature = self.config['generation']['temperature']
            
        # Görüntüyü yükle
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Dönüşümleri uygula
        image = self.transforms(image)
        image = image.unsqueeze(0)  # Batch boyutu ekle
        image = image.to(device)
        
        # Image encoder
        with torch.no_grad():
            image_features = self.image_encoder(image)
            image_features = image_features.squeeze()
            
            # Point cloud generator
            points = self.point_generator.generate(
                num_points,
                device=device
            )
            
            # Sıcaklık uygula
            if temperature != 1.0:
                points = points / temperature
                
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