import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

class TextToPointCloud(nn.Module):
    """Metin tabanlı nokta bulutu üretici sınıfı"""
    
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
        
        # Text encoder
        self.text_encoder = self._build_text_encoder()
        
        # Point cloud generator
        self.point_generator = self._build_point_generator()
        
        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            config['model']['text_encoder']
        )
        
    def _build_text_encoder(self) -> nn.Module:
        """Text encoder oluştur"""
        # T5 modelini yükle
        text_encoder = T5EncoderModel.from_pretrained(
            self.config['model']['text_encoder']
        )
        
        # Parametreleri dondur
        for param in text_encoder.parameters():
            param.requires_grad = False
            
        return text_encoder
        
    def _build_point_generator(self) -> nn.Module:
        """Point cloud generator oluştur"""
        from .point_cloud import PointCloudGenerator
        return PointCloudGenerator(self.config)
        
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
        # Text encoder
        text_features = self.text_encoder(
            input_ids=batch['text'],
            attention_mask=batch['attention_mask']
        ).last_hidden_state
        
        # Global text özelliği
        text_features = torch.mean(text_features, dim=1)
        
        # Point cloud generator
        outputs = self.point_generator(text_features)
        
        return outputs
        
    def generate(
        self,
        text: str,
        num_points: Optional[int] = None,
        temperature: Optional[float] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Metinden nokta bulutu üret
        
        Args:
            text: Metin
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
            
        # Metni tokenize et
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Veriyi GPU'ya taşı
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # Text encoder
        with torch.no_grad():
            text_features = self.text_encoder(
                input_ids=tokens['input_ids'],
                attention_mask=tokens['attention_mask']
            ).last_hidden_state
            
            # Global text özelliği
            text_features = torch.mean(text_features, dim=1)
            
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