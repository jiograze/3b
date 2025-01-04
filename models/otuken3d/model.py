import torch
import torch.nn as nn
from typing import Dict, Any
from modules.core.base_model import BaseModel
from transformers import BertModel, T5EncoderModel
from torchvision.models import resnet50

class Otuken3DModel(BaseModel):
    """Otuken3D model sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Model konfigürasyonu
        """
        super().__init__(config)
        
        # Konfigürasyonu al
        model_config = config['model']
        
        # Text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(model_config['text_encoder'])
        self.text_proj = nn.Linear(
            self.text_encoder.config.hidden_size,
            model_config['latent_dim']
        )
        
        # Image encoder
        self.image_encoder = resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(
            self.image_encoder.fc.in_features,
            model_config['latent_dim']
        )
        
        # Transformer encoder
        transformer_config = model_config['transformer']
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config['latent_dim'],
            nhead=transformer_config['num_heads'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout=transformer_config['dropout']
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_config['num_layers']
        )
        
        # Point decoder
        decoder_config = model_config['point_decoder']
        self.point_decoder = nn.Sequential(
            nn.Linear(model_config['latent_dim'], decoder_config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(decoder_config['hidden_dim'], decoder_config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(decoder_config['hidden_dim'], decoder_config['num_points'] * 3)
        )
        
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Metin kodlama
        
        Args:
            text: Metin tensörü
            
        Returns:
            Kodlanmış metin özellikleri
        """
        text_features = self.text_encoder(text).last_hidden_state
        text_features = self.text_proj(text_features)
        return text_features
        
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Görüntü kodlama
        
        Args:
            image: Görüntü tensörü
            
        Returns:
            Kodlanmış görüntü özellikleri
        """
        return self.image_encoder(image)
        
    def decode_points(self, features: torch.Tensor) -> torch.Tensor:
        """Nokta bulutu çözümleme
        
        Args:
            features: Özellik tensörü
            
        Returns:
            Nokta bulutu
        """
        batch_size = features.shape[0]
        points = self.point_decoder(features)
        return points.view(batch_size, -1, 3)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """İleri geçiş
        
        Args:
            batch: Girdi verisi
            
        Returns:
            Çıktı sözlüğü
        """
        outputs = {}
        
        # Metin kodlama
        if 'text' in batch:
            text_features = self.encode_text(batch['text'])
            outputs['text_features'] = text_features
            
        # Görüntü kodlama
        if 'image' in batch:
            image_features = self.encode_image(batch['image'])
            outputs['image_features'] = image_features
            
        # Özellik birleştirme ve transformer
        if 'text_features' in outputs and 'image_features' in outputs:
            features = torch.cat([
                outputs['text_features'],
                outputs['image_features'].unsqueeze(1)
            ], dim=1)
            features = self.transformer(features)
            outputs['combined_features'] = features
            
        # Nokta bulutu çözümleme
        if 'combined_features' in outputs:
            points = self.decode_points(outputs['combined_features'][:, 0])
            outputs['point_cloud'] = points
            
        return outputs 