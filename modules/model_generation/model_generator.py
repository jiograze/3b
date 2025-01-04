import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

class TextToModelGenerator:
    """Metin tabanlı 3D model üretici sınıfı"""
    
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
        
        # Modeli yükle
        self.model = self._load_model()
        self.model = self.model.to(device)
        
        # Tokenizer'ı yükle
        self.tokenizer = T5Tokenizer.from_pretrained(
            config['model']['text_encoder']
        )
        
    def _load_model(self) -> nn.Module:
        """Modeli yükle"""
        from models.otuken3d.model import Otuken3DModel
        return Otuken3DModel(self.config)
        
    def generate(
        self,
        text: str,
        num_points: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> np.ndarray:
        """Metinden 3D model üret
        
        Args:
            text: Metin
            num_points: Nokta sayısı
            temperature: Sıcaklık parametresi
            
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
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Modeli değerlendirme moduna al
        self.model.eval()
        
        with torch.no_grad():
            # İleri geçiş
            outputs = self.model({
                'text': tokens['input_ids'],
                'attention_mask': tokens['attention_mask']
            })
            
            # Nokta bulutunu al
            points = outputs['point_cloud'][0]  # İlk örneği al
            
            # Sıcaklık uygula
            if temperature != 1.0:
                points = points / temperature
                
            # CPU'ya taşı ve NumPy'a çevir
            points = points.cpu().numpy()
            
        return points
        
    def generate_batch(
        self,
        texts: List[str],
        num_points: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[np.ndarray]:
        """Metinlerden 3D modeller üret
        
        Args:
            texts: Metinler
            num_points: Nokta sayısı
            temperature: Sıcaklık parametresi
            
        Returns:
            Nokta bulutları
        """
        return [
            self.generate(text, num_points, temperature)
            for text in texts
        ]
        
    def save_points(
        self,
        points: np.ndarray,
        path: str
    ):
        """Nokta bulutunu kaydet
        
        Args:
            points: Nokta bulutu
            path: Kayıt yolu
        """
        np.save(path, points)