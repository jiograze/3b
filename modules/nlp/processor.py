from typing import Dict, Any, Optional, List
import torch

from .text_processor import TextProcessor
from .cultural_processor import CulturalTextProcessor

class NLPProcessor:
    def __init__(
        self,
        model_name: str = "dbmdz/bert-base-turkish-cased",
        max_length: int = 77,
        cultural_tokens_path: Optional[str] = None
    ):
        """
        NLP işleme modülü.
        Args:
            model_name: Kullanılacak dil modeli
            max_length: Maksimum token uzunluğu
            cultural_tokens_path: Kültürel token dosyasının yolu
        """
        # Alt işleyicileri oluştur
        self.cultural_processor = CulturalTextProcessor(
            model_name=model_name,
            cultural_tokens_path=cultural_tokens_path
        )
        
        self.text_processor = TextProcessor(
            model_name=model_name,
            max_length=max_length,
            cultural_processor=self.cultural_processor
        )
        
    def process_text(
        self,
        text: str,
        return_cultural: bool = True
    ) -> Dict[str, Any]:
        """
        Metni işle
        Args:
            text: İşlenecek metin
            return_cultural: Kültürel özellikleri döndür
        Returns:
            Dict[str, Any]: İşlenmiş metin ve özellikler
        """
        return self.text_processor.encode(text, return_cultural=return_cultural)
        
    def batch_process(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_cultural: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Metin listesini işle
        Args:
            texts: İşlenecek metin listesi
            batch_size: Batch boyutu
            return_cultural: Kültürel özellikleri döndür
        Returns:
            List[Dict[str, Any]]: İşlenmiş metin listesi
        """
        return self.text_processor.batch_process(
            texts,
            batch_size=batch_size
        )
        
    def extract_cultural_features(
        self,
        text: str
    ) -> Dict[str, List[str]]:
        """
        Metinden kültürel özellikleri çıkar
        Args:
            text: Metin
        Returns:
            Dict[str, List[str]]: Kültürel özellikler
        """
        return self.cultural_processor.extract_cultural_features(text)
        
    def encode_cultural_features(
        self,
        features: Dict[str, List[str]]
    ) -> torch.Tensor:
        """
        Kültürel özellikleri encode et
        Args:
            features: Kültürel özellikler
        Returns:
            torch.Tensor: Encoded özellikler
        """
        return self.cultural_processor.encode_cultural_features(features)
