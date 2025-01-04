import torch
import json
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel

class CulturalTextProcessor:
    def __init__(
        self,
        model_name: str = "dbmdz/bert-base-turkish-cased",
        cultural_tokens_path: Optional[str] = None
    ):
        """
        Kültürel metin işleme sınıfı.
        Args:
            model_name: Kullanılacak dil modeli
            cultural_tokens_path: Kültürel token dosyasının yolu
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Kültürel token dosyasını yükle
        if cultural_tokens_path is None:
            cultural_tokens_path = "models/otuken3d/configs/cultural_tokens.json"
        self.cultural_tokens = self.load_cultural_tokens(cultural_tokens_path)
        
    @staticmethod
    def load_cultural_tokens(path: str) -> Dict:
        """Kültürel token dosyasını yükle"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def extract_cultural_features(self, text: str) -> Dict[str, List[str]]:
        """
        Metinden kültürel özellikleri çıkar
        Args:
            text: Girdi metni
        Returns:
            Dict[str, List[str]]: Kültürel özellikler
        """
        features = {
            'motifs': [],
            'patterns': [],
            'styles': [],
            'materials': []
        }
        
        # Motifleri bul
        for motif in self.cultural_tokens['motifs']:
            if motif.lower() in text.lower():
                features['motifs'].append(motif)
                
        # Desenleri bul
        for pattern in self.cultural_tokens['patterns']:
            if pattern.lower() in text.lower():
                features['patterns'].append(pattern)
                
        # Stilleri bul
        for style in self.cultural_tokens['styles']:
            if style.lower() in text.lower():
                features['styles'].append(style)
                
        # Malzemeleri bul
        for material in self.cultural_tokens['materials']:
            if material.lower() in text.lower():
                features['materials'].append(material)
                
        return features
        
    def encode_cultural_features(
        self,
        features: Dict[str, List[str]]
    ) -> torch.Tensor:
        """
        Kültürel özellikleri encode et
        Args:
            features: Kültürel özellikler sözlüğü
        Returns:
            torch.Tensor: Encoded özellikler
        """
        # Özellikleri birleştir
        text = ""
        for category, items in features.items():
            if items:
                text += f"{category}: {', '.join(items)}. "
                
        # Tokenize ve encode et
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.last_hidden_state
        
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Metni işle ve kültürel özellikleri çıkar
        Args:
            text: Girdi metni
        Returns:
            Dict[str, torch.Tensor]: İşlenmiş metin ve özellikler
        """
        # Kültürel özellikleri çıkar
        cultural_features = self.extract_cultural_features(text)
        
        # Metni encode et
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        
        with torch.no_grad():
            text_outputs = self.model(**text_inputs)
            
        # Kültürel özellikleri encode et
        cultural_encoding = self.encode_cultural_features(cultural_features)
        
        return {
            'text_encoding': text_outputs.last_hidden_state,
            'cultural_encoding': cultural_encoding,
            'cultural_features': cultural_features
        } 