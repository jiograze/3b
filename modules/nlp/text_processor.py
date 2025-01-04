import re
import torch
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModel
from .cultural_processor import CulturalTextProcessor

class TextProcessor:
    def __init__(
        self,
        model_name: str = "dbmdz/bert-base-turkish-cased",
        max_length: int = 77,
        cultural_processor: Optional[CulturalTextProcessor] = None
    ):
        """
        Gelişmiş metin işleme sınıfı.
        Args:
            model_name: Kullanılacak dil modeli
            max_length: Maksimum token uzunluğu
            cultural_processor: Kültürel metin işleyici
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.cultural_processor = cultural_processor or CulturalTextProcessor()
        
    def clean_text(self, text: str) -> str:
        """
        Metni temizler ve normalize eder
        Args:
            text: Ham metin
        Returns:
            str: Temizlenmiş metin
        """
        # Özel karakterleri ve fazladan boşlukları temizle
        text = re.sub(r'[^a-zA-ZğüşöçİĞÜŞÖÇ\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Noktalama işaretlerini düzenle
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        text = text.strip()
        
        return text
        
    def normalize_text(self, text: str) -> str:
        """
        Metni normalize eder
        Args:
            text: Temizlenmiş metin
        Returns:
            str: Normalize edilmiş metin
        """
        # Türkçe karakterleri düzenle
        replacements = {
            'İ': 'i', 'I': 'ı',
            'Ğ': 'ğ', 'Ü': 'ü',
            'Ş': 'ş', 'Ö': 'ö',
            'Ç': 'ç'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text.lower()
        
    def tokenize(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Metni tokenize eder
        Args:
            text: Metin veya metin listesi
            add_special_tokens: Özel tokenleri ekle
        Returns:
            Dict[str, torch.Tensor]: Tokenize edilmiş metin
        """
        # Metin temizleme ve normalizasyon
        if isinstance(text, str):
            text = self.normalize_text(self.clean_text(text))
        else:
            text = [self.normalize_text(self.clean_text(t)) for t in text]
            
        return self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens
        )
        
    def encode(
        self,
        text: Union[str, List[str]],
        return_cultural: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Metni encode eder
        Args:
            text: Metin veya metin listesi
            return_cultural: Kültürel özellikleri de döndür
        Returns:
            Dict[str, torch.Tensor]: Encoded metin ve özellikler
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Model çıktısı
        with torch.no_grad():
            outputs = self.model(**tokens)
            
        result = {
            'text_encoding': outputs.last_hidden_state,
            'pooled_encoding': outputs.last_hidden_state.mean(dim=1)
        }
        
        # Kültürel özellikler
        if return_cultural:
            if isinstance(text, str):
                cultural_outputs = self.cultural_processor.process_text(text)
            else:
                cultural_outputs = [
                    self.cultural_processor.process_text(t)
                    for t in text
                ]
                # Batch işleme için tensörleri birleştir
                cultural_outputs = {
                    k: torch.cat([d[k] for d in cultural_outputs])
                    if isinstance(d[k], torch.Tensor) else [d[k] for d in cultural_outputs]
                    for k in cultural_outputs[0].keys()
                }
                
            result.update(cultural_outputs)
            
        return result
        
    def batch_process(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Metin listesini batch halinde işle
        Args:
            texts: Metin listesi
            batch_size: Batch boyutu
        Returns:
            List[Dict[str, torch.Tensor]]: İşlenmiş metin listesi
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.encode(batch)
            results.extend([
                {k: v[j] if isinstance(v, torch.Tensor) else v[j]
                 for k, v in batch_results.items()}
                for j in range(len(batch))
            ])
        return results