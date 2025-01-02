import re
import torch
import transformers
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

class TextProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        
    def clean_text(self, text: str) -> str:
        """
        Metni temizler ve normalize eder
        
        Args:
            text (str): Ham metin
        
        Returns:
            str: Temizlenmiş metin
        """
        # Özel karakterleri ve fazladan boşlukları temizle
        text = re.sub(r'[^a-zA-ZğüşöçİĞÜŞÖÇ\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Metni tokenize eder ve modele uygun formata getirir
        
        Args:
            text (str): Temizlenmiş metin
        
        Returns:
            Dict[str, torch.Tensor]: Tokenize edilmiş metin
        """
        cleaned_text = self.clean_text(text)
        return self.tokenizer(cleaned_text, return_tensors='pt', padding=True, truncation=True)
    
    def generate_embedding(self, text: str) -> torch.Tensor:
        """
        Metinden anlamsal embedding oluşturur
        
        Args:
            text (str): Girdi metni
        
        Returns:
            torch.Tensor: Metin embeddingi
        """
        tokens = self.tokenize(text)
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        # CLS token'ının embeddingi
        return outputs.last_hidden_state[:, 0, :]
    
    def process(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)