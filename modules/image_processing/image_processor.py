import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
from typing import Union

class ImageProcessor:
    def __init__(self, image_size: int = 224):
        """
        Görüntü işleme için gerekli dönüşümleri ve modelleri hazırlar
        
        Args:
            image_size (int): Görüntü boyutu
        """
        self.image_size = image_size
        self.target_size = (256, 256)
        
        # Resnet50 özellik çıkarıcı
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, image_file: Union[str, Image.Image]) -> Image.Image:
        """
        Görüntüyü yükler
        
        Args:
            image_file (Union[str, Image.Image]): Görüntü dosyası veya PIL Image
        
        Returns:
            Image.Image: Yüklenmiş görüntü
        """
        if isinstance(image_file, str):
            return Image.open(image_file).convert('RGB')
        return image_file
    
    def preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Görüntüyü ön işleme tabi tutar
        
        Args:
            image (Union[str, Image.Image]): Görüntü dosyası veya PIL Image
        
        Returns:
            torch.Tensor: Ön işlenmiş görüntü tensörü
        """
        img = self.load_image(image)
        img = img.resize(self.target_size)
        image_tensor = torch.from_numpy(np.array(img)).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return image_tensor.unsqueeze(0)  # Batch boyutu ekle
    
    def extract_features(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Görüntüden özellik çıkarır
        
        Args:
            image (Union[str, Image.Image]): Görüntü dosyası veya PIL Image
        
        Returns:
            torch.Tensor: Çıkarılmış özellikler
        """
        preprocessed_image = self.preprocess(image)
        
        with torch.no_grad():
            features = self.feature_extractor(preprocessed_image)
        
        return features
    
    def process(self, image_file: Union[str, Image.Image]) -> torch.Tensor:
        """
        Görüntüyü tam olarak işler
        
        Args:
            image_file (Union[str, Image.Image]): Görüntü dosyası veya PIL Image
        
        Returns:
            torch.Tensor: İşlenmiş görüntü özellikleri
        """
        return self.extract_features(image_file)