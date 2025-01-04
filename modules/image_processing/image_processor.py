import torch
import torchvision.transforms as T
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import numpy as np

class ImageProcessor:
    """Görüntü işleme sınıfı"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Konfigürasyon
        """
        self.config = config
        
    def process_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Görüntüyü işle
        
        Args:
            image: Görüntü
            
        Returns:
            İşlenmiş görüntü
        """
        # Görüntüyü yükle
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Dönüşümleri uygula
        transformer = ImageTransformer(self.config)
        return transformer(image)

class ImageTransformer:
    """Görüntü dönüştürme sınıfı"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = 'train'
    ):
        """
        Args:
            config: Konfigürasyon
            mode: Mod ('train' veya 'eval')
        """
        self.config = config
        self.mode = mode
        
        # Dönüşümleri ayarla
        self.transforms = self._setup_transforms()
        
    def _setup_transforms(self) -> T.Compose:
        """Dönüşümleri ayarla"""
        transform_list = []
        
        # Yeniden boyutlandırma
        transform_list.append(
            T.Resize(
                self.config['image_size'],
                interpolation=T.InterpolationMode.BICUBIC
            )
        )
        
        # Eğitim modunda veri artırma
        if self.mode == 'train':
            # Rastgele kırpma
            transform_list.append(
                T.RandomCrop(
                    self.config['image_size'],
                    padding=self.config.get('padding', 4)
                )
            )
            
            # Rastgele yatay çevirme
            if self.config.get('random_flip', True):
                transform_list.append(T.RandomHorizontalFlip())
                
            # Rastgele renk dönüşümleri
            if self.config.get('color_jitter', True):
                transform_list.append(
                    T.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1
                    )
                )
                
            # Rastgele gri tonlama
            if self.config.get('random_grayscale', True):
                transform_list.append(T.RandomGrayscale(p=0.2))
                
        else:
            # Değerlendirme modunda merkezi kırpma
            transform_list.append(
                T.CenterCrop(self.config['image_size'])
            )
            
        # Tensöre çevir
        transform_list.extend([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        return T.Compose(transform_list)
        
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Dönüşümleri uygula
        
        Args:
            image: Görüntü
            
        Returns:
            Dönüştürülmüş görüntü
        """
        # NumPy dizisini PIL görüntüsüne çevir
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Tensörü PIL görüntüsüne çevir
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
            image = image.numpy()
            image = Image.fromarray((image * 255).astype(np.uint8))
            
        # Dönüşümleri uygula
        return self.transforms(image)
        
    def inverse_transform(
        self,
        tensor: torch.Tensor
    ) -> Union[np.ndarray, torch.Tensor]:
        """Ters dönüşüm uygula
        
        Args:
            tensor: Görüntü tensörü
            
        Returns:
            Orijinal görüntü
        """
        # Normalizasyonu geri al
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor * std + mean
        
        # Tensörü NumPy dizisine çevir
        image = tensor.permute(1, 2, 0).numpy()
        
        # [0, 1] aralığına normalize et
        image = np.clip(image, 0, 1)
        
        return image