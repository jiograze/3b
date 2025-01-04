"""
Doku İşleme Modülü
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
from io import BytesIO

from src.core.base import BaseProcessor
from src.core.types import PathLike, TextureMap, ProcessingResult

class TextureProcessor(BaseProcessor):
    """Doku haritalarını işlemek için sınıf"""
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tiff'}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: İşlemci yapılandırması
        """
        super().__init__(config)
        self.required_keys = ['max_size', 'quality']
        
    def process(self, input_path: PathLike, output_path: PathLike,
                resize: Optional[Tuple[int, int]] = None,
                quality: Optional[int] = None) -> None:
        """Doku haritasını işler
        
        Args:
            input_path: Girdi dosyası yolu
            output_path: Çıktı dosyası yolu
            resize: Yeni boyut (genişlik, yükseklik)
            quality: Çıktı kalitesi (0-100)
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        self.validate_file(input_path)
        self.ensure_dir(output_path.parent)
        
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {input_path.suffix}")
            
        try:
            # Dokuyu yükle
            image = Image.open(input_path)
            
            # Boyutlandırma
            if resize:
                image = image.resize(resize, Image.Resampling.LANCZOS)
                
            # Kalite ayarı
            save_opts = {}
            if quality is not None:
                if quality < 0 or quality > 100:
                    raise ValueError("Kalite 0-100 arasında olmalı")
                save_opts['quality'] = quality
                
            # Dokuyu kaydet
            image.save(output_path, **save_opts)
            
            self.logger.info(f"Doku işlendi: {input_path} -> {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Doku işleme hatası: {str(e)}")
            
    def batch_process(self, input_dir: PathLike, output_dir: PathLike,
                     resize: Optional[Tuple[int, int]] = None,
                     quality: Optional[int] = None) -> ProcessingResult:
        """Birden fazla dokuyu işler
        
        Args:
            input_dir: Girdi klasörü
            output_dir: Çıktı klasörü
            resize: Yeni boyut (genişlik, yükseklik)
            quality: Çıktı kalitesi (0-100)
            
        Returns:
            İşlem sonuçları
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        self.validate_dir(input_dir)
        self.ensure_dir(output_dir)
        
        results = {
            'successful': [],
            'failed': []
        }
        
        # Desteklenen formattaki dosyaları bul
        for format_ in self.SUPPORTED_FORMATS:
            for input_path in input_dir.glob(f"*{format_}"):
                try:
                    output_path = output_dir / input_path.name
                    self.process(
                        input_path,
                        output_path,
                        resize=resize,
                        quality=quality
                    )
                    results['successful'].append(str(input_path))
                except Exception as e:
                    self.logger.error(f"İşleme hatası ({input_path}): {str(e)}")
                    results['failed'].append({
                        'path': str(input_path),
                        'error': str(e)
                    })
                    
        return results
        
    def load_texture(self, path: PathLike) -> TextureMap:
        """Doku haritasını yükler
        
        Args:
            path: Dosya yolu
            
        Returns:
            Doku haritası dizisi
        """
        path = Path(path)
        self.validate_file(path)
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {path.suffix}")
            
        try:
            image = Image.open(path)
            return np.array(image)
        except Exception as e:
            raise RuntimeError(f"Doku yükleme hatası: {str(e)}")
            
    def save_texture(self, texture: TextureMap, path: PathLike,
                    quality: Optional[int] = None) -> None:
        """Doku haritasını kaydeder
        
        Args:
            texture: Doku haritası dizisi
            path: Dosya yolu
            quality: Çıktı kalitesi (0-100)
        """
        path = Path(path)
        self.ensure_dir(path.parent)
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {path.suffix}")
            
        try:
            image = Image.fromarray(texture)
            
            save_opts = {}
            if quality is not None:
                if quality < 0 or quality > 100:
                    raise ValueError("Kalite 0-100 arasında olmalı")
                save_opts['quality'] = quality
                
            image.save(path, **save_opts)
            
            self.logger.info(f"Doku kaydedildi: {path}")
            
        except Exception as e:
            raise RuntimeError(f"Doku kaydetme hatası: {str(e)}")
            
    def optimize(self, texture: TextureMap,
                max_size: Optional[int] = None,
                quality: Optional[int] = None) -> TextureMap:
        """Doku haritasını optimize eder
        
        Args:
            texture: Doku haritası dizisi
            max_size: Maksimum boyut
            quality: Çıktı kalitesi (0-100)
            
        Returns:
            Optimize edilmiş doku haritası
        """
        try:
            image = Image.fromarray(texture)
            original_mode = image.mode
            
            # Boyut sınırlama
            if max_size:
                w, h = image.size
                if w > max_size or h > max_size:
                    ratio = min(max_size/w, max_size/h)
                    new_size = (int(w*ratio), int(h*ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    
            # Kalite ayarı - format bazlı optimizasyon
            if quality is not None:
                if quality < 0 or quality > 100:
                    raise ValueError("Kalite 0-100 arasında olmalı")
                
                # Geçici dosyaya kaydet ve tekrar yükle
                with BytesIO() as buffer:
                    if original_mode == 'RGBA' or original_mode == 'LA':
                        # PNG formatını kullan - alfa kanalı varsa
                        image.save(buffer, format='PNG',
                                 optimize=True,
                                 quality=quality)
                    elif original_mode == 'P':
                        # Palette modunda PNG kullan
                        image.save(buffer, format='PNG',
                                 optimize=True)
                    else:
                        # RGB/L modları için JPEG kullan
                        image.save(buffer, format='JPEG',
                                 quality=quality,
                                 optimize=True)
                    
                    buffer.seek(0)
                    image = Image.open(buffer)
                    
                    # Orijinal moda geri dön
                    if image.mode != original_mode:
                        image = image.convert(original_mode)
                    
            return np.array(image)
            
        except Exception as e:
            raise RuntimeError(f"Doku optimizasyon hatası: {str(e)}")
            
    def validate(self, texture: TextureMap) -> bool:
        """Doku haritasının geçerliliğini kontrol eder
        
        Args:
            texture: Doku haritası dizisi
            
        Returns:
            Geçerliyse True
        """
        try:
            if not isinstance(texture, np.ndarray):
                return False
                
            if texture.ndim != 3:
                return False
                
            if texture.dtype != np.uint8:
                return False
                
            return True
            
        except Exception:
            return False 