"""
3B model format dönüştürücü modülü
"""

import os
from pathlib import Path
from typing import Dict, Any, Set, Optional
import logging
from contextlib import contextmanager
import tempfile
import shutil

from src.core.types import PathLike
from src.core.exceptions import ProcessingError

logger = logging.getLogger(__name__)

class FormatConverter:
    """3B model format dönüştürücü sınıfı"""
    
    # Desteklenen formatlar
    SUPPORTED_FORMATS = {'.obj', '.stl', '.ply', '.glb', '.gltf', '.fbx', '.dae'}
    
    # Format dönüşüm matrisi
    CONVERSION_MATRIX = {
        '.obj': {'.stl', '.ply', '.glb', '.gltf'},
        '.stl': {'.obj', '.ply'},
        '.ply': {'.obj', '.stl', '.glb'},
        '.glb': {'.gltf', '.obj', '.ply'},
        '.gltf': {'.glb', '.obj'},
        '.fbx': {'.obj', '.glb', '.gltf'},
        '.dae': {'.obj', '.fbx', '.glb'}
    }
    
    def __init__(self):
        """Format dönüştürücü başlatıcı"""
        self.temp_files = set()
        
    @contextmanager
    def _temp_directory(self):
        """Geçici dizin context manager'ı"""
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="otuken3d_converter_")
            yield Path(temp_dir)
        finally:
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Geçici dizin temizleme hatası: {str(e)}")
                    
    def _validate_formats(self, input_format: str, output_format: str) -> None:
        """Format uyumluluğunu kontrol eder
        
        Args:
            input_format: Girdi formatı
            output_format: Çıktı formatı
            
        Raises:
            ValueError: Desteklenmeyen veya uyumsuz format
        """
        if input_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen girdi formatı: {input_format}")
            
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen çıktı formatı: {output_format}")
            
        if output_format not in self.CONVERSION_MATRIX[input_format]:
            raise ValueError(f"Uyumsuz format dönüşümü: {input_format} -> {output_format}")
            
    def convert(self, 
                input_path: PathLike,
                output_format: str,
                output_path: Optional[PathLike] = None,
                **kwargs) -> Dict[str, Any]:
        """3B model dosyasını dönüştürür
        
        Args:
            input_path: Girdi dosyası yolu
            output_format: Hedef format
            output_path: Çıktı dosyası yolu (None ise geçici dosya oluşturulur)
            **kwargs: Dönüşüm parametreleri
            
        Returns:
            Dönüşüm sonucu ve dosya bilgileri
            
        Raises:
            ProcessingError: Dönüşüm hatası
            ValueError: Geçersiz parametre
        """
        try:
            input_path = Path(input_path)
            if not input_path.is_file():
                raise ValueError(f"Girdi dosyası bulunamadı: {input_path}")
                
            input_format = input_path.suffix.lower()
            output_format = output_format.lower() if output_format.startswith('.') else f'.{output_format.lower()}'
            
            # Format kontrolü
            self._validate_formats(input_format, output_format)
            
            if output_path:
                output_path = Path(output_path)
                if output_path.suffix.lower() != output_format:
                    output_path = output_path.with_suffix(output_format)
                    
            with self._temp_directory() as temp_dir:
                try:
                    # Dönüşüm işlemi
                    if output_path:
                        result_path = output_path
                    else:
                        result_path = temp_dir / f"converted{output_format}"
                        
                    self._convert_file(input_path, result_path, **kwargs)
                    
                    return {
                        'status': 'success',
                        'input_path': str(input_path),
                        'output_path': str(result_path),
                        'input_format': input_format,
                        'output_format': output_format,
                        'parameters': kwargs
                    }
                    
                except Exception as e:
                    logger.error(f"Dönüştürme hatası: {str(e)}")
                    raise ProcessingError(f"Dönüştürme hatası: {str(e)}")
                    
        except Exception as e:
            logger.error(f"İşlem hatası: {str(e)}")
            raise
            
    def _convert_file(self, input_path: Path, output_path: Path, **kwargs) -> None:
        """Dosya dönüşümünü gerçekleştirir
        
        Args:
            input_path: Girdi dosyası yolu
            output_path: Çıktı dosyası yolu
            **kwargs: Dönüşüm parametreleri
            
        Raises:
            ProcessingError: Dönüşüm hatası
        """
        try:
            # TODO: Format dönüşüm kodu
            # Örnek: trimesh, assimp veya blender kullanarak dönüşüm
            pass
            
        except Exception as e:
            raise ProcessingError(f"Dosya dönüştürme hatası: {str(e)}")
            
    def __del__(self):
        """Yıkıcı metod - kaynakları temizler"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Geçici dosya temizleme hatası: {str(e)}") 