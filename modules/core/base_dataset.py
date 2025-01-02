from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import Dataset
from pathlib import Path
from utils.logging.logger import get_logger
from utils.helpers.exceptions import DatasetError

logger = get_logger('dataset')

class BaseDataset(Dataset, ABC):
    """Temel veri seti sınıfı"""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self._load_dataset()
        
    @abstractmethod
    def _load_dataset(self) -> None:
        """Veri setini yükle"""
        pass
        
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Belirtilen indeksteki örneği getir"""
        pass
        
    def __len__(self) -> int:
        """Veri seti uzunluğunu döndür"""
        return len(self.samples)
        
    def get_sample_paths(self) -> List[Path]:
        """Örnek dosya yollarını getir"""
        return [Path(sample['path']) for sample in self.samples]
        
    def get_class_distribution(self) -> Dict[str, int]:
        """Sınıf dağılımını getir"""
        try:
            distribution = {}
            for sample in self.samples:
                label = sample.get('label')
                if label:
                    distribution[label] = distribution.get(label, 0) + 1
            return distribution
        except Exception as e:
            logger.warning(f"Sınıf dağılımı hesaplanamadı: {str(e)}")
            return {}
            
    def get_split_indices(self, train_ratio: float = 0.8, 
                         val_ratio: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
        """Veri seti bölme indekslerini getir"""
        try:
            import numpy as np
            indices = np.arange(len(self))
            np.random.shuffle(indices)
            
            train_size = int(train_ratio * len(self))
            val_size = int(val_ratio * len(self))
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()
            
        except Exception as e:
            raise DatasetError(f"Veri seti bölme hatası: {str(e)}")
            
    def verify_samples(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Örnek dosyaların varlığını kontrol et"""
        valid_samples = []
        errors = []
        
        for sample in self.samples:
            path = Path(sample['path'])
            if not path.exists():
                errors.append(f"Dosya bulunamadı: {path}")
            else:
                valid_samples.append(sample)
                
        if errors:
            logger.warning(f"{len(errors)} geçersiz örnek bulundu")
            
        return valid_samples, errors
        
    def clean_invalid_samples(self) -> int:
        """Geçersiz örnekleri temizle"""
        original_count = len(self.samples)
        valid_samples, _ = self.verify_samples()
        self.samples = valid_samples
        removed_count = original_count - len(self.samples)
        
        if removed_count > 0:
            logger.info(f"{removed_count} geçersiz örnek temizlendi")
            
        return removed_count
        
    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """Belirtilen indeksteki örnek bilgilerini getir"""
        try:
            sample = self.samples[index]
            info = {
                'index': index,
                'path': str(sample['path']),
                'size': Path(sample['path']).stat().st_size,
                'label': sample.get('label', 'Unknown')
            }
            return info
        except Exception as e:
            raise DatasetError(f"Örnek bilgisi alınamadı: {str(e)}")
            
    def print_dataset_summary(self) -> None:
        """Veri seti özetini yazdır"""
        logger.info("Veri Seti Özeti:")
        logger.info(f"Toplam örnek sayısı: {len(self)}")
        logger.info(f"Kök dizin: {self.root_dir}")
        
        class_dist = self.get_class_distribution()
        if class_dist:
            logger.info("Sınıf dağılımı:")
            for label, count in class_dist.items():
                logger.info(f"- {label}: {count}")
                
    def save_metadata(self, path: Optional[str] = None) -> None:
        """Veri seti meta verilerini kaydet"""
        try:
            if path is None:
                path = self.root_dir / 'metadata.json'
                
            import json
            metadata = {
                'num_samples': len(self),
                'root_dir': str(self.root_dir),
                'class_distribution': self.get_class_distribution(),
                'samples': self.samples
            }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            logger.info(f"Meta veriler kaydedildi: {path}")
            
        except Exception as e:
            raise DatasetError(f"Meta veri kaydetme hatası: {str(e)}") 