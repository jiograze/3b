import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from .dataset_loaders import PointCloudDataLoader

class UnifiedDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.loader = PointCloudDataLoader()
        
        # Tüm veri setlerini birleştir
        self.data_sources = {
            'thingi10k': self.root_dir / 'datasets/thingi10k',
            'shapenet': self.root_dir / 'datasets/ShapeNet',
            'abc': self.root_dir / 'datasets/abc',
            'google_scanned': self.root_dir / 'datasets/google_scanned',
            'kitti360': self.root_dir / 'datasets/KITTI360'
        }
        
        # Veri setlerini indexle
        self.samples = self._index_datasets()
        
    def _index_datasets(self):
        """Tüm veri setlerindeki örnekleri indexle"""
        samples = []
        
        for source_name, source_path in self.data_sources.items():
            if source_path.exists():
                # Her veri setine özel indexleme
                if source_name == 'shapenet':
                    samples.extend(self._index_shapenet(source_path))
                elif source_name == 'thingi10k':
                    samples.extend(self._index_thingi10k(source_path))
                # Diğer veri setleri için benzer indexleme...
                
        return samples
        
    def _index_shapenet(self, path):
        """ShapeNet veri setini indexle"""
        samples = []
        for model_path in path.rglob('*.obj'):
            metadata_path = model_path.parent / 'metadata.json'
            if metadata_path.exists():
                samples.append({
                    'path': model_path,
                    'type': 'shapenet',
                    'metadata': metadata_path
                })
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Veriyi yükle
        point_cloud = self.loader.load_point_cloud(sample['path'])
        metadata = self._load_metadata(sample['metadata'])
        
        # Dönüşümleri uygula
        if self.transforms:
            point_cloud = self.transforms(point_cloud)
            
        return metadata['description'], point_cloud
        
    def _load_metadata(self, path):
        """Metadata dosyasını yükle"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) 