import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class Otuken3DDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Tüm sınıf dizinlerini bul
        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name != '__MACOSX']
        if not self.class_dirs:
            raise ValueError(f"Sınıf dizinleri bulunamadı: {self.data_dir}")
            
        # Her sınıf için split dizinindeki OFF dosyalarını bul
        self.samples = []
        self.labels = []
        for class_idx, class_dir in enumerate(sorted(self.class_dirs)):
            split_dir = class_dir / split
            if not split_dir.exists():
                continue
                
            off_files = list(split_dir.glob('*.off'))
            self.samples.extend(off_files)
            self.labels.extend([class_idx] * len(off_files))
            
        if not self.samples:
            raise ValueError(f"OFF dosyası bulunamadı: {split} split")
            
        print(f"{split} veri seti yüklendi: {len(self.samples)} örnek")

    def __len__(self):
        return len(self.samples)

    def read_off(self, file_path):
        """OFF dosyasını oku ve voxelize et."""
        vertices = []
        faces = []
        
        with open(file_path) as f:
            # İlk satırı oku (OFF başlığı)
            line = f.readline().strip()
            if line != 'OFF':
                line = f.readline().strip()
                
            # Vertex ve face sayılarını oku
            n_verts, n_faces, _ = map(int, f.readline().strip().split())
            
            # Vertexleri oku
            for i in range(n_verts):
                vertex = list(map(float, f.readline().strip().split()))
                vertices.append(vertex)
                
            # Faceları oku
            for i in range(n_faces):
                face = list(map(int, f.readline().strip().split()))
                faces.append(face[1:])  # İlk sayı face vertex sayısı
                
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        # Normalize vertices to unit cube
        vertices -= vertices.min(axis=0)
        vertices /= vertices.max()
        
        # Simple voxelization (64x64x64)
        voxel_size = 64
        voxels = np.zeros((voxel_size, voxel_size, voxel_size))
        
        # Vertex positions to voxel coordinates
        voxel_coords = (vertices * (voxel_size-1)).astype(int)
        
        # Mark voxels that contain vertices
        for coord in voxel_coords:
            x, y, z = coord
            voxels[x, y, z] = 1
            
        return voxels

    def __getitem__(self, idx):
        # OFF dosyasını oku ve voxelize et
        off_path = self.samples[idx]
        voxel_data = self.read_off(off_path)
        voxel_tensor = torch.from_numpy(voxel_data).float()
        
        # Kanal boyutunu ekle
        if len(voxel_tensor.shape) == 3:
            voxel_tensor = voxel_tensor.unsqueeze(0)
            
        # Transform uygula
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)
            
        # Label tensor'a çevir
        label = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return voxel_tensor, label

def create_dataloader(data_dir, split='train', batch_size=32, num_workers=4):
    """Veri yükleyici oluşturur."""
    dataset = Otuken3DDataset(data_dir, split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

class VoxelAugmentation:
    """Voxel verisi için basit veri artırma teknikleri."""
    
    @staticmethod
    def random_rotate(voxel, k=None):
        """Voxel'i rastgele döndürür."""
        if k is None:
            k = np.random.randint(4)
        return torch.rot90(voxel, k, dims=[2, 3])
    
    @staticmethod
    def random_flip(voxel):
        """Voxel'i rastgele çevirir."""
        if np.random.random() > 0.5:
            voxel = torch.flip(voxel, dims=[2])
        if np.random.random() > 0.5:
            voxel = torch.flip(voxel, dims=[3])
        return voxel
    
    @staticmethod
    def add_noise(voxel, noise_factor=0.05):
        """Gürültü ekler."""
        noise = torch.randn_like(voxel) * noise_factor
        return torch.clamp(voxel + noise, 0, 1)

def apply_augmentation(voxel_batch):
    """Batch'e veri artırma uygular."""
    augmentation = VoxelAugmentation()
    
    # Rastgele dönüşümler uygula
    voxel_batch = augmentation.random_rotate(voxel_batch)
    voxel_batch = augmentation.random_flip(voxel_batch)
    
    # %50 olasılıkla gürültü ekle
    if np.random.random() > 0.5:
        voxel_batch = augmentation.add_noise(voxel_batch)
        
    return voxel_batch 