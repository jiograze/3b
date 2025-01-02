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
        
        # Veri dizinini kontrol et
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise ValueError(f"Veri dizini bulunamadı: {split_dir}")
            
        # Veri dosyalarını listele
        self.voxel_files = list(split_dir.glob('*.npy'))
        if not self.voxel_files:
            raise ValueError(f"Veri dosyası bulunamadı: {split_dir}")
            
        print(f"{split} veri seti yüklendi: {len(self.voxel_files)} örnek")

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        # Voxel dosyasını yükle
        voxel_path = self.voxel_files[idx]
        voxel_data = np.load(voxel_path)
        voxel_tensor = torch.from_numpy(voxel_data).float()
        
        # Kanal boyutunu ekle
        if len(voxel_tensor.shape) == 3:
            voxel_tensor = voxel_tensor.unsqueeze(0)
            
        # Transform uygula
        if self.transform:
            voxel_tensor = self.transform(voxel_tensor)
            
        return voxel_tensor

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