import os
import numpy as np
from pathlib import Path

def create_sample_voxels(size=128, num_samples=100):
    """Örnek voxel verisi oluşturur."""
    # Ana veri dizinini oluştur
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Train ve validation dizinlerini oluştur
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Örnek voxel verisi oluştur
    for i in range(num_samples):
        # Rastgele basit geometrik şekiller oluştur
        voxels = np.zeros((size, size, size), dtype=np.float32)
        
        # Küp oluştur
        cube_size = np.random.randint(10, size//4)
        start = np.random.randint(0, size-cube_size, size=3)
        voxels[
            start[0]:start[0]+cube_size,
            start[1]:start[1]+cube_size,
            start[2]:start[2]+cube_size
        ] = 1.0
        
        # Küresel yapı ekle
        center = np.random.randint(size//4, 3*size//4, size=3)
        radius = np.random.randint(5, size//8)
        x, y, z = np.ogrid[:size, :size, :size]
        sphere = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2
        voxels[sphere] = 1.0
        
        # Gürültü ekle
        noise = np.random.normal(0, 0.1, voxels.shape)
        voxels = np.clip(voxels + noise, 0, 1)
        
        # Train/val split
        if i < int(num_samples * 0.8):  # %80 train
            save_dir = train_dir
        else:  # %20 val
            save_dir = val_dir
            
        # Veriyi kaydet
        np.save(save_dir / f"sample_{i:04d}.npy", voxels)
        
    print(f"Örnek veri oluşturuldu:")
    print(f"Train örnekleri: {len(list(train_dir.glob('*.npy')))} adet")
    print(f"Validation örnekleri: {len(list(val_dir.glob('*.npy')))} adet")

if __name__ == "__main__":
    create_sample_voxels(size=128, num_samples=100) 