import torch
import numpy as np
from typing import List, Tuple, Optional, Union
import open3d as o3d  # Model3D yerine open3d kullanacağız

class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            data = t(data)
        return data

class ToTensor:
    def __call__(self, array: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(array, np.ndarray):
            return torch.from_numpy(array)
        return array

class Normalize:
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.mean = mean
        self.std = std
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            self.mean = points.mean(dim=0)
        if self.std is None:
            self.std = points.std(dim=0)
        
        points = (points - self.mean) / self.std
        return points

class RandomRotation:
    def __init__(self, degrees: Tuple[float, float], p: float = 0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            angle = torch.rand(1) * (self.degrees[1] - self.degrees[0]) + self.degrees[0]
            angle = angle * np.pi / 180
            
            # Rotasyon matrisi
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            rotation_matrix = torch.tensor([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ], dtype=points.dtype)
            
            # Noktaları döndür
            points = torch.matmul(points, rotation_matrix)
        
        return points

class RandomScale:
    def __init__(self, scale_range: Tuple[float, float], p: float = 0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            scale = torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            points = points * scale
        
        return points

class RandomTranslation:
    def __init__(self, translation_range: Tuple[float, float], p: float = 0.5):
        self.translation_range = translation_range
        self.p = p
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            translation = torch.rand(3) * (self.translation_range[1] - self.translation_range[0]) + self.translation_range[0]
            points = points + translation
        
        return points

class Model3DTransform:
    def __init__(self, operation: str = "simplify", target_faces: int = 1000):
        self.operation = operation
        self.target_faces = target_faces
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        # Noktaları Open3D point cloud'a dönüştür
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        
        if self.operation == "simplify":
            # Nokta bulutunu basitleştir
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
        
        return points