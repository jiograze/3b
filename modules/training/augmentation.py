"""
Data augmentation utilities for 3D point clouds and text.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

class Augmentation3D:
    """
    3D data augmentation for point clouds.
    """
    def __init__(
        self,
        device: torch.device,
        rotation_range: float = 180.0,
        translation_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.02,
        random_crop_prob: float = 0.0,
        random_crop_size_range: Tuple[float, float] = (0.8, 1.0),
    ):
        self.device = device
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.random_crop_prob = random_crop_prob
        self.random_crop_size_range = random_crop_size_range

    def random_rotation_matrix(self, batch_size: int) -> torch.Tensor:
        """
        Generate random rotation matrices.
        
        Args:
            batch_size: Number of rotation matrices to generate
            
        Returns:
            Tensor of shape [batch_size, 3, 3] containing rotation matrices
        """
        # Generate random Euler angles
        angles = torch.rand(batch_size, 3, device=self.device) * 2 * np.pi
        
        # Convert to rotation matrices
        cos_a = torch.cos(angles[:, 0])
        sin_a = torch.sin(angles[:, 0])
        cos_b = torch.cos(angles[:, 1])
        sin_b = torch.sin(angles[:, 1])
        cos_c = torch.cos(angles[:, 2])
        sin_c = torch.sin(angles[:, 2])
        
        R = torch.zeros(batch_size, 3, 3, device=self.device)
        
        # Rotation around X
        R[:, 0, 0] = 1
        R[:, 1, 1] = cos_a
        R[:, 1, 2] = -sin_a
        R[:, 2, 1] = sin_a
        R[:, 2, 2] = cos_a
        
        # Rotation around Y
        R_y = torch.zeros_like(R)
        R_y[:, 0, 0] = cos_b
        R_y[:, 0, 2] = sin_b
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin_b
        R_y[:, 2, 2] = cos_b
        R = torch.bmm(R_y, R)
        
        # Rotation around Z
        R_z = torch.zeros_like(R)
        R_z[:, 0, 0] = cos_c
        R_z[:, 0, 1] = -sin_c
        R_z[:, 1, 0] = sin_c
        R_z[:, 1, 1] = cos_c
        R_z[:, 2, 2] = 1
        R = torch.bmm(R_z, R)
        
        return R

    def __call__(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation to point clouds.
        
        Args:
            points: Point cloud tensor of shape [batch_size, num_points, 3]
            normals: Optional normal vectors of shape [batch_size, num_points, 3]
            
        Returns:
            Tuple of (augmented points, augmented normals)
        """
        batch_size = points.shape[0]
        
        # Random rotation
        R = self.random_rotation_matrix(batch_size)
        points = torch.bmm(points, R.transpose(1, 2))
        if normals is not None:
            normals = torch.bmm(normals, R.transpose(1, 2))
        
        # Random translation
        if self.translation_range > 0:
            t = torch.rand(batch_size, 1, 3, device=self.device) * 2 - 1
            t = t * self.translation_range
            points = points + t
        
        # Random scaling
        if self.scale_range[0] < self.scale_range[1]:
            s = torch.rand(batch_size, 1, 1, device=self.device)
            s = s * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            points = points * s
        
        # Random noise
        if self.noise_std > 0:
            noise = torch.randn_like(points) * self.noise_std
            points = points + noise
        
        # Random cropping
        if self.random_crop_prob > 0 and torch.rand(1).item() < self.random_crop_prob:
            size = torch.rand(1).item() * (self.random_crop_size_range[1] - self.random_crop_size_range[0])
            size = size + self.random_crop_size_range[0]
            center = torch.rand(batch_size, 1, 3, device=self.device) * 2 - 1
            dist = torch.norm(points - center, dim=-1)
            mask = dist < size
            points = points * mask.unsqueeze(-1)
            if normals is not None:
                normals = normals * mask.unsqueeze(-1)
        
        return points, normals

class TextAugmentation:
    """
    Text augmentation for training data.
    """
    def __init__(
        self,
        templates: Optional[List[str]] = None,
        random_drop_prob: float = 0.1,
        random_replace_prob: float = 0.1,
        max_length: int = 77,
    ):
        self.templates = templates or [
            "a 3D model of {}",
            "a rendering of {}",
            "{} in 3D",
            "3D {} model",
            "{} object",
            "3D object of {}",
            "{} shape",
            "3D shape of {}",
        ]
        self.random_drop_prob = random_drop_prob
        self.random_replace_prob = random_replace_prob
        self.max_length = max_length
        
        # Common words for random replacement
        self.common_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "with", "by", "from", "up", "down", "over", "under",
        ]

    def __call__(self, text: str) -> str:
        """
        Apply augmentation to text.
        
        Args:
            text: Input text string
            
        Returns:
            Augmented text string
        """
        # Apply template
        if self.templates and torch.rand(1).item() < 0.5:
            template = np.random.choice(self.templates)
            text = template.format(text)
        
        # Random word dropping
        if self.random_drop_prob > 0:
            words = text.split()
            words = [w for w in words if torch.rand(1).item() > self.random_drop_prob]
            text = " ".join(words)
        
        # Random word replacement
        if self.random_replace_prob > 0 and self.common_words:
            words = text.split()
            for i in range(len(words)):
                if torch.rand(1).item() < self.random_replace_prob:
                    words[i] = np.random.choice(self.common_words)
            text = " ".join(words)
        
        # Truncate to max length
        if self.max_length > 0:
            words = text.split()[:self.max_length]
            text = " ".join(words)
        
        return text 