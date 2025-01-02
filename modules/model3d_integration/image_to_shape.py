"""Image to 3D shape generation module."""

import torch
import torch.nn as nn
from torchvision import transforms, models
import trimesh
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

from ..core.base import BaseModel
from ..core.logger import setup_logger
from ..core.exceptions import ModelError, ProcessingError
from ..core.constants import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_ATTENTION_HEADS,
    VOXEL_RESOLUTION
)

logger = setup_logger(__name__)

class ImageEncoder(nn.Module):
    """Image encoding module using pre-trained CNN."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        features = self.encoder(images)
        return features.view(features.size(0), -1)  # [batch_size, feature_dim]

class Shape3DDecoder(nn.Module):
    """3D shape decoding module."""
    
    def __init__(
        self,
        input_dim: int = 2048,  # ResNet50 feature dimension
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_ATTENTION_HEADS,
        voxel_resolution: int = VOXEL_RESOLUTION
    ):
        super().__init__()
        
        self.voxel_resolution = voxel_resolution
        
        # 3D shape generation network
        self.decoder = nn.Sequential(
            # Feature processing
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            
            # Upsampling layers
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 4),
            
            # Final voxel grid generation
            nn.Linear(hidden_dim * 4, voxel_resolution * voxel_resolution * voxel_resolution),
            nn.Sigmoid()
        )
        
        # Optional: Multi-view fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads
        )
        
    def forward(self, x: torch.Tensor, multi_view: bool = False) -> torch.Tensor:
        """Generate 3D voxel grid from image features."""
        batch_size = x.shape[0]
        
        if multi_view and x.dim() > 2:
            # Fuse multiple views using attention
            x = x.transpose(0, 1)  # [num_views, batch_size, feature_dim]
            x, _ = self.fusion(x, x, x)
            x = x.mean(dim=0)  # [batch_size, feature_dim]
        
        voxels = self.decoder(x)
        return voxels.view(batch_size, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)

class ImageToShape(BaseModel):
    """Main image to 3D shape model."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(device)
        self.config = config or {}
        
        # Initialize components
        self.image_encoder = ImageEncoder()
        self.shape_decoder = Shape3DDecoder()
        
        # Move to device
        self.to(self.device)
    
    def load(self, path: str) -> None:
        """Load model weights."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.image_encoder.load_state_dict(checkpoint["image_encoder"])
            self.shape_decoder.load_state_dict(checkpoint["shape_decoder"])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def save(self, path: str) -> None:
        """Save model weights."""
        try:
            torch.save({
                "image_encoder": self.image_encoder.state_dict(),
                "shape_decoder": self.shape_decoder.state_dict()
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            raise ModelError(f"Failed to save model: {str(e)}")
    
    def predict(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]],
        return_mesh: bool = True
    ) -> Union[torch.Tensor, trimesh.Trimesh]:
        """Generate 3D shape from input image(s)."""
        try:
            # Process input images
            if isinstance(images, (str, Image.Image)):
                images = [images]
            
            processed_images = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                processed_images.append(self.image_encoder.preprocess(img))
            
            # Stack images and move to device
            image_batch = torch.stack(processed_images).to(self.device)
            
            # Generate 3D shape
            with torch.no_grad():
                # Encode images
                image_features = self.image_encoder(image_batch)
                
                # Generate shape (with multi-view fusion if multiple images)
                voxels = self.shape_decoder(
                    image_features,
                    multi_view=len(images) > 1
                )
            
            if return_mesh:
                # Convert voxels to mesh
                voxels_np = voxels[0].cpu().numpy()
                vertices, faces = self._voxels_to_mesh(voxels_np)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                return mesh
            
            return voxels
            
        except Exception as e:
            raise ProcessingError(f"Shape generation failed: {str(e)}")
    
    def _voxels_to_mesh(
        self,
        voxels: np.ndarray,
        threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert voxel grid to mesh using marching cubes."""
        try:
            from skimage import measure
            
            # Apply marching cubes
            vertices, faces, _, _ = measure.marching_cubes(voxels, threshold)
            
            # Normalize vertices to [-1, 1]
            vertices = vertices / self.shape_decoder.voxel_resolution
            vertices = vertices * 2 - 1
            
            return vertices, faces
            
        except Exception as e:
            raise ProcessingError(f"Voxel to mesh conversion failed: {str(e)}") 