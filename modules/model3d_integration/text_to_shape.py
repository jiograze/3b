"""Text to 3D shape generation module."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from diffusers import DiffusionPipeline
import trimesh
import numpy as np
from typing import Optional, Dict, Any, Union

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

class TextEncoder(nn.Module):
    """Text encoding module using pre-trained transformer."""
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
    def forward(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]

class ShapeGenerator(nn.Module):
    """3D shape generation module."""
    
    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_ATTENTION_HEADS,
        voxel_resolution: int = VOXEL_RESOLUTION
    ):
        super().__init__()
        
        self.voxel_resolution = voxel_resolution
        
        # Upsampling network
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 4),
            
            nn.Linear(hidden_dim * 4, voxel_resolution * voxel_resolution * voxel_resolution),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate 3D voxel grid from encoded text."""
        batch_size = x.shape[0]
        voxels = self.generator(x)
        return voxels.view(batch_size, self.voxel_resolution, self.voxel_resolution, self.voxel_resolution)

class TextToShape(BaseModel):
    """Main text to 3D shape model."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(device)
        self.config = config or {}
        
        # Initialize components
        self.text_encoder = TextEncoder()
        self.shape_generator = ShapeGenerator()
        
        # Optional diffusion model for refinement
        self.diffusion = None
        if self.config.get("use_diffusion", False):
            self.diffusion = DiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16
            )
        
        # Move to device
        self.to(self.device)
        
    def load(self, path: str) -> None:
        """Load model weights."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.text_encoder.load_state_dict(checkpoint["text_encoder"])
            self.shape_generator.load_state_dict(checkpoint["shape_generator"])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def save(self, path: str) -> None:
        """Save model weights."""
        try:
            torch.save({
                "text_encoder": self.text_encoder.state_dict(),
                "shape_generator": self.shape_generator.state_dict()
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            raise ModelError(f"Failed to save model: {str(e)}")
    
    def predict(
        self,
        text: Union[str, list[str]],
        return_mesh: bool = True
    ) -> Union[torch.Tensor, trimesh.Trimesh]:
        """Generate 3D shape from text description."""
        try:
            # Convert to list if single string
            if isinstance(text, str):
                text = [text]
            
            # Generate voxels
            with torch.no_grad():
                # Encode text
                text_features = self.text_encoder(text)
                
                # Generate shape
                voxels = self.shape_generator(text_features)
                
                # Optional diffusion refinement
                if self.diffusion is not None:
                    voxels = self.diffusion(
                        voxels,
                        guidance_scale=7.5
                    ).images
            
            if return_mesh:
                # Convert voxels to mesh using marching cubes
                voxels_np = voxels[0].cpu().numpy()  # Take first sample
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
            vertices = vertices / self.shape_generator.voxel_resolution
            vertices = vertices * 2 - 1
            
            return vertices, faces
            
        except Exception as e:
            raise ProcessingError(f"Voxel to mesh conversion failed: {str(e)}") 