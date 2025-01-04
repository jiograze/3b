"""
Point cloud to mesh conversion utilities.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .sdf import PointCloudSDFModel

def pc_to_mesh(
    model: PointCloudSDFModel,
    points: torch.Tensor,
    resolution: int = 64,
    threshold: float = 0.0,
    device: Optional[torch.device] = None,
    progress: bool = True,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a point cloud to a mesh using a SDF model.
    
    Args:
        model: The SDF model to use for conversion
        points: Input point cloud tensor of shape [batch_size, 3, n_points]
        resolution: Resolution of the output mesh grid
        threshold: SDF threshold for surface extraction
        device: Device to run the model on
        progress: Whether to show progress bar
        batch_size: Batch size for processing grid points
        
    Returns:
        vertices: Numpy array of vertex positions
        faces: Numpy array of face indices
    """
    if device is None:
        device = points.device
        
    # Create grid points
    grid = create_grid(resolution, device=device)
    grid = grid.unsqueeze(0).expand(points.shape[0], -1, -1)
    
    # Compute SDF values
    sdf_values = []
    for i in range(0, grid.shape[1], batch_size):
        batch = grid[:, i:i+batch_size]
        with torch.no_grad():
            sdf = model.compute_sdf(batch)
        sdf_values.append(sdf.cpu())
    sdf_values = torch.cat(sdf_values, dim=1)
    
    # Extract mesh using marching cubes
    vertices, faces = extract_mesh(
        sdf_values.reshape(resolution, resolution, resolution).numpy(),
        threshold=threshold
    )
    
    return vertices, faces

def create_grid(resolution: int, device: torch.device) -> torch.Tensor:
    """
    Create a 3D grid of points.
    
    Args:
        resolution: Number of points per dimension
        device: Device to create the grid on
        
    Returns:
        Tensor of shape [resolution^3, 3] containing grid points
    """
    points = torch.linspace(-1, 1, resolution, device=device)
    grid = torch.stack(torch.meshgrid(points, points, points, indexing='ij'), dim=-1)
    return grid.reshape(-1, 3).permute(1, 0)

def extract_mesh(
    sdf: np.ndarray,
    threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a mesh from a 3D SDF grid using marching cubes.
    
    Args:
        sdf: 3D numpy array of SDF values
        threshold: SDF threshold for surface extraction
        
    Returns:
        vertices: Numpy array of vertex positions
        faces: Numpy array of face indices
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("Please install scikit-image to use mesh extraction functionality")
        
    vertices, faces, _, _ = measure.marching_cubes(sdf, level=threshold)
    
    # Normalize vertices to [-1, 1]
    vertices = vertices / (sdf.shape[0] - 1) * 2 - 1
    
    return vertices, faces 