"""
Point-E utilities.
"""

from .point_cloud import PointCloud
from .plotting import plot_point_cloud, plot_point_clouds
from .pc_to_mesh import pc_to_mesh
from .ply_util import write_ply

class PointCloudUtils:
    """
    Utility class for point cloud operations.
    """
    @staticmethod
    def normalize(points):
        """
        Normalize point cloud to unit cube.
        """
        min_coords = points.min(dim=1, keepdim=True)[0]
        max_coords = points.max(dim=1, keepdim=True)[0]
        scale = (max_coords - min_coords).max()
        points = (points - min_coords) / scale
        points = points * 2 - 1
        return points
        
    @staticmethod
    def center(points):
        """
        Center point cloud at origin.
        """
        mean = points.mean(dim=1, keepdim=True)
        return points - mean
        
    @staticmethod
    def random_rotate(points):
        """
        Apply random rotation to point cloud.
        """
        import torch
        device = points.device
        batch_size = points.shape[0]
        
        # Generate random rotation matrices
        theta = torch.rand(batch_size, device=device) * 2 * 3.14159
        phi = torch.acos(2 * torch.rand(batch_size, device=device) - 1)
        psi = torch.rand(batch_size, device=device) * 2 * 3.14159
        
        # Compute rotation matrices
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        
        R = torch.zeros(batch_size, 3, 3, device=device)
        R[:, 0, 0] = cos_theta * cos_psi - cos_phi * sin_theta * sin_psi
        R[:, 0, 1] = -cos_theta * sin_psi - cos_phi * sin_theta * cos_psi
        R[:, 0, 2] = sin_phi * sin_theta
        R[:, 1, 0] = sin_theta * cos_psi + cos_phi * cos_theta * sin_psi
        R[:, 1, 1] = -sin_theta * sin_psi + cos_phi * cos_theta * cos_psi
        R[:, 1, 2] = -sin_phi * cos_theta
        R[:, 2, 0] = sin_phi * sin_psi
        R[:, 2, 1] = sin_phi * cos_psi
        R[:, 2, 2] = cos_phi
        
        # Apply rotation
        return torch.bmm(R, points)
        
    @staticmethod
    def random_scale(points, min_scale=0.8, max_scale=1.2):
        """
        Apply random scaling to point cloud.
        """
        import torch
        device = points.device
        batch_size = points.shape[0]
        
        # Generate random scales
        scales = torch.rand(batch_size, 1, 1, device=device) * (max_scale - min_scale) + min_scale
        
        return points * scales

class Visualization:
    """
    Utility class for visualization operations.
    """
    @staticmethod
    def plot_point_cloud(pc, color=True, grid_size=1, fixed_bounds=None):
        """
        Plot a point cloud.
        """
        return plot_point_cloud(pc, color, grid_size, fixed_bounds)
        
    @staticmethod
    def plot_point_clouds(pcs, color=True, grid_size=1, fixed_bounds=None):
        """
        Plot multiple point clouds.
        """
        return plot_point_clouds(pcs, color, grid_size, fixed_bounds)
        
    @staticmethod
    def save_ply(f, coords, rgb=None, faces=None):
        """
        Save point cloud or mesh to PLY file.
        """
        write_ply(f, coords, rgb, faces)

__all__ = [
    'PointCloud',
    'plot_point_cloud',
    'plot_point_clouds',
    'pc_to_mesh',
    'write_ply',
    'PointCloudUtils',
    'Visualization'
]
