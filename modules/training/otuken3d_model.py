import torch
import torch.nn as nn
import numpy as np

class Otuken3DModel(nn.Module):
    def __init__(self, voxel_size=128, latent_dim=768, device='cuda'):
        super().__init__()
        self.voxel_size = voxel_size
        self.latent_dim = latent_dim
        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            
            nn.Flatten(),
            nn.Linear(256 * (voxel_size // 16) ** 3, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * (voxel_size // 16) ** 3),
            nn.Unflatten(1, (256, voxel_size // 16, voxel_size // 16, voxel_size // 16)),
            
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        latent = self.encoder(x)
        
        # Decoding
        reconstruction = self.decoder(latent)
        
        return reconstruction, latent

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    @staticmethod
    def voxelize_pointcloud(points, voxel_size):
        """Convert point cloud to voxel grid"""
        # Normalize points to [0, voxel_size-1]
        min_coords = points.min(dim=0)[0]
        max_coords = points.max(dim=0)[0]
        points = (points - min_coords) * (voxel_size - 1) / (max_coords - min_coords)
        points = points.long()

        # Create empty voxel grid
        voxels = torch.zeros((voxel_size, voxel_size, voxel_size))

        # Fill voxels
        valid_points = (points >= 0) & (points < voxel_size)
        valid_points = valid_points.all(dim=1)
        points = points[valid_points]
        voxels[points[:, 0], points[:, 1], points[:, 2]] = 1

        return voxels.unsqueeze(0)  # Add channel dimension 