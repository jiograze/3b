import torch
import torch.nn as nn

class Otuken3DModel(nn.Module):
    def __init__(self, voxel_size=64, latent_dim=768, num_classes=10, device='cuda'):
        super().__init__()
        
        self.device = device
        self.voxel_size = voxel_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 3D CNN encoder
        self.encoder = nn.Sequential(
            # Input: Bx1x64x64x64
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),  # -> Bx32x32x32x32
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),  # -> Bx64x16x16x16
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # -> Bx128x8x8x8
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # -> Bx256x4x4x4
            nn.BatchNorm3d(256),
            nn.ReLU(),
            
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),  # -> Bx512x2x2x2
            nn.BatchNorm3d(512),
            nn.ReLU(),
            
            nn.Conv3d(512, latent_dim, kernel_size=2, stride=1, padding=0),  # -> BxLx1x1x1
            nn.BatchNorm3d(latent_dim),
            nn.ReLU()
        )
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, num_classes)
        )
        
        # Move model to device
        self.to(device)
        
    def forward(self, x):
        # Encode
        x = self.encoder(x)  # -> BxLx1x1x1
        x = x.view(x.size(0), -1)  # -> BxL
        
        # Classify
        logits = self.classifier(x)  # -> BxC
        
        return logits
        
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'voxel_size': self.voxel_size,
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.voxel_size = checkpoint.get('voxel_size', 64)
        self.latent_dim = checkpoint.get('latent_dim', 768)
        self.num_classes = checkpoint.get('num_classes', 10)
        self.eval() 