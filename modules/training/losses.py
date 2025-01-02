import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency
)
from pytorch3d.ops import sample_points_from_meshes
from typing import Dict, Tuple

class Otuken3DLoss(nn.Module):
    """Ötüken3D model loss sınıfı."""
    
    def __init__(
        self,
        chamfer_weight: float = 1.0,
        edge_weight: float = 0.1,
        normal_weight: float = 0.01,
        laplacian_weight: float = 0.1,
        num_samples: int = 5000
    ):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.edge_weight = edge_weight
        self.normal_weight = normal_weight
        self.laplacian_weight = laplacian_weight
        self.num_samples = num_samples
        
    def chamfer_loss(
        self,
        pred_points: torch.Tensor,
        target_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chamfer distance hesaplar."""
        return chamfer_distance(pred_points, target_points)
        
    def mesh_regularization_loss(
        self,
        pred_mesh
    ) -> Dict[str, torch.Tensor]:
        """Mesh düzenlileştirme loss'larını hesaplar."""
        edge_loss = mesh_edge_loss(pred_mesh)
        normal_loss = mesh_normal_consistency(pred_mesh)
        laplacian_loss = mesh_laplacian_smoothing(pred_mesh)
        
        return {
            "edge_loss": edge_loss,
            "normal_loss": normal_loss,
            "laplacian_loss": laplacian_loss
        }
        
    def forward(
        self,
        pred_voxels: torch.Tensor,
        target_mesh,
        return_components: bool = False
    ) -> torch.Tensor:
        """Loss hesaplar."""
        # Voxel'lerden mesh oluştur
        from mcubes import marching_cubes
        vertices, faces = [], []
        
        for voxel in pred_voxels:
            v, f = marching_cubes(voxel[0].cpu().numpy(), 0.5)
            vertices.append(torch.tensor(v, device=pred_voxels.device))
            faces.append(torch.tensor(f, device=pred_voxels.device))
            
        pred_mesh = torch.nn.utils.rnn.pad_sequence(vertices, batch_first=True)
        pred_faces = torch.nn.utils.rnn.pad_sequence(faces, batch_first=True)
        
        # Nokta bulutu örnekle
        pred_points = sample_points_from_meshes(pred_mesh, self.num_samples)
        target_points = sample_points_from_meshes(target_mesh, self.num_samples)
        
        # Chamfer loss
        chamfer_loss, _ = self.chamfer_loss(pred_points, target_points)
        
        # Mesh regularization
        reg_losses = self.mesh_regularization_loss(pred_mesh)
        
        # Toplam loss
        total_loss = (
            self.chamfer_weight * chamfer_loss +
            self.edge_weight * reg_losses["edge_loss"] +
            self.normal_weight * reg_losses["normal_loss"] +
            self.laplacian_weight * reg_losses["laplacian_loss"]
        )
        
        if return_components:
            return {
                "total_loss": total_loss,
                "chamfer_loss": chamfer_loss,
                **reg_losses
            }
            
        return total_loss

class Metrics:
    """Model değerlendirme metrikleri."""
    
    @staticmethod
    def compute_iou(
        pred_voxels: torch.Tensor,
        target_voxels: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Intersection over Union (IoU) hesaplar."""
        pred_binary = (pred_voxels > threshold).float()
        target_binary = (target_voxels > threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = (pred_binary + target_binary).clamp(0, 1).sum()
        
        return intersection / (union + 1e-8)
        
    @staticmethod
    def compute_fscore(
        pred_points: torch.Tensor,
        target_points: torch.Tensor,
        threshold: float = 0.01
    ) -> torch.Tensor:
        """F-score hesaplar."""
        # Chamfer distance
        dist1, dist2 = chamfer_distance(pred_points, target_points)
        
        # Precision ve recall
        precision = (dist2 < threshold).float().mean()
        recall = (dist1 < threshold).float().mean()
        
        # F-score
        f_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f_score
        
    @staticmethod
    def compute_accuracy(
        pred_points: torch.Tensor,
        target_points: torch.Tensor,
        threshold: float = 0.01
    ) -> torch.Tensor:
        """Nokta bulutu doğruluğu hesaplar."""
        dist1, _ = chamfer_distance(pred_points, target_points)
        accuracy = (dist1 < threshold).float().mean()
        return accuracy 