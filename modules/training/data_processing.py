import numpy as np
import torch
from typing import List, Optional, Tuple, Union
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import random_rotations, Rotate

class PointCloudTransform:
    """Nokta bulutu augmentasyon sınıfı."""
    
    def __init__(
        self,
        rotation_range: float = 180.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        translation_range: float = 0.1,
        jitter_sigma: float = 0.01,
        random_crop_prob: float = 0.5
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.jitter_sigma = jitter_sigma
        self.random_crop_prob = random_crop_prob
        
    def random_rotation(self, points: torch.Tensor) -> torch.Tensor:
        """Rastgele rotasyon uygular."""
        angle = torch.rand(1) * self.rotation_range * np.pi / 180
        rotation = Rotate(random_rotations(1))
        return rotation.transform_points(points)
        
    def random_scale(self, points: torch.Tensor) -> torch.Tensor:
        """Rastgele ölçekleme uygular."""
        scale = torch.rand(1) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        return points * scale
        
    def random_translation(self, points: torch.Tensor) -> torch.Tensor:
        """Rastgele öteleme uygular."""
        translation = (torch.rand(3) * 2 - 1) * self.translation_range
        return points + translation
        
    def add_jitter(self, points: torch.Tensor) -> torch.Tensor:
        """Gaussian gürültü ekler."""
        noise = torch.randn_like(points) * self.jitter_sigma
        return points + noise
        
    def random_crop(self, points: torch.Tensor, num_points: int) -> torch.Tensor:
        """Rastgele nokta seçimi yapar."""
        if torch.rand(1) < self.random_crop_prob:
            idx = torch.randperm(points.shape[0])[:num_points]
            return points[idx]
        return points
        
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Tüm augmentasyonları uygular."""
        points = self.random_rotation(points)
        points = self.random_scale(points)
        points = self.random_translation(points)
        points = self.add_jitter(points)
        points = self.random_crop(points, int(points.shape[0] * 0.8))
        return points

class MeshPreprocessor:
    """3D mesh önişleme sınıfı."""
    
    def __init__(
        self,
        target_vertices: int = 5000,
        normalize: bool = True,
        center: bool = True,
        smooth: bool = True
    ):
        self.target_vertices = target_vertices
        self.normalize = normalize
        self.center = center
        self.smooth = smooth
        
    def simplify_mesh(self, mesh: Meshes) -> Meshes:
        """Mesh'i basitleştirir."""
        from pytorch3d.ops import mesh_face_areas_normals
        
        # Yüz alanlarını hesapla
        face_areas, _ = mesh_face_areas_normals(mesh)
        
        # Hedef yüz sayısını hesapla
        current_faces = mesh.faces_packed().shape[0]
        target_faces = self.target_vertices // 3
        
        if current_faces <= target_faces:
            return mesh
            
        # En küçük alanlı yüzleri kaldır
        _, indices = torch.sort(face_areas)
        keep_faces = indices[current_faces - target_faces:]
        new_faces = mesh.faces_packed()[keep_faces]
        
        return Meshes(verts=[mesh.verts_packed()], faces=[new_faces])
        
    def normalize_mesh(self, mesh: Meshes) -> Meshes:
        """Mesh'i normalize eder."""
        if self.normalize:
            verts = mesh.verts_packed()
            verts = verts / verts.abs().max()
            return Meshes(verts=[verts], faces=[mesh.faces_packed()])
        return mesh
        
    def center_mesh(self, mesh: Meshes) -> Meshes:
        """Mesh'i merkezler."""
        if self.center:
            verts = mesh.verts_packed()
            center = verts.mean(dim=0)
            verts = verts - center
            return Meshes(verts=[verts], faces=[mesh.faces_packed()])
        return mesh
        
    def smooth_mesh(self, mesh: Meshes, iterations: int = 3) -> Meshes:
        """Mesh'i yumuşatır."""
        if self.smooth:
            from pytorch3d.ops import mesh_laplacian_smoothing
            return mesh_laplacian_smoothing(mesh, iterations)
        return mesh
        
    def __call__(self, mesh: Meshes) -> Meshes:
        """Tüm önişlemeleri uygular."""
        mesh = self.simplify_mesh(mesh)
        mesh = self.normalize_mesh(mesh)
        mesh = self.center_mesh(mesh)
        mesh = self.smooth_mesh(mesh)
        return mesh

class TextPreprocessor:
    """Metin önişleme sınıfı."""
    
    def __init__(
        self,
        max_length: int = 77,
        clean_text: bool = True,
        add_special_tokens: bool = True
    ):
        self.max_length = max_length
        self.clean_text = clean_text
        self.add_special_tokens = add_special_tokens
        
    def clean(self, text: str) -> str:
        """Metni temizler."""
        if self.clean_text:
            # Gereksiz boşlukları temizle
            text = " ".join(text.split())
            # Noktalama işaretlerini düzenle
            text = text.replace(",", " , ").replace(".", " . ")
            # Küçük harfe çevir
            text = text.lower()
        return text
        
    def truncate(self, text: str) -> str:
        """Metni kırpar."""
        words = text.split()
        if len(words) > self.max_length:
            words = words[:self.max_length]
        return " ".join(words)
        
    def add_tokens(self, text: str) -> str:
        """Özel tokenleri ekler."""
        if self.add_special_tokens:
            return f"<start> {text} <end>"
        return text
        
    def __call__(self, text: str) -> str:
        """Tüm önişlemeleri uygular."""
        text = self.clean(text)
        text = self.truncate(text)
        text = self.add_tokens(text)
        return text 