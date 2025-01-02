import torch
import numpy as np
import trimesh
from typing import Dict, Tuple

def convert_mesh_format(vertices: torch.Tensor,
                       faces: torch.Tensor) -> trimesh.Trimesh:
    """Convert mesh to trimesh format"""
    vertices = vertices.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def optimize_topology(mesh: trimesh.Trimesh,
                     target_faces: int = 5000) -> trimesh.Trimesh:
    """Mesh topolojisini optimize et"""
    return mesh.simplify_quadratic_decimation(target_faces)

def compute_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Mesh normallerini hesapla"""
    return mesh.vertex_normals
