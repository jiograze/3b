import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
from datasets import load_dataset
from typing import Dict, Any, List, Tuple
from pathlib import Path
from ..model_generation.point_e_manager import PointEManager
from ..point_e.util.point_cloud import PointCloud

class BaseDatasetLoader:
    def __init__(self):
        self.supported_extensions = ['.obj', '.stl', '.ply']
    
    def load_mesh(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """3D model dosyasını yükler ve nokta bulutu ile normal vektörleri döndürür."""
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.compute_vertex_normals()
        
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        
        return points, normals

class ShapeNetDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.models = []
        self.categories = {}
        
        # ShapeNet metadata'yı yükle
        with open(os.path.join(path, 'taxonomy.json'), 'r') as f:
            self.taxonomy = json.load(f)
        
        # Model listesini oluştur
        for category in self.taxonomy:
            category_path = os.path.join(path, category['synsetId'])
            if os.path.exists(category_path):
                self.categories[category['synsetId']] = category['name']
                for model_id in os.listdir(category_path):
                    model_path = os.path.join(category_path, model_id, 'models', 'model_normalized.obj')
                    if os.path.exists(model_path):
                        self.models.append({
                            'path': model_path,
                            'category': category['name'],
                            'id': model_id
                        })
    
    def __len__(self) -> int:
        return len(self.models)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        model = self.models[idx]
        points, normals = BaseDatasetLoader().load_mesh(model['path'])
        
        return {
            'points': torch.from_numpy(points),
            'normals': torch.from_numpy(normals),
            'text': f"a {model['category']}",
            'category': model['category'],
            'id': model['id']
        }

class ModelNetDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.models = []
        
        # ModelNet kategorilerini tara
        for category in os.listdir(path):
            category_path = os.path.join(path, category)
            if os.path.isdir(category_path):
                for file in os.listdir(category_path):
                    if file.endswith('.off'):
                        self.models.append({
                            'path': os.path.join(category_path, file),
                            'category': category
                        })
    
    def __len__(self) -> int:
        return len(self.models)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        model = self.models[idx]
        # OFF dosyasını yükle ve mesh'e dönüştür
        mesh = self._load_off(model['path'])
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        
        return {
            'points': torch.from_numpy(points),
            'normals': torch.from_numpy(normals),
            'text': f"a {model['category']}",
            'category': model['category']
        }
    
    def _load_off(self, file_path: str) -> o3d.geometry.TriangleMesh:
        """OFF dosya formatını yükler"""
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.compute_vertex_normals()
        return mesh

class Thingi10KDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.models = []
        
        # Thingi10K modellerini tara
        for file in os.listdir(path):
            if file.endswith(('.stl', '.obj')):
                self.models.append({
                    'path': os.path.join(path, file),
                    'id': os.path.splitext(file)[0]
                })
    
    def __len__(self) -> int:
        return len(self.models)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        model = self.models[idx]
        points, normals = BaseDatasetLoader().load_mesh(model['path'])
        
        return {
            'points': torch.from_numpy(points),
            'normals': torch.from_numpy(normals),
            'text': f"3D printed object {model['id']}",
            'id': model['id']
        }

class HuggingFaceDataset(Dataset):
    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)
        if isinstance(self.dataset, dict):
            self.dataset = self.dataset['train']  # Varsayılan olarak train split'i kullan
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # HuggingFace dataset'in yapısına göre uyarla
        points = torch.tensor(item['points'], dtype=torch.float32)
        normals = torch.tensor(item['normals'], dtype=torch.float32)
        
        return {
            'points': points,
            'normals': normals,
            'text': item.get('text', 'a 3D object'),
            'metadata': item.get('metadata', {})
        }

class PointCloudDataLoader:
    def __init__(self):
        self.point_e = PointEManager()
        
    def load_point_cloud(self, file_path):
        """3D nokta bulutu dosyasını yükle"""
        try:
            if file_path.endswith('.pt'):
                return torch.load(file_path)
            else:
                # Point-E'nin kendi nokta bulutu formatını kullan
                return PointCloud.load(file_path)
        except Exception as e:
            print(f"Nokta bulutu yükleme hatası: {str(e)}")
            return None
            
    def save_point_cloud(self, point_cloud, file_path):
        """3D nokta bulutunu kaydet"""
        try:
            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(point_cloud, PointCloud):
                point_cloud.save(output_path)
            else:
                torch.save(point_cloud, output_path)
            return True
        except Exception as e:
            print(f"Nokta bulutu kaydetme hatası: {str(e)}")
            return False
            
    def generate_from_text(self, text_prompt, output_path=None, num_points=1024):
        """Metin açıklamasından 3D nokta bulutu oluştur"""
        return self.point_e.generate_point_cloud(
            text_prompt,
            num_points=num_points,
            output_path=output_path
        )

def create_dataset_loader(dataset_type: str, path: str) -> Dataset:
    """Factory method for dataset loaders"""
    loaders = {
        'shapenet': ShapeNetDataset,
        'modelnet': ModelNetDataset,
        'thingi10k': Thingi10KDataset,
        'huggingface': HuggingFaceDataset
    }
    
    if dataset_type not in loaders:
        raise ValueError(f"Desteklenmeyen veri seti türü: {dataset_type}")
    
    return loaders[dataset_type](path) 