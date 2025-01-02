import os
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional
import json
import random

class ShapeDataset(Dataset):
    def __init__(self, dataset_path: str, split: str = "train", num_points: int = 2048):
        """
        3D şekil veri seti
        
        Args:
            dataset_path (str): Veri seti dizini
            split (str): Veri seti bölümü ("train", "val", "test")
            num_points (int): Her modelden örneklenecek nokta sayısı
        """
        super().__init__()
        
        self.dataset_path = dataset_path
        self.split = split
        self.num_points = num_points
        
        # Veri seti yollarını yükle
        self.model_paths = []
        self.text_descriptions = []
        
        # Metadata dosyasını yükle
        metadata_path = os.path.join(dataset_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Split'e göre veriyi filtrele
            for item in metadata:
                if item["split"] == split:
                    model_path = os.path.join(dataset_path, item["model_path"])
                    if os.path.exists(model_path):
                        self.model_paths.append(model_path)
                        self.text_descriptions.append(item["description"])
        
        if len(self.model_paths) == 0:
            raise ValueError(f"Veri seti boş: {dataset_path}")
        
        print(f"Veri seti yüklendi: {len(self.model_paths)} model")
    
    def __len__(self) -> int:
        return len(self.model_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Veri setinden bir örnek al
        
        Args:
            idx (int): Örnek indeksi
            
        Returns:
            Dict[str, torch.Tensor]: Model verisi
        """
        # Model dosyasını yükle
        model_path = self.model_paths[idx]
        mesh = o3d.io.read_triangle_mesh(model_path)
        
        # Nokta bulutu örnekle
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        points = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        
        # Rastgele nokta örnekleme
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
            normals = normals[indices]
        elif len(points) < self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]
            normals = normals[indices]
        
        # Tensörlere dönüştür
        points = torch.from_numpy(points.astype(np.float32))
        normals = torch.from_numpy(normals.astype(np.float32))
        
        # Normalizasyon
        center = points.mean(dim=0)
        points = points - center
        scale = points.abs().max()
        points = points / scale
        
        return {
            "points": points,
            "normals": normals,
            "text": self.text_descriptions[idx],
            "path": model_path
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Batch oluştur
        
        Args:
            batch (List[Dict[str, torch.Tensor]]): Örnekler listesi
            
        Returns:
            Dict[str, torch.Tensor]: Batch verisi
        """
        # Batch boyutunu al
        batch_size = len(batch)
        
        # Her örneğin boyutlarını kontrol et
        num_points = batch[0]["points"].size(0)
        
        # Boş tensörler oluştur
        points = torch.zeros(batch_size, num_points, 3)
        normals = torch.zeros(batch_size, num_points, 3)
        texts = []
        paths = []
        
        # Batch'i doldur
        for i, sample in enumerate(batch):
            points[i] = sample["points"]
            normals[i] = sample["normals"]
            texts.append(sample["text"])
            paths.append(sample["path"])
        
        return {
            "points": points,
            "normals": normals,
            "text": texts,
            "path": paths
        }
    
    def get_item_by_text(self, text: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Metin açıklamasına göre model bul
        
        Args:
            text (str): Metin açıklaması
            
        Returns:
            Optional[Dict[str, torch.Tensor]]: Model verisi
        """
        for i, desc in enumerate(self.text_descriptions):
            if text.lower() in desc.lower():
                return self[i]
        return None