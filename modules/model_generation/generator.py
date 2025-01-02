import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Optional, List, Tuple
import numpy as np
import open3d as o3d
import os

class ModelGenerator(nn.Module):
    def __init__(self, config: Dict):
        """
        Model Generator sınıfı
        
        Args:
            config (Dict): Model yapılandırması
        """
        super().__init__()
        
        print("Model Generator başlatılıyor...")
        print("Yapılandırma yüklendi.")
        
        # Cihazı ayarla
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else 
                                 "cpu")
        print(f"Device: {self.device}")
        
        # CLIP modelini yükle
        print("CLIP modeli yükleniyor...")
        self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP modeli yüklendi.")
        
        # Model parametreleri
        self.hidden_size = config.get("hidden_size", 512)
        self.num_points = config.get("num_points", 2048)
        
        # Encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(512, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Decoder
        self.point_decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_points * 3)
        )
        
        # Normal tahmin edici
        self.normal_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_points * 3)
        )
        
        # Kamera parametreleri
        print("Kamera parametreleri ayarlanıyor...")
        self.camera_distance = 2.0
        self.elevation = 30.0
        self.azimuth = 0.0
        print("Kamera parametreleri ayarlandı.")
        
        # Modeli GPU'ya taşı
        self.to(self.device)
        print("Model Generator başarıyla başlatıldı.")
    
    def forward(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        İleri geçiş
        
        Args:
            text (str): Metin açıklaması
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Nokta bulutu ve normal vektörleri
        """
        # Metni CLIP ile kodla
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        text_features = self.clip(**tokens).last_hidden_state.mean(dim=1)
        
        # Metin özelliklerini kodla
        latent = self.text_encoder(text_features)
        
        # Nokta bulutu ve normalleri oluştur
        points = self.point_decoder(latent).view(-1, self.num_points, 3)
        normals = self.normal_predictor(latent).view(-1, self.num_points, 3)
        normals = F.normalize(normals, dim=-1)
        
        return points, normals
    
    def generate(self, text: str, output_path: Optional[str] = None, format: str = "obj") -> str:
        """
        3D model oluştur
        
        Args:
            text (str): Metin açıklaması
            output_path (Optional[str]): Çıktı dosya yolu
            format (str): Çıktı formatı ("obj", "stl", "ply", "off", "gltf", "fbx")
            
        Returns:
            str: Oluşturulan model dosyasının yolu
        """
        # Desteklenen formatları kontrol et
        supported_formats = ["obj", "stl", "ply", "off", "gltf"]
        format = format.lower()
        if format not in supported_formats:
            raise ValueError(f"Desteklenmeyen format: {format}. Desteklenen formatlar: {supported_formats}")
        
        # Modeli değerlendirme moduna al
        self.eval()
        
        with torch.no_grad():
            # Nokta bulutu ve normalleri oluştur
            points, normals = self(text)
            
            # CPU'ya taşı ve numpy'a dönüştür
            points = points[0].cpu().numpy()
            normals = normals[0].cpu().numpy()
            
            # Open3D point cloud oluştur
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # Mesh oluştur
            print(f"Mesh oluşturuluyor: {text}")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)
            
            # Mesh'i optimize et
            print("Mesh optimize ediliyor...")
            mesh.compute_vertex_normals()
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=len(mesh.triangles) // 2)
            
            # Çıktı yolunu ayarla
            if output_path is None:
                output_path = f"generated_model.{format}"
            else:
                # Uzantıyı değiştir
                output_path = os.path.splitext(output_path)[0] + f".{format}"
            
            # Çıktı dizinini oluştur
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            print(f"Model kaydediliyor: {output_path}")
            
            # Formatına göre kaydet
            if format == "obj":
                o3d.io.write_triangle_mesh(output_path, mesh)
            elif format == "stl":
                o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
            elif format == "ply":
                o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
            elif format == "off":
                o3d.io.write_triangle_mesh(output_path, mesh)
            elif format == "gltf":
                o3d.io.write_triangle_mesh(output_path, mesh)
            
            return output_path
    
    def update_camera(self, distance: float, elevation: float, azimuth: float) -> None:
        """
        Kamera parametrelerini güncelle
        
        Args:
            distance (float): Kamera mesafesi
            elevation (float): Yükseklik açısı
            azimuth (float): Azimut açısı
        """
        self.camera_distance = distance
        self.elevation = elevation
        self.azimuth = azimuth
