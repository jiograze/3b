import os
import torch
import numpy as np
from typing import Optional, Union
import trimesh
import matplotlib.pyplot as plt
from PIL import Image

from modules.training.otuken3d_model import Otuken3DModel
from modules.training.data_processing import TextPreprocessor

class Otuken3DInference:
    """Ötüken3D model inference sınıfı."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        output_dir: str = "outputs/demo"
    ):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model yükle
        self.model = Otuken3DModel(device=device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Önişleyiciler
        self.text_processor = TextPreprocessor()
        
    @torch.no_grad()
    def generate_mesh(
        self,
        text: str,
        condition: Optional[str] = None,
        output_path: Optional[str] = None,
        resolution_scale: float = 1.0,
        return_mesh: bool = False
    ) -> Optional[trimesh.Trimesh]:
        """Metinden 3D mesh üretir."""
        # Metni önişle
        text = self.text_processor(text)
        if condition:
            condition = self.text_processor(condition)
            
        # Mesh üret
        mesh = self.model.generate_mesh(
            text=text,
            condition=condition,
            resolution_scale=resolution_scale
        )
        
        # Trimesh'e dönüştür
        vertices = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()
        mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Kaydet
        if output_path:
            mesh_trimesh.export(output_path)
            
        if return_mesh:
            return mesh_trimesh
            
    def visualize_mesh(
        self,
        mesh: trimesh.Trimesh,
        output_path: Optional[str] = None,
        elevation: float = 30.0,
        azimuth: float = 45.0
    ) -> Optional[Image.Image]:
        """Mesh'i görselleştirir."""
        # Scene oluştur
        scene = trimesh.Scene()
        scene.add_geometry(mesh)
        
        # Kamera ayarları
        camera_transform = trimesh.transformations.rotation_matrix(
            angle=np.radians(elevation),
            direction=[1, 0, 0]
        )
        camera_transform = trimesh.transformations.rotation_matrix(
            angle=np.radians(azimuth),
            direction=[0, 0, 1],
            point=scene.centroid
        ) @ camera_transform
        
        # Render
        img = scene.save_image(
            resolution=(800, 800),
            camera_transform=camera_transform
        )
        
        # PIL Image'e dönüştür
        img = Image.fromarray(img)
        
        # Kaydet
        if output_path:
            img.save(output_path)
            
        return img
        
    def demo(
        self,
        text: str,
        condition: Optional[str] = None,
        output_prefix: str = "demo",
        views: int = 4
    ):
        """Demo çalıştırır."""
        # Mesh üret
        mesh = self.generate_mesh(
            text=text,
            condition=condition,
            return_mesh=True
        )
        
        # Farklı açılardan görselleştir
        fig, axes = plt.subplots(1, views, figsize=(4*views, 4))
        for i in range(views):
            azimuth = i * (360 / views)
            img = self.visualize_mesh(mesh, elevation=30.0, azimuth=azimuth)
            axes[i].imshow(img)
            axes[i].axis('off')
            
        # Başlık ekle
        title = f"Text: {text}"
        if condition:
            title += f"\nCondition: {condition}"
        fig.suptitle(title, fontsize=12)
        
        # Kaydet
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{output_prefix}_views.png"))
        plt.close()
        
        # Mesh'i kaydet
        mesh.export(os.path.join(self.output_dir, f"{output_prefix}.obj"))
        
def main():
    """Demo script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ötüken3D demo")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Model checkpoint dosyası")
    parser.add_argument("--text", type=str, required=True,
                      help="3D model üretmek için metin açıklaması")
    parser.add_argument("--condition", type=str,
                      help="Opsiyonel condition metni")
    parser.add_argument("--output_dir", type=str, default="outputs/demo",
                      help="Çıktı dizini")
    parser.add_argument("--views", type=int, default=4,
                      help="Görselleştirme için görüntü sayısı")
    
    args = parser.parse_args()
    
    # Inference
    demo = Otuken3DInference(
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    # Demo çalıştır
    demo.demo(
        text=args.text,
        condition=args.condition,
        views=args.views
    )

if __name__ == "__main__":
    main() 