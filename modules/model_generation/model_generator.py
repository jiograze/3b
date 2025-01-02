import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Dict, Iterator
from transformers import pipeline, CLIPTextModel, CLIPTokenizer
import open3d as o3d
from tqdm import tqdm
import traceback

class ModelGenerator(nn.Module):
    def __init__(self, config: Dict = None):
        """
        3D model üretimi için gerekli modelleri ve araçları hazırlar
        
        Args:
            config (Dict): Model konfigürasyonu
        """
        super().__init__()
        try:
            print("Model Generator başlatılıyor...")
            # Yapılandırmayı yükle
            self.config = config or {}
            print("Yapılandırma yüklendi.")
            
            # Device'ı ayarla
            if "training" in self.config and "device" in self.config["training"]:
                self.device = torch.device(self.config["training"]["device"])
            else:
                self.device = torch.device("cpu")
            print(f"Device: {self.device}")
            
            # CLIP model ve tokenizer'ı yükle
            print("CLIP modeli yükleniyor...")
            self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP modeli yüklendi.")
            
            # Kamera parametreleri
            print("Kamera parametreleri ayarlanıyor...")
            self.camera = self._setup_camera()
            print("Kamera parametreleri ayarlandı.")
            
            # Eğitim modu
            self.training = True
            
            print("Model Generator başarıyla başlatıldı.")
            
        except Exception as e:
            print(f"\nModel Generator başlatılırken hata oluştu: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise
    
    def train(self, mode: bool = True):
        """
        Modeli eğitim moduna alır
        
        Args:
            mode (bool): True ise eğitim modu, False ise değerlendirme modu
        """
        self.training = mode
        self.clip_model.train(mode)
        return self
    
    def eval(self):
        """
        Modeli değerlendirme moduna alır
        """
        return self.train(False)
    
    def state_dict(self):
        """
        Modelin durumunu döndürür
        """
        return {
            "clip_model": self.clip_model.state_dict(),
            "config": self.config
        }
    
    def load_state_dict(self, state_dict: Dict):
        """
        Modelin durumunu yükler
        
        Args:
            state_dict (Dict): Yüklenecek durum
        """
        self.clip_model.load_state_dict(state_dict["clip_model"])
        self.config = state_dict["config"]
    
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Eğitilebilir parametreleri döndürür
        
        Returns:
            Iterator[torch.nn.Parameter]: Eğitilebilir parametreler
        """
        try:
            return self.clip_model.parameters()
        except Exception as e:
            print(f"\nParametreler alınırken hata: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise
    
    def to(self, device: Union[str, torch.device]) -> 'ModelGenerator':
        """
        Modeli belirtilen cihaza taşır
        
        Args:
            device: Hedef cihaz
        
        Returns:
            ModelGenerator: Kendisi
        """
        try:
            if isinstance(device, str):
                device = torch.device(device)
            
            # CUDA kontrolü
            if device.type == "cuda" and not torch.cuda.is_available():
                print("CUDA kullanılamıyor, CPU'ya geçiliyor...")
                device = torch.device("cpu")
            
            self.device = device
            self.clip_model = self.clip_model.to(device)
            
            return self
            
        except Exception as e:
            print(f"\nModel cihaza taşınırken hata: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise
    
    def _setup_camera(self):
        """Kamera parametrelerini ayarlar"""
        try:
            # Kamera parametrelerini ayarla
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                width=256,
                height=256,
                fx=1000.0,
                fy=1000.0,
                cx=128.0,
                cy=128.0
            )
            
            # Kamera dış parametreleri
            extrinsic = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 3],
                [0, 0, 0, 1]
            ])
            
            return {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic
            }
            
        except Exception as e:
            print(f"\nKamera parametreleri ayarlanırken hata oluştu: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise
    
    def generate_from_text(self, text_prompt: str) -> Optional[torch.Tensor]:
        """
        Metin prompt'undan 3D model oluşturur
        
        Args:
            text_prompt (str): Model oluşturmak için kullanılacak metin
        
        Returns:
            Optional[torch.Tensor]: Oluşturulan 3D model
        """
        try:
            # CLIP ile metin embeddingi oluştur
            inputs = self.clip_tokenizer(
                text_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            text_embeddings = self.clip_model.get_text_features(**inputs)
            
            # Basit bir 3D nokta bulutu oluştur
            num_points = 1024
            points = torch.randn(num_points, 3).to(self.device)
            points = points / torch.norm(points, dim=1, keepdim=True)
            
            # Nokta bulutunu mesh'e dönüştür
            vertices, faces = self._point_cloud_to_mesh(points)
            
            # Mesh'i tensor'a dönüştür
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([1, 0, 0])
            
            return mesh
            
        except Exception as e:
            print(f"\nText-to-3D model oluşturma hatası: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            return None
    
    def generate_from_image(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """
        2D görüntüden 3D model oluşturur
        
        Args:
            image (torch.Tensor): İşlenmiş görüntü tensoru
        
        Returns:
            Optional[torch.Tensor]: Oluşturulan 3D model
        """
        try:
            # Görüntüyü normalize et
            image = (image - image.min()) / (image.max() - image.min())
            
            # Derinlik tahmini yap
            depth_map = self._estimate_depth(image)
            
            # Derinlik haritasından nokta bulutu oluştur
            point_cloud = self._depth_to_point_cloud(depth_map)
            
            # Nokta bulutunu mesh'e dönüştür
            vertices, faces = self._point_cloud_to_mesh(point_cloud)
            
            # Mesh'i tensor'a dönüştür
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([1, 0, 0])
            
            return mesh
            
        except Exception as e:
            print(f"\nImage-to-3D model oluşturma hatası: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            return None
    
    def _point_cloud_to_mesh(self, point_cloud: torch.Tensor) -> tuple:
        """Nokta bulutunu mesh'e dönüştürür"""
        try:
            # Poisson yüzey rekonstrüksiyonu
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
            
            # Normal vektörleri hesapla
            pcd.estimate_normals()
            
            # Mesh oluştur
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)[0]
            
            # Mesh'i tensor'a dönüştür
            vertices = torch.tensor(np.asarray(mesh.vertices), device=self.device)
            faces = torch.tensor(np.asarray(mesh.triangles), device=self.device)
            
            return vertices, faces
        except Exception as e:
            print(f"\nNokta bulutu mesh'e dönüştürülürken hata: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise
    
    def _estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        """Görüntüden derinlik tahmini yapar"""
        try:
            # MiDaS modelini kullan
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
            midas.to(self.device).eval()
            
            with torch.no_grad():
                depth = midas(image)
            
            return depth
        except Exception as e:
            print(f"\nDerinlik tahmini yapılırken hata: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise
    
    def _depth_to_point_cloud(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Derinlik haritasından nokta bulutu oluşturur"""
        try:
            height, width = depth_map.shape[-2:]
            
            # Piksel koordinatları oluştur
            y, x = torch.meshgrid(
                torch.arange(height, device=self.device),
                torch.arange(width, device=self.device)
            )
            
            # Normalize et
            x = (x - width/2) / width
            y = (y - height/2) / height
            
            # 3D koordinatlara dönüştür
            z = depth_map.squeeze()
            points = torch.stack([x, y, z], dim=-1)
            
            return points
        except Exception as e:
            print(f"\nDerinlik haritası nokta bulutuna dönüştürülürken hata: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise
    
    def save_model(self, mesh, filepath: str):
        """3D modeli dosyaya kaydeder"""
        try:
            import trimesh
            
            # Mesh'i trimesh formatına dönüştür
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            # Trimesh mesh'i oluştur
            mesh_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Dosyaya kaydet
            mesh_trimesh.export(filepath)
        except Exception as e:
            print(f"\nModel kaydedilirken hata: {str(e)}")
            print("\nHata detayları:")
            traceback.print_exc()
            raise