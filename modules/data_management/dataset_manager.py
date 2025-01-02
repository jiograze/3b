import os
import requests
import zipfile
from tqdm import tqdm
from typing import Dict, List, Optional
import json

class DatasetManager:
    def __init__(self, base_path: str = "data/datasets"):
        """
        Veri seti yöneticisi
        
        Args:
            base_path (str): Veri seti ana dizini
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        # Veri seti bilgileri
        self.dataset_info = {
            "shapenet": {
                "name": "ShapeNet",
                "url": "https://shapenet.org/download/shapenetcore",
                "description": "3D model veri seti",
                "size": "30GB",
                "type": "3d",
                "priority": 1
            },
            "modelnet": {
                "name": "ModelNet",
                "url": "http://modelnet.cs.princeton.edu/",
                "description": "3D CAD model veri seti",
                "size": "2GB",
                "type": "3d",
                "priority": 2
            },
            "pix3d": {
                "name": "Pix3D",
                "url": "http://pix3d.csail.mit.edu/",
                "description": "3D model ve 2D görüntü eşleştirme veri seti",
                "size": "5GB",
                "type": "3d",
                "priority": 3
            }
        }
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """
        Veri setini indir
        
        Args:
            dataset_name (str): Veri seti adı
            force (bool): Varolan veri setini yeniden indir
            
        Returns:
            bool: İndirme başarılı mı
        """
        if dataset_name not in self.dataset_info:
            print(f"HATA: Veri seti bulunamadı: {dataset_name}")
            return False
        
        dataset = self.dataset_info[dataset_name]
        target_dir = os.path.join(self.base_path, dataset_name)
        
        # Veri seti zaten var mı kontrol et
        if os.path.exists(target_dir) and not force:
            print(f"Veri seti zaten mevcut: {target_dir}")
            return True
        
        # İndirme URL'sini al
        url = dataset["url"]
        if not url:
            print(f"HATA: İndirme URL'si bulunamadı: {dataset_name}")
            return False
        
        print(f"Veri seti indiriliyor: {dataset_name}")
        print(f"Boyut: {dataset['size']}")
        
        try:
            # İndirme işlemi
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Geçici dosya oluştur
            temp_file = os.path.join(self.base_path, f"{dataset_name}.zip")
            
            with open(temp_file, 'wb') as f, tqdm(
                desc=dataset_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            
            # ZIP dosyasını çıkart
            print("ZIP dosyası çıkartılıyor...")
            with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            
            # Geçici dosyayı sil
            os.remove(temp_file)
            
            print(f"Veri seti başarıyla indirildi: {target_dir}")
            return True
            
        except Exception as e:
            print(f"İndirme hatası: {str(e)}")
            return False
    
    def list_datasets(self) -> List[Dict]:
        """
        Mevcut veri setlerini listele
        
        Returns:
            List[Dict]: Veri seti bilgileri
        """
        datasets = []
        for name, info in self.dataset_info.items():
            path = os.path.join(self.base_path, name)
            info["installed"] = os.path.exists(path)
            datasets.append(info)
        
        # Önceliğe göre sırala
        datasets.sort(key=lambda x: x["priority"])
        return datasets
    
    def download_sample_dataset(self) -> bool:
        """
        Örnek veri seti indir
        
        Returns:
            bool: İndirme başarılı mı
        """
        # Örnek veri seti dizini
        sample_dir = os.path.join(self.base_path, "sample")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Örnek modeller oluştur
        models = [
            {
                "name": "cube",
                "description": "a simple cube",
                "vertices": [
                    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
                ],
                "faces": [
                    [0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7],
                    [0, 4, 1], [1, 4, 5], [2, 3, 6], [3, 7, 6],
                    [0, 2, 4], [2, 6, 4], [1, 5, 3], [3, 5, 7]
                ]
            },
            {
                "name": "pyramid",
                "description": "a simple pyramid",
                "vertices": [
                    [-1, -1, -1], [1, -1, -1], [0, -1, 1], [0, 1, 0]
                ],
                "faces": [
                    [0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]
                ]
            }
        ]
        
        # Metadata dosyası oluştur
        metadata = []
        
        # Her model için OBJ dosyası oluştur
        for model in models:
            obj_path = os.path.join(sample_dir, f"{model['name']}.obj")
            
            with open(obj_path, 'w') as f:
                # Vertices
                for v in model["vertices"]:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                
                # Faces (1-indexed)
                for face in model["faces"]:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            
            # Metadata'ya ekle
            metadata.append({
                "model_path": f"{model['name']}.obj",
                "description": model["description"],
                "split": "train"
            })
        
        # Metadata dosyasını kaydet
        with open(os.path.join(sample_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Örnek veri seti oluşturuldu: {sample_dir}")
        return True