import os
import sys
import torch
import json
import re

# Proje kök dizinini Python yoluna ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Bağımlılıkları kontrol et
try:
    from scripts.setup import check_and_install_dependencies, setup_environment
    check_and_install_dependencies()
    setup_environment()
except ImportError:
    print("UYARI: Bağımlılık kontrol modülü bulunamadı.")
    print("Lütfen önce 'python scripts/setup.py' komutunu çalıştırın.")
    sys.exit(1)

from modules.model_generation.generator import ModelGenerator
from modules.utils.config import load_config

def get_epoch_from_checkpoint(filename):
    """Checkpoint dosya adından epoch numarasını çıkar"""
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def export_model(source_checkpoint: str, target_path: str, model_info: dict = None):
    """
    Eğitilmiş modeli Ötüken3D formatına aktar
    
    Args:
        source_checkpoint (str): Kaynak checkpoint dosyası
        target_path (str): Hedef model dosyası
        model_info (dict): Model meta verileri
    """
    print(f"Model aktarımı başlatılıyor...")
    print(f"Kaynak: {source_checkpoint}")
    
    # Yapılandırmayı yükle
    config = load_config()
    
    # Modeli oluştur
    model = ModelGenerator(config["model"])
    
    try:
        # Checkpoint'i yükle
        print("Checkpoint yükleniyor...")
        checkpoint = torch.load(source_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint başarıyla yüklendi!")
    except Exception as e:
        print(f"HATA: Checkpoint yüklenirken bir hata oluştu: {str(e)}")
        sys.exit(1)
    
    # Model meta verilerini hazırla
    if model_info is None:
        model_info = {
            "name": "Ötüken3D Text2Shape Model",
            "version": "1.0.0",
            "description": "Metinden 3D model oluşturma modeli",
            "architecture": {
                "name": "Text2Shape",
                "type": "generative",
                "input": "text",
                "output": "3d_model"
            },
            "training": {
                "epochs": checkpoint.get("epoch", 0),
                "final_loss": float(checkpoint.get("loss", 0.0)),
                "dataset": "ShapeNet",
                "optimizer": "Adam"
            },
            "performance": {
                "average_generation_time": "8-10 seconds",
                "supported_formats": ["obj", "stl", "ply", "off", "gltf"]
            },
            "requirements": {
                "python": ">=3.8",
                "torch": ">=2.0.0",
                "cuda": ">=11.0 (optional)"
            }
        }
    
    # Ötüken3D model formatını oluştur
    otuken3d_model = {
        "model_info": model_info,
        "model_state": model.state_dict(),
        "config": config["model"]
    }
    
    # Modeli kaydet
    print(f"Model kaydediliyor: {target_path}")
    torch.save(otuken3d_model, target_path)
    
    # Meta verileri ayrı kaydet
    meta_path = os.path.splitext(target_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)
    
    print("Model aktarımı tamamlandı!")
    print(f"Model dosyası: {target_path}")
    print(f"Meta veri dosyası: {meta_path}")

def main():
    # En son checkpoint'i bul
    config = load_config()
    checkpoint_dir = config["training"]["checkpoint_dir"]
    
    if not os.path.exists(checkpoint_dir):
        print(f"HATA: Checkpoint dizini bulunamadı: {checkpoint_dir}")
        sys.exit(1)
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    
    if not checkpoints:
        print("HATA: Checkpoint bulunamadı!")
        print(f"Dizin: {checkpoint_dir}")
        sys.exit(1)
    
    print("Bulunan checkpointler:")
    for cp in checkpoints:
        print(f"- {cp}")
    
    # En son checkpoint'i seç
    latest_checkpoint = max(checkpoints, key=get_epoch_from_checkpoint)
    source_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"\nSeçilen checkpoint: {latest_checkpoint}")
    
    # Hedef yolu belirle
    target_dir = os.path.join(project_root, "models", "otuken3d")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "text2shape_model.pt")
    
    # Modeli aktar
    export_model(source_path, target_path)

if __name__ == "__main__":
    main() 