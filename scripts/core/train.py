import os
import sys
from datasets import load_dataset
import torch_directml  # AMD GPU desteği için
from torch.utils.data import DataLoader, ConcatDataset
import wandb
from tqdm import tqdm
from modules.model_generation.generator import ModelGenerator
from modules.data_management.dataset import ShapeDataset
from modules.utils.config import load_config
from modules.utils.losses import chamfer_distance, normal_consistency_loss
import open3d as o3d
from modules.data_management.dataset_loaders import (
    ShapeNetLoader,
    ModelNetLoader,
    Thingi10KLoader,
    HuggingFaceLoader
)

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from modules.model_generation.generator import ModelGenerator
from modules.data_management.dataset import ShapeDataset
from modules.utils.config import load_config
from modules.utils.losses import chamfer_distance, normal_consistency_loss
import open3d as o3d

# device ayarlama kısmını güncelleyelim
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch_directml.is_available():  # AMD GPU desteği
        return torch_directml.device()
    else:
        return torch.device("cpu")

def train():
    print("Eğitim başlatılıyor...")
    
    # Yapılandırmayı yükle
    config = load_config()
    print("Yapılandırma yüklendi.")
    
    # Güncellenmiş device ayarı
    device = get_device()
    print(f"Kullanılan cihaz: {device}")
    
    # Çoklu veri seti yükleme
    print("Veri setleri yükleniyor...")
    datasets = []
    
    if config["data"].get("use_shapenet", False):
        datasets.append(ShapeNetLoader(config["data"]["shapenet_path"]).load())
    
    if config["data"].get("use_modelnet", False):
        datasets.append(ModelNetLoader(config["data"]["modelnet_path"]).load())
    
    if config["data"].get("use_thingi10k", False):
        datasets.append(Thingi10KLoader(config["data"]["thingi10k_path"]).load())
    
    if config["data"].get("use_huggingface_datasets", False):
        hf_datasets = config["data"]["huggingface_datasets"]
        for dataset_name in hf_datasets:
            datasets.append(HuggingFaceLoader(dataset_name).load())

    # Veri setlerini birleştir
    combined_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    
    dataloader = DataLoader(combined_dataset,
                          batch_size=config["training"]["batch_size"],
                          shuffle=True,
                          num_workers=config["training"]["num_workers"],
                          collate_fn=combined_dataset.collate_fn if hasattr(combined_dataset, 'collate_fn') else None)
    print(f"Toplam {len(combined_dataset)} örnek yüklendi.")
    
    # Modeli oluştur
    print("Model oluşturuluyor...")
    model = ModelGenerator(config["model"])
    model = model.to(device)
    print("Model oluşturuldu.")
    
    # Optimizasyon ve kayıp fonksiyonlarını ayarla
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=config["training"]["scheduler_step_size"],
                                        gamma=config["training"]["scheduler_gamma"])
    
    # Wandb başlat
    if config["logging"]["use_wandb"]:
        print("Wandb başlatılıyor...")
        wandb.init(project=config["logging"]["project_name"],
                  name=config["logging"]["run_name"],
                  config=config)
        print("Wandb başlatıldı.")
    
    # Eğitim döngüsü
    print("Eğitim döngüsü başlıyor...")
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Veriyi cihaza taşı
                points = batch["points"].to(device)
                normals = batch["normals"].to(device)
                text = batch["text"]
                
                # İleri geçiş
                pred_points, pred_normals = model(text)
                
                # Kayıpları hesapla
                cd_loss = chamfer_distance(pred_points, points)
                normal_loss = normal_consistency_loss(pred_normals, normals)
                loss = cd_loss + config["training"]["normal_weight"] * normal_loss
                
                # Geriye yayılım
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # İstatistikleri güncelle
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Progress bar güncelle
                pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
                
                # Wandb günlüğü
                if config["logging"]["use_wandb"]:
                    wandb.log({
                        "loss": loss.item(),
                        "chamfer_distance": cd_loss.item(),
                        "normal_loss": normal_loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"]
                    })
        
        # Learning rate güncelle
        scheduler.step()
        
        # Epoch sonunda model kaydet
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            save_path = os.path.join(config["training"]["checkpoint_dir"], 
                                   f"model_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss
            }, save_path)
            print(f"Model kaydedildi: {save_path}")
            
            # Örnek çıktı üret
            if config["training"]["generate_samples"]:
                print("\nÖrnek model oluşturuluyor...")
                model.eval()
                with torch.no_grad():
                    sample_texts = [
                        "a simple cube",
                        "a pyramid with square base",
                        "a chair with four legs",
                        "a table with round top",
                        "a vase with narrow neck"
                    ]
                    
                    for text in sample_texts:
                        output_path = os.path.join(config["training"]["sample_dir"], 
                                                 f"sample_epoch_{epoch+1}_{text.replace(' ', '_')}.obj")
                        try:
                            model.generate(text, output_path)
                            print(f"Model oluşturuldu: {output_path}")
                        except Exception as e:
                            print(f"Model oluşturma hatası ({text}): {str(e)}")
                
                print("Örnek modeller oluşturuldu.")
    
    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    train() 