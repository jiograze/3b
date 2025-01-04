import os
import torch
import yaml
from modules.training.trainer import Trainer
from modules.data.dataloader import create_dataloader
from models.otuken3d.model import Otuken3DModel

def load_config(config_path):
    """Konfigürasyon dosyasını yükle"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # Konfigürasyonu yükle
    config = load_config('modules/training/config.yaml')
    
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim cihazı: {device}")
    
    # Model oluştur
    model = Otuken3DModel(config)
    model = model.to(device)
    print("Model oluşturuldu")
    
    # Veri yükleyicileri oluştur
    train_loader = create_dataloader(
        data_dir=config['data']['train_data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        split='train'
    )
    
    val_loader = create_dataloader(
        data_dir=config['data']['val_data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        split='val'
    )
    print("Veri yükleyiciler hazır")
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        config=config,
        device=device
    )
    print("Trainer hazır")
    
    # Eğitimi başlat
    print("Eğitim başlıyor...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs']
    )
    print("Eğitim tamamlandı!")

if __name__ == '__main__':
    main() 