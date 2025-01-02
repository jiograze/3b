import os
import argparse
import torch
import wandb

from modules.training.otuken3d_model import Otuken3DModel
from modules.training.data_loader import create_dataloader
from modules.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Ötüken3D model eğitimi")
    
    # Veri yolu argümanları
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Eğitim verilerinin bulunduğu dizin")
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="Çıktı dizini")
                      
    # Model argümanları
    parser.add_argument("--voxel_size", type=int, default=128,
                      help="Voxel grid boyutu")
    parser.add_argument("--latent_dim", type=int, default=768,
                      help="Latent uzay boyutu")
                      
    # Eğitim argümanları
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch boyutu")
    parser.add_argument("--num_epochs", type=int, default=100,
                      help="Epoch sayısı")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Öğrenme oranı")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="DataLoader worker sayısı")
    parser.add_argument("--mixed_precision", action="store_true",
                      help="Mixed precision training kullan")
    parser.add_argument("--save_every", type=int, default=5,
                      help="Kaç epoch'ta bir checkpoint kaydedileceği")
                      
    # Checkpoint argümanları
    parser.add_argument("--resume_from", type=str,
                      help="Eğitimi devam ettirmek için checkpoint dosyası")
                      
    args = parser.parse_args()
    
    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Weights & Biases başlat
    wandb.init(
        project="otuken3d",
        config=vars(args),
        name=f"otuken3d_v{args.voxel_size}"
    )
    
    # Device seç
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Model oluştur
    model = Otuken3DModel(
        voxel_size=args.voxel_size,
        latent_dim=args.latent_dim,
        device=device
    )
    
    # Veri yükleyicileri oluştur
    train_loader = create_dataloader(
        args.data_dir,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    val_loader = create_dataloader(
        args.data_dir,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Trainer oluştur
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision,
        device=device
    )
    
    # Checkpoint'ten devam et
    start_epoch = 0
    if args.resume_from:
        start_epoch = trainer.load_checkpoint(args.resume_from)
        print(f"Checkpoint yüklendi: {args.resume_from}")
        print(f"Eğitim epoch {start_epoch}'tan devam ediyor")
    
    # Eğitimi başlat
    trainer.train(
        num_epochs=args.num_epochs,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        save_every=args.save_every
    )
    
    wandb.finish()

if __name__ == "__main__":
    main() 