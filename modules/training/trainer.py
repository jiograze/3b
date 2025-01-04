"""Training pipeline for Ötüken3D models."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
from typing import Dict, Any, Optional

class Trainer:
    """Model eğitim sınıfı"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        mixed_precision: bool = False,
        device: Optional[str] = None
    ):
        """
        Args:
            model: Eğitilecek model
            train_loader: Eğitim veri yükleyici
            val_loader: Doğrulama veri yükleyici
            learning_rate: Öğrenme oranı
            weight_decay: Ağırlık düşüşü
            mixed_precision: Karışık hassasiyet kullan
            device: Eğitim cihazı
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mixed_precision = mixed_precision
        
        # Optimizasyon ve kayıp fonksiyonlarını ayarla
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Mixed precision için scaler
        self.scaler = GradScaler() if mixed_precision else None
        
    def train_step(self, batch):
        """Tek eğitim adımı"""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Mixed precision training
        if self.mixed_precision:
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
        # Accuracy hesapla
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        
        return {
            'loss': loss.item(),
            'accuracy': correct / total * 100
        }
        
    def validate_step(self, batch):
        """Tek doğrulama adımı"""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Accuracy hesapla
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            total = targets.size(0)
            
        return {
            'loss': loss.item(),
            'accuracy': correct / total * 100
        }
        
    def train_epoch(self, epoch):
        """Tek epoch eğitim"""
        self.model.train()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False) as pbar:
            for batch_idx, batch in enumerate(pbar):
                self.optimizer.zero_grad()
                
                metrics = self.train_step(batch)
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                
                # Progress bar güncelle
                pbar.set_postfix({
                    'loss': metrics['loss'],
                    'acc': f"{metrics['accuracy']:.2f}%"
                })
                
        # Epoch ortalamaları
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Wandb'ye kaydet
        wandb.log({
            'train_loss': avg_loss,
            'train_accuracy': avg_accuracy,
            'epoch': epoch
        })
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
        
    def validate_epoch(self, epoch):
        """Tek epoch doğrulama"""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self.validate_step(batch)
                
                total_loss += metrics['loss']
                total_accuracy += metrics['accuracy']
                
        # Epoch ortalamaları
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Wandb'ye kaydet
        wandb.log({
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy,
            'epoch': epoch
        })
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
        
    def train(self, num_epochs, checkpoint_dir='checkpoints', save_every=5):
        """Modeli eğit"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Eğitim
            train_metrics = self.train_epoch(epoch)
            print(f'Epoch {epoch}: Train Loss: {train_metrics["loss"]:.4f}, '
                  f'Train Accuracy: {train_metrics["accuracy"]:.2f}%')
            
            # Doğrulama
            val_metrics = self.validate_epoch(epoch)
            print(f'Epoch {epoch}: Val Loss: {val_metrics["loss"]:.4f}, '
                  f'Val Accuracy: {val_metrics["accuracy"]:.2f}%')
            
            # En iyi modeli kaydet
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, 'best_model.pt'),
                    epoch=epoch,
                    metrics={
                        'train': train_metrics,
                        'val': val_metrics
                    }
                )
                
            # Periyodik kayıt
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt'),
                    epoch=epoch,
                    metrics={
                        'train': train_metrics,
                        'val': val_metrics
                    }
                )
                
    def save_checkpoint(self, path, epoch, metrics):
        """Model durumunu kaydet"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
        
    def load_checkpoint(self, path):
        """Model durumunu yükle"""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch']

if __name__ == "__main__":
    # Test trainer
    model = Otuken3DModel()
    train_loader = create_dataloader("data/3d_models", split="train")
    val_loader = create_dataloader("data/3d_models", split="val")
    
    if train_loader is not None:
        trainer = Trainer(model, train_loader, val_loader)
        trainer.train(num_epochs=1)  # Test için 1 epoch
    else:
        print("Veri yükleyici oluşturulamadı") 