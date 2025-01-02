"""Training pipeline for Ötüken3D models."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-4,
        weight_decay=0.01,
        mixed_precision=False,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mixed_precision = mixed_precision
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        
        # Mixed precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # Metrics
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}', unit='batch') as pbar:
            for batch_idx, voxels in enumerate(pbar):
                voxels = voxels.to(self.device)
                
                # Mixed precision training
                if self.mixed_precision:
                    with autocast():
                        reconstructed, _ = self.model(voxels)
                        loss = self.reconstruction_loss(reconstructed, voxels)
                        
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    reconstructed, _ = self.model(voxels)
                    loss = self.reconstruction_loss(reconstructed, voxels)
                    
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Log to wandb
                wandb.log({
                    'train_loss': loss.item(),
                    'train_avg_loss': avg_loss,
                    'epoch': epoch,
                    'batch': batch_idx
                })
        
        return total_loss / num_batches
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for voxels in self.val_loader:
                voxels = voxels.to(self.device)
                
                reconstructed, _ = self.model(voxels)
                loss = self.reconstruction_loss(reconstructed, voxels)
                
                total_loss += loss.item()
        
        avg_val_loss = total_loss / num_batches
        
        # Log validation metrics
        wandb.log({
            'val_loss': avg_val_loss,
            'epoch': epoch
        })
        
        return avg_val_loss
    
    def train(self, num_epochs, checkpoint_dir='checkpoints', save_every=5):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate(epoch)
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'),
                    epoch,
                    train_loss,
                    val_loss
                )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, 'best_model.pt'),
                    epoch,
                    train_loss,
                    val_loss
                )
    
    def save_checkpoint(self, path, epoch, train_loss, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        print(f'Checkpoint kaydedildi: {path}')
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.best_val_loss = checkpoint['best_val_loss']
        
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