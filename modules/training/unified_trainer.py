import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from .config_manager import ConfigManager
from .losses import Otuken3DLoss, Metrics

class UnifiedTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Birleşik trainer sınıfı.
        Args:
            model: Eğitilecek model
            config: Eğitim konfigürasyonu
            config_manager: Konfigürasyon yöneticisi
        """
        self.model = model
        self.config_manager = config_manager or ConfigManager()
        self.config = config or self.config_manager.base_config
        self.device = next(model.parameters()).device
        
        self.setup_training()
        
    def setup_training(self):
        """Eğitim bileşenlerini hazırla"""
        # Optimizer
        self.optimizer = self.create_optimizer()
        
        # Scheduler
        self.scheduler = self.create_scheduler()
        
        # Loss function
        self.loss_fn = self.create_loss_fn()
        
        # Metrics
        self.metrics = Metrics()
        
        # Mixed precision
        self.use_amp = self.config['resources'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Optimizer oluştur"""
        optimizer_config = self.config['training']['optimizer']
        optimizer_class = getattr(torch.optim, optimizer_config['type'])
        
        return optimizer_class(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            **optimizer_config['params']
        )
        
    def create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Learning rate scheduler oluştur"""
        scheduler_config = self.config['training']['scheduler']
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config['type'])
        
        return scheduler_class(
            self.optimizer,
            **scheduler_config['params']
        )
        
    def create_loss_fn(self) -> nn.Module:
        """Loss function oluştur"""
        return Otuken3DLoss()
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Bir epoch eğitim yap"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc='Training', unit='batch') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Batch'i GPU'ya taşı
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(batch)
                    loss_dict = self.loss_fn(outputs, batch, return_components=True)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Log to wandb
                if self.config['logging']['wandb']['enabled']:
                    wandb.log({
                        'train_loss': loss.item(),
                        'train_avg_loss': avg_loss,
                        **{f'train_{k}': v.item() for k, v in loss_dict.items()}
                    })
        
        return {'loss': total_loss / num_batches}
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validasyon yap"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                # Batch'i GPU'ya taşı
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                loss_dict = self.loss_fn(outputs, batch, return_components=True)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
        
        avg_val_loss = total_loss / num_batches
        
        # Log validation metrics
        if self.config['logging']['wandb']['enabled']:
            wandb.log({'val_loss': avg_val_loss})
        
        return {'loss': avg_val_loss}
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ):
        """Tam eğitim döngüsü"""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
            
        # Checkpoint dizini
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint(checkpoint_dir / 'best_model.pt', epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['early_stopping_patience']:
                    print(f'Early stopping triggered after {epoch} epochs')
                    break
            
            # Regular checkpoint saving
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'checkpoint_epoch_{epoch}.pt',
                    epoch
                )
                
    def save_checkpoint(self, path: str, epoch: int):
        """Checkpoint kaydet"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(checkpoint, path)
        print(f'Checkpoint kaydedildi: {path}')
        
    def load_checkpoint(self, path: str) -> int:
        """Checkpoint yükle"""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch'] 