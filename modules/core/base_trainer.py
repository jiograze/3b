from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import wandb
from tqdm import tqdm
from utils.logging.logger import get_logger
from utils.helpers.exceptions import TrainingError
from .base_model import BaseModel
from .base_dataset import BaseDataset

logger = get_logger('trainer')

class BaseTrainer(ABC):
    """Temel eğitim yöneticisi sınıfı"""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to_device(self.device)
        self._setup_training()
        
    def _setup_training(self) -> None:
        """Eğitim ayarlarını yapılandır"""
        try:
            # Optimizasyon ve kayıp fonksiyonlarını ayarla
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            self.criterion = self._create_criterion()
            
            # Checkpoint ve log dizinlerini oluştur
            self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
            self.log_dir = Path(self.config['training']['log_dir'])
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Wandb yapılandırması
            if self.config['logging']['use_wandb']:
                self._setup_wandb()
                
        except Exception as e:
            raise TrainingError(f"Eğitim ayarları yapılandırılamadı: {str(e)}")
            
    @abstractmethod
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Optimizer oluştur"""
        pass
        
    @abstractmethod
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Learning rate scheduler oluştur"""
        pass
        
    @abstractmethod
    def _create_criterion(self) -> Any:
        """Kayıp fonksiyonu oluştur"""
        pass
        
    def _setup_wandb(self) -> None:
        """Weights & Biases yapılandırması"""
        try:
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['run_name'],
                config=self.config
            )
            wandb.watch(self.model)
        except Exception as e:
            logger.warning(f"Wandb başlatılamadı: {str(e)}")
            
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Bir epoch eğitimi gerçekleştir"""
        self.model.train()
        epoch_metrics = {}
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Veriyi cihaza taşı
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    # İleri ve geri yayılım
                    self.optimizer.zero_grad()
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs, batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    # Metrikleri güncelle
                    batch_metrics = self._compute_metrics(outputs, batch)
                    for k, v in batch_metrics.items():
                        epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                        
                    # Progress bar güncelle
                    pbar.set_postfix({k: f"{v/(batch_idx+1):.4f}" 
                                    for k, v in epoch_metrics.items()})
                                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} işlenirken hata: {str(e)}")
                    continue
                    
        # Epoch metriklerini ortala
        epoch_metrics = {k: v/len(dataloader) for k, v in epoch_metrics.items()}
        return epoch_metrics
        
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validasyon gerçekleştir"""
        self.model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    try:
                        # Veriyi cihaza taşı
                        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                                for k, v in batch.items()}
                        
                        # İleri geçiş
                        outputs = self.model(**batch)
                        
                        # Metrikleri hesapla
                        batch_metrics = self._compute_metrics(outputs, batch)
                        for k, v in batch_metrics.items():
                            val_metrics[k] = val_metrics.get(k, 0) + v
                            
                        # Progress bar güncelle
                        pbar.set_postfix({k: f"{v/(batch_idx+1):.4f}" 
                                        for k, v in val_metrics.items()})
                                        
                    except Exception as e:
                        logger.error(f"Validation batch {batch_idx} işlenirken hata: {str(e)}")
                        continue
                        
        # Validasyon metriklerini ortala
        val_metrics = {k: f"{v/len(dataloader):.4f}" for k, v in val_metrics.items()}
        return val_metrics
        
    @abstractmethod
    def _compute_metrics(self, outputs: Any, batch: Dict[str, Any]) -> Dict[str, float]:
        """Metrikleri hesapla"""
        pass
        
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Checkpoint kaydet"""
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            extra_data = {
                'epoch': epoch,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'metrics': metrics
            }
            self.model.save_checkpoint(checkpoint_path, extra_data)
            
        except Exception as e:
            raise TrainingError(f"Checkpoint kaydedilemedi: {str(e)}")
            
    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, float]]:
        """Checkpoint yükle"""
        try:
            checkpoint = self.model.load_checkpoint(path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint['epoch'], checkpoint['metrics']
            
        except Exception as e:
            raise TrainingError(f"Checkpoint yüklenemedi: {str(e)}")
            
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Metrikleri logla"""
        # Konsol ve dosya loglaması
        log_str = f"Step {step} - " + " - ".join(f"{k}: {v}" for k, v in metrics.items())
        logger.info(log_str)
        
        # Wandb loglaması
        if self.config['logging']['use_wandb']:
            wandb.log(metrics, step=step)
            
    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Eğitimi gerçekleştir"""
        num_epochs = num_epochs or self.config['training']['num_epochs']
        history = {'train': [], 'val': []}
        
        try:
            for epoch in range(num_epochs):
                # Eğitim
                train_metrics = self.train_epoch(train_loader)
                history['train'].append(train_metrics)
                self.log_metrics(train_metrics, epoch)
                
                # Validasyon
                if val_loader:
                    val_metrics = self.validate(val_loader)
                    history['val'].append(val_metrics)
                    self.log_metrics(val_metrics, epoch)
                
                # Learning rate güncelle
                if self.scheduler:
                    self.scheduler.step()
                
                # Checkpoint kaydet
                if (epoch + 1) % self.config['training']['save_interval'] == 0:
                    self.save_checkpoint(epoch + 1, 
                                      val_metrics if val_loader else train_metrics)
                    
            return history
            
        except Exception as e:
            raise TrainingError(f"Eğitim sırasında hata: {str(e)}")
        finally:
            if self.config['logging']['use_wandb']:
                wandb.finish() 