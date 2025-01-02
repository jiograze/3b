import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from ..point_e.models.configs import MODEL_CONFIGS
from ..data_management.dataset_loaders import UnifiedDataset

class Otuken3DTrainer:
    def __init__(self, config_path='config/training_config.yaml'):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path)
        self.model = None
        self.optimizer = None
        
    def _load_config(self, config_path):
        """Eğitim konfigürasyonunu yükle"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Konfigürasyon yükleme hatası: {str(e)}")
            return None
            
    def prepare_model(self):
        """Point-E modelini eğitim için hazırla"""
        try:
            # Base modeli yükle
            base_config = MODEL_CONFIGS['base40M-textvec']
            self.model = base_config.create_model()
            
            # Ötüken3D için özelleştir
            self.model.add_cultural_embeddings(
                vocab_size=self.config['cultural_vocab_size'],
                embedding_dim=self.config['embedding_dim']
            )
            
            self.model = self.model.to(self.device)
            self.optimizer = Adam(
                self.model.parameters(),
                lr=self.config['learning_rate']
            )
            return True
        except Exception as e:
            self.logger.error(f"Model hazırlama hatası: {str(e)}")
            return False
            
    def train(self, num_epochs=100):
        """Modeli eğit"""
        try:
            # Veri setlerini hazırla
            dataset = UnifiedDataset(
                root_dir=self.config['dataset_path'],
                transforms=self.config['transforms']
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers']
            )
            
            # Eğitim döngüsü
            for epoch in range(num_epochs):
                total_loss = 0
                for batch in dataloader:
                    # Veriyi GPU'ya taşı
                    text, points = batch
                    text = text.to(self.device)
                    points = points.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(text)
                    loss = self.model.compute_loss(output, points)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                # Epoch sonuçlarını logla
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                
                # Checkpoint kaydet
                if (epoch + 1) % self.config['save_interval'] == 0:
                    self.save_checkpoint(f"otuken3d_epoch_{epoch+1}.pt")
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Eğitim hatası: {str(e)}")
            return False
            
    def save_checkpoint(self, filename):
        """Model checkpoint'ini kaydet"""
        try:
            checkpoint_dir = Path(self.config['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'config': self.config
            }
            
            torch.save(checkpoint, checkpoint_dir / filename)
            self.logger.info(f"Checkpoint kaydedildi: {filename}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint kaydetme hatası: {str(e)}") 