import torch
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional, Union, Type

class OptimizerManager:
    """Optimizasyon yöneticisi"""
    
    OPTIMIZER_MAPPING = {
        'adam': Adam,
        'adamw': AdamW
    }
    
    SCHEDULER_MAPPING = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'one_cycle': torch.optim.lr_scheduler.OneCycleLR,
        'cosine_warmup': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    }
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any]
    ):
        """
        Args:
            model: Optimize edilecek model
            config: Optimizasyon konfigürasyonu
        """
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
    def _create_optimizer(self) -> Optimizer:
        """Optimizer oluştur"""
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config['type'].lower()
        
        if optimizer_type not in self.OPTIMIZER_MAPPING:
            raise ValueError(f"Desteklenmeyen optimizer tipi: {optimizer_type}")
            
        optimizer_class = self.OPTIMIZER_MAPPING[optimizer_type]
        
        # Parametre grupları oluştur
        param_groups = self._create_param_groups()
        
        return optimizer_class(
            param_groups,
            lr=self.config['training']['learning_rate'],
            **optimizer_config.get('params', {})
        )
        
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Scheduler oluştur"""
        scheduler_config = self.config['training'].get('scheduler')
        if not scheduler_config:
            return None
            
        scheduler_type = scheduler_config['type'].lower()
        if scheduler_type not in self.SCHEDULER_MAPPING:
            raise ValueError(f"Desteklenmeyen scheduler tipi: {scheduler_type}")
            
        scheduler_class = self.SCHEDULER_MAPPING[scheduler_type]
        
        return scheduler_class(
            self.optimizer,
            **scheduler_config.get('params', {})
        )
        
    def _create_param_groups(self) -> List[Dict[str, Any]]:
        """Parametre grupları oluştur"""
        # Varsayılan grup
        default_group = {
            'params': [],
            'weight_decay': self.config['training'].get('weight_decay', 0.0)
        }
        
        # Özel gruplar
        special_groups = []
        param_group_config = self.config['training'].get('param_groups', {})
        
        for name, param in self.model.named_parameters():
            # Özel grup kontrolü
            assigned = False
            for group_name, group_config in param_group_config.items():
                if any(pattern in name for pattern in group_config.get('patterns', [])):
                    special_groups.append({
                        'params': [param],
                        'weight_decay': group_config.get('weight_decay', 0.0),
                        'lr': group_config.get('learning_rate', self.config['training']['learning_rate'])
                    })
                    assigned = True
                    break
                    
            # Varsayılan gruba ekle
            if not assigned:
                default_group['params'].append(param)
                
        return [default_group] + special_groups
        
    def step_optimizer(self, loss: torch.Tensor = None):
        """Optimizer adımı"""
        self.optimizer.step()
        
    def step_scheduler(self, metric: float = None):
        """Scheduler adımı"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
                
    def zero_grad(self):
        """Gradyanları sıfırla"""
        self.optimizer.zero_grad()
        
    def get_lr(self) -> float:
        """Güncel learning rate"""
        return self.optimizer.param_groups[0]['lr']
        
    def state_dict(self) -> Dict[str, Any]:
        """Durum sözlüğü"""
        state = {
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Durum sözlüğünü yükle"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler']) 