"""Distributed training and model optimization module."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import horovod.torch as hvd
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path

from ..core.logger import setup_logger
from ..core.exceptions import TrainingError
from .trainer import BaseTrainer
from ..model3d_integration.text_to_shape import TextToShape
from ..model3d_integration.image_to_shape import ImageToShape

logger = setup_logger(__name__)

class DistributedTrainer(BaseTrainer):
    """Distributed training with multi-GPU support."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        backend: str = "nccl"
    ):
        super().__init__(model, config)
        
        self.backend = backend
        self.world_size = torch.cuda.device_count()
        
        # Initialize distributed environment
        if self.world_size > 1:
            dist.init_process_group(backend=backend)
            self.local_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            
            # Wrap model with DDP
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank]
            )
            
            logger.info(f"Distributed training initialized: {self.world_size} GPUs")
    
    def prepare_dataloader(self, dataloader):
        """Prepare dataloader for distributed training."""
        if self.world_size > 1:
            return torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size // self.world_size,
                shuffle=False,
                num_workers=dataloader.num_workers,
                sampler=DistributedSampler(dataloader.dataset)
            )
        return dataloader
    
    def cleanup(self):
        """Cleanup distributed environment."""
        if self.world_size > 1:
            dist.destroy_process_group()

class HorovodTrainer(BaseTrainer):
    """Distributed training with Horovod."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # Initialize Horovod
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        
        # Wrap optimizer with Horovod
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer,
            named_parameters=self.model.named_parameters()
        )
        
        # Broadcast parameters
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        
        logger.info(f"Horovod training initialized: {hvd.size()} workers")
    
    def prepare_dataloader(self, dataloader):
        """Prepare dataloader for Horovod."""
        return torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size // hvd.size(),
            shuffle=False,
            num_workers=dataloader.num_workers,
            sampler=torch.utils.data.distributed.DistributedSampler(
                dataloader.dataset,
                num_replicas=hvd.size(),
                rank=hvd.rank()
            )
        )

class ModelOptimizer:
    """Model optimization techniques."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def quantize_model(
        self,
        quantization_config: Dict[str, Any]
    ) -> nn.Module:
        """Quantize model to reduce size and improve inference speed."""
        try:
            # Dynamic quantization
            if quantization_config.get("type") == "dynamic":
                return torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear},
                    dtype=torch.qint8
                )
            
            # Static quantization
            elif quantization_config.get("type") == "static":
                model_fp32 = self.model
                model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model_fp32, inplace=True)
                torch.quantization.convert(model_fp32, inplace=True)
                return model_fp32
            
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_config.get('type')}")
            
        except Exception as e:
            raise TrainingError(f"Quantization failed: {str(e)}")
    
    def prune_model(
        self,
        pruning_config: Dict[str, Any]
    ) -> nn.Module:
        """Prune model to reduce size and improve efficiency."""
        try:
            # L1 unstructured pruning
            if pruning_config.get("type") == "l1_unstructured":
                parameters_to_prune = []
                for module in self.model.modules():
                    if isinstance(module, nn.Linear):
                        parameters_to_prune.append((module, 'weight'))
                
                torch.nn.utils.prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=torch.nn.utils.prune.L1Unstructured,
                    amount=pruning_config.get("amount", 0.2)
                )
            
            # Structured pruning
            elif pruning_config.get("type") == "structured":
                for module in self.model.modules():
                    if isinstance(module, nn.Linear):
                        torch.nn.utils.prune.ln_structured(
                            module,
                            name="weight",
                            amount=pruning_config.get("amount", 0.2),
                            n=2,
                            dim=0
                        )
            
            else:
                raise ValueError(f"Unsupported pruning type: {pruning_config.get('type')}")
            
            return self.model
            
        except Exception as e:
            raise TrainingError(f"Pruning failed: {str(e)}")
    
    def distill_knowledge(
        self,
        teacher_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        distillation_config: Dict[str, Any]
    ) -> nn.Module:
        """Knowledge distillation from teacher to student model."""
        try:
            temperature = distillation_config.get("temperature", 3.0)
            alpha = distillation_config.get("alpha", 0.1)
            
            criterion_kd = nn.KLDivLoss(reduction="batchmean")
            criterion_task = nn.MSELoss()
            
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=distillation_config.get("learning_rate", 1e-4)
            )
            
            # Training loop
            self.model.train()
            teacher_model.eval()
            
            for epoch in range(distillation_config.get("epochs", 10)):
                total_loss = 0
                
                for batch in dataloader:
                    optimizer.zero_grad()
                    
                    # Move batch to device
                    batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    
                    # Forward passes
                    with torch.no_grad():
                        teacher_output = teacher_model(batch)
                    student_output = self.model(batch)
                    
                    # Knowledge distillation loss
                    kd_loss = criterion_kd(
                        torch.log_softmax(student_output / temperature, dim=1),
                        torch.softmax(teacher_output / temperature, dim=1)
                    ) * (temperature * temperature)
                    
                    # Task-specific loss
                    task_loss = criterion_task(student_output, teacher_output)
                    
                    # Combined loss
                    loss = alpha * task_loss + (1 - alpha) * kd_loss
                    loss.backward()
                    
                    optimizer.step()
                    total_loss += loss.item()
                
                logger.info(f"Distillation Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
            
            return self.model
            
        except Exception as e:
            raise TrainingError(f"Knowledge distillation failed: {str(e)}")

def optimize_model_pipeline(
    model: nn.Module,
    optimization_config: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> nn.Module:
    """Complete model optimization pipeline."""
    try:
        optimizer = ModelOptimizer(model)
        
        # Apply optimizations in sequence
        if "quantization" in optimization_config:
            logger.info("Applying quantization...")
            model = optimizer.quantize_model(optimization_config["quantization"])
        
        if "pruning" in optimization_config:
            logger.info("Applying pruning...")
            model = optimizer.prune_model(optimization_config["pruning"])
        
        if "distillation" in optimization_config:
            logger.info("Applying knowledge distillation...")
            teacher_model = optimization_config["distillation"]["teacher_model"]
            dataloader = optimization_config["distillation"]["dataloader"]
            model = optimizer.distill_knowledge(
                teacher_model,
                dataloader,
                optimization_config["distillation"]
            )
        
        # Save optimized model
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "optimized_model.pth")
            logger.info(f"Optimized model saved to {output_dir}")
        
        return model
        
    except Exception as e:
        raise TrainingError(f"Model optimization pipeline failed: {str(e)}")

def create_distributed_trainer(
    model_type: str,
    config: Dict[str, Any],
    distributed_backend: str = "nccl"
) -> BaseTrainer:
    """Create appropriate distributed trainer based on configuration."""
    try:
        # Create base model
        if model_type == "text_to_shape":
            model = TextToShape(config=config)
        elif model_type == "image_to_shape":
            model = ImageToShape(config=config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create distributed trainer
        if distributed_backend == "horovod":
            return HorovodTrainer(model, config)
        else:
            return DistributedTrainer(model, config, backend=distributed_backend)
            
    except Exception as e:
        raise TrainingError(f"Failed to create distributed trainer: {str(e)}") 