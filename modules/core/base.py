"""Base classes for Ötüken3D models and processors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import torch
import numpy as np

from .logger import setup_logger
from .exceptions import ModelError, ProcessingError

logger = setup_logger(__name__)

class BaseModel(ABC):
    """Base class for all models in Ötüken3D."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = {}
        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from path."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights to path."""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions using the model."""
        pass
    
    def to(self, device: torch.device) -> 'BaseModel':
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self

class BaseProcessor(ABC):
    """Base class for all data processors in Ötüken3D."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data."""
        pass
    
    @abstractmethod
    def postprocess(self, data: Any) -> Any:
        """Postprocess model output."""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate input data."""
        return True

class BaseOptimizer:
    """Base class for model optimizers."""
    
    def __init__(self, model: BaseModel, **kwargs):
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer(**kwargs)
    
    def _setup_optimizer(self, **kwargs):
        """Setup optimizer and scheduler."""
        pass
    
    def step(self):
        """Perform optimization step."""
        if self.optimizer is None:
            raise ModelError("Optimizer not initialized")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
    
    def zero_grad(self):
        """Zero gradients."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

class BasePipeline:
    """Base class for processing pipelines."""
    
    def __init__(self, model: BaseModel, processor: BaseProcessor):
        self.model = model
        self.processor = processor
        logger.info(f"Initializing {self.__class__.__name__}")
    
    def __call__(self, input_data: Any) -> Any:
        """Run the complete pipeline."""
        try:
            processed_input = self.processor.preprocess(input_data)
            model_output = self.model.predict(processed_input)
            final_output = self.processor.postprocess(model_output)
            return final_output
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            raise ProcessingError(f"Pipeline execution failed: {str(e)}") 