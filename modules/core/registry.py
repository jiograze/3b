"""Registry system for models and processors."""

from typing import Dict, Type, Any
from .base import BaseModel, BaseProcessor
from .exceptions import ModelError

class Registry:
    """Base registry class."""
    
    def __init__(self):
        self._registry: Dict[str, Type[Any]] = {}
    
    def register(self, name: str, class_type: Type[Any]) -> None:
        """Register a new class."""
        if name in self._registry:
            raise ValueError(f"{name} is already registered")
        self._registry[name] = class_type
    
    def get(self, name: str) -> Type[Any]:
        """Get a registered class by name."""
        if name not in self._registry:
            raise KeyError(f"{name} is not registered")
        return self._registry[name]
    
    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

class ModelRegistry(Registry):
    """Registry for model classes."""
    
    def register(self, name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class."""
        if not issubclass(model_class, BaseModel):
            raise ModelError(f"{model_class.__name__} must inherit from BaseModel")
        super().register(name, model_class)

class ProcessorRegistry(Registry):
    """Registry for processor classes."""
    
    def register(self, name: str, processor_class: Type[BaseProcessor]) -> None:
        """Register a new processor class."""
        if not issubclass(processor_class, BaseProcessor):
            raise ModelError(f"{processor_class.__name__} must inherit from BaseProcessor")
        super().register(name, processor_class)

# Global registry instances
model_registry = ModelRegistry()
processor_registry = ProcessorRegistry() 