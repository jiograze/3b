"""Performance optimization module for Ötüken3D."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time
from dataclasses import dataclass
import redis
from sqlalchemy import create_engine, Index
from sqlalchemy.orm import sessionmaker
import mmap
import psutil

from ..core.logger import setup_logger
from ..core.exceptions import OptimizationError

logger = setup_logger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration."""
    redis_url: str = "redis://localhost:6379/0"
    max_memory: int = 1024  # MB
    eviction_policy: str = "allkeys-lru"
    ttl: int = 3600  # seconds

@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

class CacheManager:
    """Cache management system."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client = redis.Redis.from_url(
            config.redis_url,
            decode_responses=True
        )
        
        # Configure Redis
        self.redis_client.config_set("maxmemory", f"{config.max_memory}mb")
        self.redis_client.config_set("maxmemory-policy", config.eviction_policy)
    
    @lru_cache(maxsize=1000)
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get result from cache."""
        try:
            result = self.redis_client.get(key)
            return result if result else None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    def cache_result(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Cache result with optional TTL."""
        try:
            self.redis_client.set(
                key,
                value,
                ex=ttl or self.config.ttl
            )
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    def invalidate_cache(self, pattern: str = "*"):
        """Invalidate cache entries matching pattern."""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")

class DatabaseOptimizer:
    """Database optimization utility."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = create_engine(
            config.url,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout
        )
        self.Session = sessionmaker(bind=self.engine)
    
    def create_indexes(self, table_name: str, columns: List[str]):
        """Create database indexes."""
        try:
            for column in columns:
                Index(
                    f"idx_{table_name}_{column}",
                    f"{table_name}.{column}"
                ).create(self.engine)
        except Exception as e:
            raise OptimizationError(f"Failed to create index: {str(e)}")
    
    def optimize_query(self, query: str) -> str:
        """Optimize SQL query."""
        # Implement query optimization logic
        # Add EXPLAIN analysis
        # Add index hints
        return query
    
    def connection_pooling(self):
        """Configure connection pooling."""
        # Already configured in engine setup
        pass

class BatchProcessor:
    """Batch processing utility."""
    
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.queue = queue.Queue()
    
    async def process_batch(
        self,
        items: List[Any],
        process_func: callable
    ) -> List[Any]:
        """Process items in batches."""
        try:
            results = []
            for i in range(0, len(items), self.batch_size):
                batch = items[i:i + self.batch_size]
                batch_results = await asyncio.gather(
                    *[process_func(item) for item in batch]
                )
                results.extend(batch_results)
            return results
        except Exception as e:
            raise OptimizationError(f"Batch processing failed: {str(e)}")
    
    def add_to_queue(self, item: Any):
        """Add item to processing queue."""
        self.queue.put(item)
    
    def process_queue(self, process_func: callable):
        """Process items from queue."""
        while True:
            try:
                item = self.queue.get(timeout=1)
                self.executor.submit(process_func, item)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}")

class MemoryManager:
    """Memory management utility."""
    
    def __init__(self):
        self.memory_maps = {}
    
    def create_memory_map(
        self,
        name: str,
        size: int
    ) -> mmap.mmap:
        """Create memory mapped file."""
        try:
            mm = mmap.mmap(-1, size)
            self.memory_maps[name] = mm
            return mm
        except Exception as e:
            raise OptimizationError(f"Failed to create memory map: {str(e)}")
    
    def get_memory_map(self, name: str) -> Optional[mmap.mmap]:
        """Get memory mapped file."""
        return self.memory_maps.get(name)
    
    def release_memory_map(self, name: str):
        """Release memory mapped file."""
        if name in self.memory_maps:
            self.memory_maps[name].close()
            del self.memory_maps[name]
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor memory usage."""
        process = psutil.Process()
        return {
            "rss": process.memory_info().rss / 1024 / 1024,  # MB
            "vms": process.memory_info().vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }

class LazyLoader:
    """Lazy loading implementation."""
    
    def __init__(self):
        self._loaded_objects = {}
        self._load_lock = threading.Lock()
    
    def get_or_load(
        self,
        key: str,
        load_func: callable
    ) -> Any:
        """Get object, loading if necessary."""
        if key not in self._loaded_objects:
            with self._load_lock:
                if key not in self._loaded_objects:
                    self._loaded_objects[key] = load_func()
        return self._loaded_objects[key]
    
    def unload(self, key: str):
        """Unload object from memory."""
        if key in self._loaded_objects:
            del self._loaded_objects[key]

class PerformanceOptimizer:
    """Complete performance optimization system."""
    
    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        db_config: Optional[DatabaseConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        self.cache_manager = CacheManager(cache_config or CacheConfig())
        if db_config:
            self.db_optimizer = DatabaseOptimizer(db_config)
        self.batch_processor = BatchProcessor(batch_size, num_workers)
        self.memory_manager = MemoryManager()
        self.lazy_loader = LazyLoader()
    
    async def optimize_request(
        self,
        request_id: str,
        process_func: callable,
        *args,
        **kwargs
    ) -> Any:
        """Optimize request processing."""
        try:
            # Check cache
            cached_result = self.cache_manager.get_cached_result(request_id)
            if cached_result:
                return cached_result
            
            # Process request
            result = await process_func(*args, **kwargs)
            
            # Cache result
            self.cache_manager.cache_result(request_id, result)
            
            return result
            
        except Exception as e:
            raise OptimizationError(f"Request optimization failed: {str(e)}")
    
    async def batch_process(
        self,
        items: List[Any],
        process_func: callable
    ) -> List[Any]:
        """Process items in optimized batches."""
        return await self.batch_processor.process_batch(items, process_func)
    
    def optimize_memory_usage(
        self,
        data: Any,
        threshold_mb: float = 100
    ) -> Any:
        """Optimize memory usage for data."""
        try:
            memory_usage = self.memory_manager.monitor_memory()
            
            # If memory usage is high, use memory mapping
            if memory_usage["rss"] > threshold_mb:
                mm = self.memory_manager.create_memory_map(
                    f"data_{time.time()}",
                    len(data)
                )
                mm.write(data)
                return mm
            
            return data
            
        except Exception as e:
            raise OptimizationError(f"Memory optimization failed: {str(e)}")
    
    def lazy_load(
        self,
        key: str,
        load_func: callable
    ) -> Any:
        """Lazy load object."""
        return self.lazy_loader.get_or_load(key, load_func)

def create_optimizer(
    cache_config: Optional[CacheConfig] = None,
    db_config: Optional[DatabaseConfig] = None
) -> PerformanceOptimizer:
    """Create and configure performance optimizer."""
    try:
        optimizer = PerformanceOptimizer(
            cache_config=cache_config,
            db_config=db_config
        )
        return optimizer
        
    except Exception as e:
        raise OptimizationError(f"Failed to create optimizer: {str(e)}") 