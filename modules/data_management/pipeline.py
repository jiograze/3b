"""Data processing pipelines for Ötüken3D."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import trimesh
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import h5py
from tqdm import tqdm

from ..core.base import BaseProcessor
from ..core.logger import setup_logger
from ..core.exceptions import DataError, ProcessingError
from ..core.constants import (
    MAX_POINTS,
    VOXEL_RESOLUTION,
    MESH_SIMPLIFICATION_TARGET
)

logger = setup_logger(__name__)

class DatasetProcessor(BaseProcessor):
    """Dataset processing and management."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config = config or {}
    
    def preprocess_mesh(
        self,
        mesh_path: Union[str, Path],
        target_points: int = MAX_POINTS,
        normalize: bool = True
    ) -> np.ndarray:
        """Preprocess 3D mesh file."""
        try:
            # Load mesh
            mesh = trimesh.load(mesh_path)
            
            # Sample points
            points = mesh.sample(target_points)
            
            if normalize:
                # Center and scale to unit sphere
                center = points.mean(axis=0)
                points = points - center
                scale = np.abs(points).max()
                points = points / scale
            
            return points
            
        except Exception as e:
            raise ProcessingError(f"Mesh preprocessing failed: {str(e)}")
    
    def preprocess_image(
        self,
        image_path: Union[str, Path],
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Preprocess image file."""
        try:
            # Load and resize image
            image = Image.open(image_path).convert("RGB")
            image = image.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize
            image = np.array(image) / 255.0
            
            return image
            
        except Exception as e:
            raise ProcessingError(f"Image preprocessing failed: {str(e)}")
    
    def create_voxel_grid(
        self,
        points: np.ndarray,
        resolution: int = VOXEL_RESOLUTION
    ) -> np.ndarray:
        """Convert point cloud to voxel grid."""
        try:
            # Create empty voxel grid
            voxels = np.zeros((resolution, resolution, resolution))
            
            # Scale points to voxel coordinates
            points = (points + 1) * (resolution - 1) / 2
            points = points.astype(int)
            
            # Fill voxels
            for point in points:
                if (0 <= point).all() and (point < resolution).all():
                    voxels[point[0], point[1], point[2]] = 1
            
            return voxels
            
        except Exception as e:
            raise ProcessingError(f"Voxel grid creation failed: {str(e)}")
    
    def simplify_mesh(
        self,
        mesh: trimesh.Trimesh,
        target_faces: int = MESH_SIMPLIFICATION_TARGET
    ) -> trimesh.Trimesh:
        """Simplify mesh to target number of faces."""
        try:
            if len(mesh.faces) > target_faces:
                mesh = mesh.simplify_quadratic_decimation(target_faces)
            return mesh
            
        except Exception as e:
            raise ProcessingError(f"Mesh simplification failed: {str(e)}")

class Otuken3DDataset(Dataset):
    """Custom dataset for Ötüken3D."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
        target_points: int = MAX_POINTS
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_points = target_points
        
        # Load split
        split_file = self.data_dir / f"{split}.json"
        if not split_file.exists():
            raise DataError(f"Split file not found: {split_file}")
        
        with open(split_file) as f:
            self.samples = json.load(f)
        
        # Setup processor
        self.processor = DatasetProcessor()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load mesh
            mesh_path = self.data_dir / "meshes" / sample["mesh_file"]
            points = self.processor.preprocess_mesh(
                mesh_path,
                target_points=self.target_points
            )
            
            # Create voxels
            voxels = self.processor.create_voxel_grid(points)
            
            # Load images if available
            images = []
            if "image_files" in sample:
                for img_file in sample["image_files"]:
                    img_path = self.data_dir / "images" / img_file
                    image = self.processor.preprocess_image(img_path)
                    images.append(image)
            
            # Create data dict
            data = {
                "points": torch.FloatTensor(points),
                "voxels": torch.FloatTensor(voxels),
                "text": sample.get("text", ""),
                "metadata": sample.get("metadata", {})
            }
            
            if images:
                data["images"] = torch.FloatTensor(np.stack(images))
            
            # Apply transforms
            if self.transform is not None:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return next valid sample
            return self.__getitem__((idx + 1) % len(self))

class DataPipeline:
    """Complete data processing pipeline."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        **kwargs
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
    
    def create_dataloaders(self) -> Dict[str, DataLoader]:
        """Create train/val/test dataloaders."""
        dataloaders = {}
        
        for split in ["train", "val", "test"]:
            dataset = Otuken3DDataset(
                self.data_dir,
                split=split,
                **self.kwargs
            )
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == "train"),
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        return dataloaders
    
    def process_dataset(
        self,
        output_file: Union[str, Path],
        splits: Optional[List[str]] = None
    ) -> None:
        """Process entire dataset and save to HDF5."""
        splits = splits or ["train", "val", "test"]
        output_file = Path(output_file)
        
        with h5py.File(output_file, "w") as f:
            for split in splits:
                logger.info(f"Processing {split} split...")
                
                dataset = Otuken3DDataset(
                    self.data_dir,
                    split=split,
                    **self.kwargs
                )
                
                # Create split group
                split_group = f.create_group(split)
                
                # Process samples
                for idx in tqdm(range(len(dataset)), desc=f"Processing {split}"):
                    try:
                        sample = dataset[idx]
                        sample_group = split_group.create_group(str(idx))
                        
                        # Save tensors
                        for key, value in sample.items():
                            if isinstance(value, torch.Tensor):
                                sample_group.create_dataset(
                                    key,
                                    data=value.numpy(),
                                    compression="gzip"
                                )
                            elif isinstance(value, (str, dict)):
                                sample_group.attrs[key] = json.dumps(value)
                    
                    except Exception as e:
                        logger.error(f"Error processing sample {idx}: {str(e)}")
                        continue
        
        logger.info(f"Dataset processed and saved to {output_file}") 