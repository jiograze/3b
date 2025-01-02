"""Evaluation metrics for 3D models."""

import torch
import numpy as np
import trimesh
from typing import Dict, List, Union, Optional, Tuple
from scipy.spatial.distance import chamfer_distance
import open3d as o3d

from ..core.logger import setup_logger
from ..core.exceptions import ProcessingError

logger = setup_logger(__name__)

def compute_chamfer_distance(
    pred_points: np.ndarray,
    target_points: np.ndarray,
    normalize: bool = True
) -> float:
    """Compute Chamfer distance between two point clouds."""
    try:
        if normalize:
            # Normalize to unit sphere
            pred_center = pred_points.mean(axis=0)
            pred_points = pred_points - pred_center
            pred_scale = np.abs(pred_points).max()
            pred_points = pred_points / pred_scale
            
            target_center = target_points.mean(axis=0)
            target_points = target_points - target_center
            target_scale = np.abs(target_points).max()
            target_points = target_points / target_scale
        
        # Compute bidirectional Chamfer distance
        dist = chamfer_distance(pred_points, target_points)
        return float(dist)
        
    except Exception as e:
        raise ProcessingError(f"Chamfer distance computation failed: {str(e)}")

def compute_iou(
    pred_voxels: np.ndarray,
    target_voxels: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Compute Intersection over Union (IoU) between two voxel grids."""
    try:
        # Binarize voxels
        pred_binary = (pred_voxels > threshold).astype(np.float32)
        target_binary = (target_voxels > threshold).astype(np.float32)
        
        # Compute intersection and union
        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        return float(iou)
        
    except Exception as e:
        raise ProcessingError(f"IoU computation failed: {str(e)}")

def compute_fscore(
    pred_points: np.ndarray,
    target_points: np.ndarray,
    threshold: float = 0.01
) -> Tuple[float, float, float]:
    """Compute F-score between two point clouds."""
    try:
        # Convert to Open3D point clouds
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        # Compute distances
        dist1 = np.asarray(pred_pcd.compute_point_cloud_distance(target_pcd))
        dist2 = np.asarray(target_pcd.compute_point_cloud_distance(pred_pcd))
        
        # Compute precision and recall
        precision = (dist1 < threshold).mean()
        recall = (dist2 < threshold).mean()
        
        # Compute F-score
        f_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        return float(f_score), float(precision), float(recall)
        
    except Exception as e:
        raise ProcessingError(f"F-score computation failed: {str(e)}")

def compute_surface_metrics(
    pred_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    num_samples: int = 10000
) -> Dict[str, float]:
    """Compute surface-based metrics between two meshes."""
    try:
        # Sample points from meshes
        pred_points = pred_mesh.sample(num_samples)
        target_points = target_mesh.sample(num_samples)
        
        # Compute metrics
        chamfer_dist = compute_chamfer_distance(pred_points, target_points)
        f_score, precision, recall = compute_fscore(pred_points, target_points)
        
        # Compute additional mesh statistics
        pred_volume = pred_mesh.volume
        target_volume = target_mesh.volume
        volume_diff = abs(pred_volume - target_volume) / (target_volume + 1e-6)
        
        pred_area = pred_mesh.area
        target_area = target_mesh.area
        area_diff = abs(pred_area - target_area) / (target_area + 1e-6)
        
        return {
            "chamfer_distance": chamfer_dist,
            "f_score": f_score,
            "precision": precision,
            "recall": recall,
            "volume_difference": float(volume_diff),
            "surface_area_difference": float(area_diff)
        }
        
    except Exception as e:
        raise ProcessingError(f"Surface metrics computation failed: {str(e)}")

class ModelEvaluator:
    """Model evaluation class."""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or [
            "chamfer_distance",
            "iou",
            "f_score",
            "surface_metrics"
        ]
    
    def evaluate_single(
        self,
        prediction: Dict[str, Union[np.ndarray, trimesh.Trimesh]],
        target: Dict[str, Union[np.ndarray, trimesh.Trimesh]]
    ) -> Dict[str, float]:
        """Evaluate a single prediction."""
        results = {}
        
        try:
            if "chamfer_distance" in self.metrics and "points" in prediction and "points" in target:
                results["chamfer_distance"] = compute_chamfer_distance(
                    prediction["points"],
                    target["points"]
                )
            
            if "iou" in self.metrics and "voxels" in prediction and "voxels" in target:
                results["iou"] = compute_iou(
                    prediction["voxels"],
                    target["voxels"]
                )
            
            if "f_score" in self.metrics and "points" in prediction and "points" in target:
                f_score, precision, recall = compute_fscore(
                    prediction["points"],
                    target["points"]
                )
                results.update({
                    "f_score": f_score,
                    "precision": precision,
                    "recall": recall
                })
            
            if "surface_metrics" in self.metrics and "mesh" in prediction and "mesh" in target:
                surface_metrics = compute_surface_metrics(
                    prediction["mesh"],
                    target["mesh"]
                )
                results.update(surface_metrics)
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {metric: float("nan") for metric in self.metrics}
    
    def evaluate_batch(
        self,
        predictions: List[Dict[str, Union[np.ndarray, trimesh.Trimesh]]],
        targets: List[Dict[str, Union[np.ndarray, trimesh.Trimesh]]]
    ) -> Dict[str, List[float]]:
        """Evaluate a batch of predictions."""
        batch_results = {metric: [] for metric in self.metrics}
        
        for pred, target in zip(predictions, targets):
            single_results = self.evaluate_single(pred, target)
            for metric, value in single_results.items():
                batch_results[metric].append(value)
        
        return batch_results
    
    def compute_statistics(
        self,
        results: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute statistics for evaluation results."""
        stats = {}
        
        for metric, values in results.items():
            values = np.array(values)
            values = values[~np.isnan(values)]  # Remove NaN values
            
            if len(values) > 0:
                stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
            else:
                stats[metric] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "min": float("nan"),
                    "max": float("nan")
                }
        
        return stats 