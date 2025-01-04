"""Evaluation metrics for 3D models."""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from utils.logging import setup_logger

logger = setup_logger(__name__)

def chamfer_distance(
    x: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    reduce_mean: bool = True,
    device: Optional[str] = None
) -> Union[float, torch.Tensor]:
    """İki nokta bulutu arasındaki Chamfer mesafesini hesapla
    
    Args:
        x: İlk nokta bulutu (N x 3)
        y: İkinci nokta bulutu (M x 3)
        reduce_mean: Ortalama al
        device: Hesaplama yapılacak cihaz (None ise CPU)
        
    Returns:
        Chamfer mesafesi
        
    Raises:
        ValueError: Geçersiz girdi boyutları veya tipleri için
    """
    # Girdi kontrolü
    if not isinstance(x, (np.ndarray, torch.Tensor)) or not isinstance(y, (np.ndarray, torch.Tensor)):
        raise ValueError("Girdiler numpy.ndarray veya torch.Tensor olmalı")
        
    # Boyut kontrolü
    if len(x.shape) != 2 or len(y.shape) != 2 or x.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError("Girdiler (N x 3) ve (M x 3) boyutlarında olmalı")
    
    # NumPy dizilerini PyTorch'a çevir
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32))
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y.astype(np.float32))
        
    # Veri tipini float yap
    x = x.float()
    y = y.float()
    
    # Cihaza taşı
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    elif x.is_cuda or y.is_cuda:
        # Eğer girdilerden biri GPU'daysa diğerini de GPU'ya taşı
        device = 'cuda'
        x = x.to(device)
        y = y.to(device)
    
    try:
        # Her x noktası için en yakın y noktasını bul
        xx = torch.sum(x ** 2, dim=1, keepdim=True)     # N x 1
        yy = torch.sum(y ** 2, dim=1)                   # M
        xy = torch.matmul(x, y.t())                     # N x M
        
        # Öklid mesafesi karesi: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2<x_i,y_j>
        dist = xx + yy - 2 * xy                         # N x M
        
        # En yakın noktaları bul
        dist_xy = torch.min(dist, dim=1)[0]             # N
        dist_yx = torch.min(dist, dim=0)[0]             # M
        
        # Chamfer mesafesi
        cd = torch.mean(dist_xy) + torch.mean(dist_yx) if reduce_mean else (dist_xy, dist_yx)
        
        return cd.cpu() if reduce_mean else (dist_xy.cpu(), dist_yx.cpu())
        
    except RuntimeError as e:
        logger.error(f"Chamfer mesafesi hesaplanırken hata: {str(e)}")
        raise

def edge_loss(points: torch.Tensor) -> torch.Tensor:
    """Kenar kaybı hesapla
    
    Args:
        points: Nokta bulutu (B x N x 3)
        
    Returns:
        Kenar kaybı
    """
    # Komşu noktalar arası mesafe
    diff = points[:, 1:] - points[:, :-1]           # B x (N-1) x 3
    edge_length = torch.norm(diff, dim=2)           # B x (N-1)
    
    # Ortalama kenar uzunluğu
    mean_length = torch.mean(edge_length, dim=1)    # B
    
    # Kenar uzunluklarının varyansı
    loss = torch.mean((edge_length - mean_length.unsqueeze(1)) ** 2)
    
    return loss

def laplacian_loss(points: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Laplacian kaybı hesapla
    
    Args:
        points: Nokta bulutu (B x N x 3)
        k: Komşu sayısı
        
    Returns:
        Laplacian kaybı
    """
    batch_size, num_points, _ = points.shape
    
    # Her nokta için en yakın k komşuyu bul
    dist = torch.cdist(points, points)              # B x N x N
    _, idx = torch.topk(dist, k=k+1, dim=2, largest=False)  # B x N x (k+1)
    idx = idx[:, :, 1:]                            # İlk indeks noktanın kendisi
    
    # Komşu noktaları al
    batch_idx = torch.arange(batch_size).view(-1, 1, 1).expand(-1, num_points, k)
    point_idx = torch.arange(num_points).view(1, -1, 1).expand(batch_size, -1, k)
    neighbors = points[batch_idx, idx]              # B x N x k x 3
    
    # Laplacian koordinatları
    mean_neighbors = torch.mean(neighbors, dim=2)   # B x N x 3
    laplacian = points - mean_neighbors
    
    # L2 normu
    loss = torch.mean(torch.sum(laplacian ** 2, dim=2))
    
    return loss

def compute_metrics(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    device: Optional[str] = None
) -> Dict[str, float]:
    """Tüm metrikleri hesapla
    
    Args:
        pred_points: Tahmin edilen nokta bulutu (B x N x 3)
        target_points: Hedef nokta bulutu (B x N x 3)
        device: Hesaplama yapılacak cihaz (None ise CPU)
        
    Returns:
        Metrik değerleri
    """
    metrics = {}
    
    # Chamfer mesafesi
    cd = chamfer_distance(pred_points, target_points, device=device)
    metrics['chamfer_distance'] = cd.item()
    
    # Kenar kaybı
    el = edge_loss(pred_points)
    metrics['edge_loss'] = el.item()
    
    # Laplacian kaybı
    ll = laplacian_loss(pred_points)
    metrics['laplacian_loss'] = ll.item()
    
    return metrics

class ModelEvaluator:
    """Model değerlendirme sınıfı"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Değerlendirme cihazı
        """
        self.device = device
        
    def evaluate_batch(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Batch değerlendir
        
        Args:
            model: Model
            batch: Veri batch'i
            
        Returns:
            Metrik değerleri
        """
        model.eval()
        
        with torch.no_grad():
            # Veriyi GPU'ya taşı
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # İleri geçiş
            outputs = model(batch)
            
            # Metrikleri hesapla
            metrics = compute_metrics(
                outputs['point_cloud'],
                batch['target_points'],
                device=self.device
            )
            
        return metrics
        
    def evaluate_loader(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Veri yükleyiciyi değerlendir
        
        Args:
            model: Model
            data_loader: Veri yükleyici
            
        Returns:
            Ortalama metrik değerleri
        """
        model.eval()
        
        total_metrics = {}
        num_batches = len(data_loader)
        
        with torch.no_grad():
            for batch in data_loader:
                # Batch değerlendir
                batch_metrics = self.evaluate_batch(model, batch)
                
                # Metrikleri topla
                for k, v in batch_metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
                    
        # Ortalama al
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return avg_metrics 