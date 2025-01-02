import torch
import torch.nn.functional as F

def chamfer_distance(pred_points, target_points):
    """
    Chamfer mesafesi hesapla
    
    Args:
        pred_points (torch.Tensor): Tahmin edilen nokta bulutu (B, N, 3)
        target_points (torch.Tensor): Hedef nokta bulutu (B, M, 3)
        
    Returns:
        torch.Tensor: Chamfer mesafesi
    """
    # Nokta bulutları arasındaki mesafeleri hesapla
    pred_expanded = pred_points.unsqueeze(2)  # (B, N, 1, 3)
    target_expanded = target_points.unsqueeze(1)  # (B, 1, M, 3)
    
    # Euclidean mesafesi hesapla
    distances = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)  # (B, N, M)
    
    # Her yöndeki minimum mesafeleri bul
    pred_to_target = torch.min(distances, dim=2)[0]  # (B, N)
    target_to_pred = torch.min(distances, dim=1)[0]  # (B, M)
    
    # Ortalama mesafeyi hesapla
    chamfer_dist = torch.mean(pred_to_target) + torch.mean(target_to_pred)
    
    return chamfer_dist

def normal_consistency_loss(pred_normals, target_normals):
    """
    Normal tutarlılık kaybı hesapla
    
    Args:
        pred_normals (torch.Tensor): Tahmin edilen normaller (B, N, 3)
        target_normals (torch.Tensor): Hedef normaller (B, N, 3)
        
    Returns:
        torch.Tensor: Normal tutarlılık kaybı
    """
    # Normalleri normalize et
    pred_normals = F.normalize(pred_normals, dim=-1)
    target_normals = F.normalize(target_normals, dim=-1)
    
    # Nokta başına normal tutarlılık kaybı
    consistency = 1 - torch.sum(pred_normals * target_normals, dim=-1)  # (B, N)
    
    # Ortalama kaybı hesapla
    loss = torch.mean(consistency)
    
    return loss

def edge_length_regularization(points, faces):
    """
    Kenar uzunluğu düzenlileştirme kaybı
    
    Args:
        points (torch.Tensor): Nokta koordinatları (B, N, 3)
        faces (torch.Tensor): Üçgen yüzey indeksleri (B, F, 3)
        
    Returns:
        torch.Tensor: Kenar uzunluğu düzenlileştirme kaybı
    """
    # Yüz köşelerini al
    v1 = torch.gather(points, 1, faces[:, :, 0:1].expand(-1, -1, 3))  # (B, F, 3)
    v2 = torch.gather(points, 1, faces[:, :, 1:2].expand(-1, -1, 3))  # (B, F, 3)
    v3 = torch.gather(points, 1, faces[:, :, 2:3].expand(-1, -1, 3))  # (B, F, 3)
    
    # Kenar vektörlerini hesapla
    e1 = v2 - v1  # (B, F, 3)
    e2 = v3 - v2  # (B, F, 3)
    e3 = v1 - v3  # (B, F, 3)
    
    # Kenar uzunluklarını hesapla
    l1 = torch.sum(e1 ** 2, dim=-1)  # (B, F)
    l2 = torch.sum(e2 ** 2, dim=-1)  # (B, F)
    l3 = torch.sum(e3 ** 2, dim=-1)  # (B, F)
    
    # Ortalama kenar uzunluğu
    mean_length = (torch.mean(l1) + torch.mean(l2) + torch.mean(l3)) / 3
    
    # Varyans hesapla
    var1 = torch.mean((l1 - mean_length) ** 2)
    var2 = torch.mean((l2 - mean_length) ** 2)
    var3 = torch.mean((l3 - mean_length) ** 2)
    
    # Toplam varyans
    total_variance = (var1 + var2 + var3) / 3
    
    return total_variance

def laplacian_loss(points, faces):
    """
    Laplacian düzgünleştirme kaybı
    
    Args:
        points (torch.Tensor): Nokta koordinatları (B, N, 3)
        faces (torch.Tensor): Üçgen yüzey indeksleri (B, F, 3)
        
    Returns:
        torch.Tensor: Laplacian düzgünleştirme kaybı
    """
    # Komşuluk matrisini oluştur
    batch_size = points.size(0)
    num_points = points.size(1)
    device = points.device
    
    # Boş komşuluk matrisi
    adj = torch.zeros((batch_size, num_points, num_points), device=device)
    
    # Yüzlerden komşulukları belirle
    for b in range(batch_size):
        for f in faces[b]:
            i, j, k = f
            adj[b, i, j] = adj[b, j, i] = 1
            adj[b, j, k] = adj[b, k, j] = 1
            adj[b, k, i] = adj[b, i, k] = 1
    
    # Derece matrisini hesapla
    degree = torch.sum(adj, dim=-1, keepdim=True)  # (B, N, 1)
    
    # Normalize edilmiş Laplacian matrisi
    lap = -adj / torch.clamp(degree, min=1)  # (B, N, N)
    lap = lap + torch.eye(num_points, device=device).unsqueeze(0)  # Add identity
    
    # Laplacian koordinatları
    lap_points = torch.bmm(lap, points)  # (B, N, 3)
    
    # L2 normu
    loss = torch.mean(torch.sum(lap_points ** 2, dim=-1))
    
    return loss 