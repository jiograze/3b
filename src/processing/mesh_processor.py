"""
Mesh İşleme Modülü
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import trimesh

from src.core.base import BaseProcessor
from src.core.types import PathLike, Mesh, ProcessingResult

class MeshProcessor(BaseProcessor):
    """3B mesh'leri işlemek için sınıf"""
    
    SUPPORTED_FORMATS = {'.obj', '.stl', '.ply', '.glb', '.gltf', '.fbx', '.dae'}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: İşlemci yapılandırması
        """
        super().__init__(config)
        self.required_keys = ['max_vertices', 'max_faces']
        
    def optimize(self, input_path: PathLike, output_path: PathLike,
                target_faces: Optional[int] = None,
                preserve_uv: bool = True) -> None:
        """Mesh'i optimize eder
        
        Args:
            input_path: Girdi dosyası yolu
            output_path: Çıktı dosyası yolu
            target_faces: Hedef üçgen sayısı
            preserve_uv: UV koordinatları korunsun mu?
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        self.validate_file(input_path)
        self.ensure_dir(output_path.parent)
        
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {input_path.suffix}")
            
        try:
            # Mesh'i yükle
            mesh = trimesh.load(input_path)
            
            # Tek mesh mi yoksa sahne mi kontrol et
            if isinstance(mesh, trimesh.Scene):
                meshes = list(mesh.geometry.values())
                if len(meshes) == 1:
                    mesh = meshes[0]
                else:
                    # Birden fazla mesh varsa birleştir
                    mesh = trimesh.util.concatenate(meshes)
                    
            # UV koordinatlarını koru ve mesh'i optimize et
            if target_faces and len(mesh.faces) > target_faces:
                if preserve_uv and hasattr(mesh, 'visual'):
                    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                        # UV koordinatlarını sakla
                        uv = mesh.visual.uv
                        # Optimize et
                        mesh = mesh.simplify_quadratic_decimation(target_faces)
                        # UV koordinatlarını geri yükle
                        mesh.visual.uv = uv
                else:
                    # UV korunmayacaksa direkt optimize et
                    mesh = mesh.simplify_quadratic_decimation(target_faces)
                    
            # Optimize edilmiş mesh'i kaydet
            mesh.export(output_path)
            
            self.logger.info(f"Mesh optimize edildi: {input_path} -> {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Mesh optimizasyon hatası: {str(e)}")
            
    def repair(self, input_path: PathLike, output_path: PathLike,
               fix_normals: bool = True,
               remove_duplicates: bool = True) -> None:
        """Mesh'teki hataları onarır
        
        Args:
            input_path: Girdi dosyası yolu
            output_path: Çıktı dosyası yolu
            fix_normals: Normaller düzeltilsin mi?
            remove_duplicates: Tekrarlanan noktalar silinsin mi?
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        self.validate_file(input_path)
        self.ensure_dir(output_path.parent)
        
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {input_path.suffix}")
            
        try:
            # Mesh'i yükle
            mesh = trimesh.load(input_path)
            
            # Tek mesh mi yoksa sahne mi kontrol et
            if isinstance(mesh, trimesh.Scene):
                meshes = list(mesh.geometry.values())
                if len(meshes) == 1:
                    mesh = meshes[0]
                else:
                    # Birden fazla mesh varsa birleştir
                    mesh = trimesh.util.concatenate(meshes)
                    
            # Normalleri düzelt
            if fix_normals:
                mesh.fix_normals()
                
            # Tekrarlanan noktaları sil
            if remove_duplicates:
                mesh.remove_duplicate_faces()
                mesh.remove_degenerate_faces()
                mesh.remove_unreferenced_vertices()
                
            # Onarılmış mesh'i kaydet
            mesh.export(output_path)
            
            self.logger.info(f"Mesh onarıldı: {input_path} -> {output_path}")
            
        except Exception as e:
            raise RuntimeError(f"Mesh onarım hatası: {str(e)}")
            
    def batch_process(self, input_dir: PathLike, output_dir: PathLike,
                     target_faces: Optional[int] = None,
                     preserve_uv: bool = True,
                     fix_normals: bool = True,
                     remove_duplicates: bool = True) -> ProcessingResult:
        """Birden fazla mesh'i işler
        
        Args:
            input_dir: Girdi klasörü
            output_dir: Çıktı klasörü
            target_faces: Hedef üçgen sayısı
            preserve_uv: UV koordinatları korunsun mu?
            fix_normals: Normaller düzeltilsin mi?
            remove_duplicates: Tekrarlanan noktalar silinsin mi?
            
        Returns:
            İşlem sonuçları
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        self.validate_dir(input_dir)
        self.ensure_dir(output_dir)
        
        results = {
            'successful': [],
            'failed': []
        }
        
        # Desteklenen formattaki dosyaları bul
        for format_ in self.SUPPORTED_FORMATS:
            for input_path in input_dir.glob(f"*{format_}"):
                try:
                    output_path = output_dir / input_path.name
                    
                    # Önce optimize et
                    self.optimize(
                        input_path,
                        output_path,
                        target_faces=target_faces,
                        preserve_uv=preserve_uv
                    )
                    
                    # Sonra onar
                    self.repair(
                        output_path,
                        output_path,
                        fix_normals=fix_normals,
                        remove_duplicates=remove_duplicates
                    )
                    
                    results['successful'].append(str(input_path))
                except Exception as e:
                    self.logger.error(f"İşleme hatası ({input_path}): {str(e)}")
                    results['failed'].append({
                        'path': str(input_path),
                        'error': str(e)
                    })
                    
        return results
        
    def load_mesh(self, path: PathLike) -> Mesh:
        """Mesh'i yükler
        
        Args:
            path: Dosya yolu
            
        Returns:
            Mesh nesnesi
        """
        path = Path(path)
        self.validate_file(path)
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {path.suffix}")
            
        try:
            mesh = trimesh.load(path)
            
            # Tek mesh mi yoksa sahne mi kontrol et
            if isinstance(mesh, trimesh.Scene):
                meshes = list(mesh.geometry.values())
                if len(meshes) == 1:
                    mesh = meshes[0]
                else:
                    # Birden fazla mesh varsa birleştir
                    mesh = trimesh.util.concatenate(meshes)
                    
            return mesh
            
        except Exception as e:
            raise RuntimeError(f"Mesh yükleme hatası: {str(e)}")
            
    def save_mesh(self, mesh: Mesh, path: PathLike) -> None:
        """Mesh'i kaydeder
        
        Args:
            mesh: Mesh nesnesi
            path: Dosya yolu
        """
        path = Path(path)
        self.ensure_dir(path.parent)
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Desteklenmeyen format: {path.suffix}")
            
        try:
            mesh.export(path)
            self.logger.info(f"Mesh kaydedildi: {path}")
        except Exception as e:
            raise RuntimeError(f"Mesh kaydetme hatası: {str(e)}")
            
    def validate(self, mesh: Mesh) -> bool:
        """Mesh'in geçerliliğini kontrol eder
        
        Args:
            mesh: Mesh nesnesi
            
        Returns:
            Geçerliyse True
        """
        try:
            if not isinstance(mesh, trimesh.Trimesh):
                return False
                
            if not mesh.is_watertight:
                return False
                
            if not mesh.is_winding_consistent:
                return False
                
            return True
            
        except Exception:
            return False 