import shutil
import os
from pathlib import Path
import logging

logging.basicConfig(
    filename='logs/setup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_point_e():
    """Mevcut Point-E dosyalarını projeye entegre et"""
    source_dir = Path('/home/klc/Masaüstü/3b/point-e/point_e')
    target_dir = Path('modules/point_e')
    
    try:
        # Hedef dizin varsa temizle
        if target_dir.exists():
            shutil.rmtree(target_dir)
            
        # Point-E dosyalarını kopyala
        shutil.copytree(source_dir, target_dir)
        
        # __init__.py dosyasını oluştur
        init_file = target_dir / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            
        logging.info("Point-E dosyaları başarıyla kopyalandı")
        return True
        
    except Exception as e:
        logging.error(f"Point-E kurulum hatası: {str(e)}")
        return False

if __name__ == "__main__":
    if not setup_point_e():
        exit(1) 