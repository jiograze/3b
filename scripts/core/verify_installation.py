import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    filename='logs/setup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def verify_installation():
    """Kurulum sonrası gerekli dosya ve dizinleri kontrol et"""
    required_dirs = [
        'data/datasets',
        'models/pretrained',
        'modules/core',
        'utils/config',
        'logs'
    ]
    
    required_files = [
        'config/config.yaml',
        'requirements.txt'
    ]
    
    # Dizin kontrolü
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            logging.error(f"Gerekli dizin bulunamadı: {dir_path}")
            return False
            
    # Dosya kontrolü
    for file_path in required_files:
        if not os.path.isfile(file_path):
            logging.error(f"Gerekli dosya bulunamadı: {file_path}")
            return False
            
    # Model dosyaları kontrolü
    model_files = list(Path('models/pretrained').glob('**/*.bin'))
    if not model_files:
        logging.warning("Hiçbir model dosyası bulunamadı!")
        return False
        
    logging.info("Kurulum doğrulama başarılı")
    return True

if __name__ == "__main__":
    if not verify_installation():
        sys.exit(1) 