import pkg_resources
import sys
import logging

logging.basicConfig(
    filename='logs/setup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_dependencies():
    """Paket uyumluluklarını kontrol et"""
    required = {
        'torch': '>=1.8.0',
        'transformers': '>=4.0.0',
        'diffusers': '>=0.3.0',
        'numpy': '>=1.19.0',
        'pillow': '>=8.0.0'
    }
    
    try:
        for package, version in required.items():
            pkg_resources.require(f"{package}{version}")
            logging.info(f"{package} versiyon kontrolü başarılı")
    except pkg_resources.VersionConflict as e:
        logging.error(f"Paket versiyon uyumsuzluğu: {str(e)}")
        return False
    except pkg_resources.DistributionNotFound as e:
        logging.error(f"Paket bulunamadı: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1) 