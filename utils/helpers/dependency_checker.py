import subprocess
import sys
from typing import List, Dict
import importlib
import pkg_resources

class DependencyChecker:
    REQUIRED_PACKAGES = {
        'yaml': 'pyyaml',
        'torch': 'torch>=2.0.0',
        'torchvision': 'torchvision>=0.15.0',
        'transformers': 'transformers>=4.11.0',
        'wandb': 'wandb>=0.12.0',
        'open3d': 'open3d>=0.13.0',
        'trimesh': 'trimesh>=3.9.0',
        'numpy': 'numpy>=1.21.0',
        'scipy': 'scipy>=1.7.0',
        'tqdm': 'tqdm>=4.62.0',
        'matplotlib': 'matplotlib>=3.4.0'
    }

    @staticmethod
    def check_and_install_packages():
        """Eksik paketleri kontrol eder ve yükler"""
        missing_packages = []
        
        print("Gerekli paketler kontrol ediliyor...")
        
        for module_name, package_name in DependencyChecker.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(module_name)
                print(f"✓ {module_name} mevcut")
            except ImportError:
                print(f"✗ {module_name} eksik")
                missing_packages.append(package_name)
        
        if missing_packages:
            print("\nEksik paketler yükleniyor...")
            try:
                # ROCm için PyTorch kurulumu
                if 'torch' in missing_packages:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "--extra-index-url", "https://download.pytorch.org/whl/rocm5.4.2",
                        "torch>=2.0.0", "torchvision>=0.15.0"
                    ])
                    missing_packages = [p for p in missing_packages if not p.startswith('torch')]
                
                if missing_packages:
                    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("\nTüm eksik paketler başarıyla yüklendi!")
            except subprocess.CalledProcessError as e:
                print(f"\nPaket yükleme hatası: {str(e)}")
                sys.exit(1)
        else:
            print("\nTüm gerekli paketler mevcut.")

    @staticmethod
    def check_gpu():
        """GPU kullanılabilirliğini kontrol eder"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                print(f"\nNVIDIA GPU kullanılabilir: {device_count} GPU bulundu")
                print(f"GPU: {device_name}")
                return True
            else:
                try:
                    # AMD GPU kontrolü
                    import os
                    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
                    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"
                    if hasattr(torch, 'hip') and torch.hip.is_available():
                        device_count = torch.hip.device_count()
                        device_name = torch.hip.get_device_name(0)
                        print(f"\nAMD GPU kullanılabilir: {device_count} GPU bulundu")
                        print(f"GPU: {device_name}")
                        return True
                    else:
                        # AMD GPU için özel kontrol
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        if str(device) == "cuda":
                            print("\nAMD GPU algılandı ve kullanılabilir")
                            return True
                except Exception as e:
                    print(f"\nAMD GPU kontrolünde hata: {str(e)}")
                
                print("\nUYARI: GPU bulunamadı, CPU kullanılacak!")
                return False
        except ImportError:
            print("\nUYARI: PyTorch yüklü değil!")
            return False

    @staticmethod
    def setup_environment():
        """Çalışma ortamını hazırlar"""
        import os
        
        # PYTHONPATH'i ayarla
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        # Hugging Face uyarısını devre dışı bırak
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        
        # AMD GPU için gerekli ortam değişkenleri
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"  # RX 6600 için
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:512"
        os.environ["ROCM_PATH"] = "C:\\AMD"  # AMD GPU araçlarının yolu

if __name__ == "__main__":
    DependencyChecker.check_and_install_packages()
    DependencyChecker.setup_environment() 