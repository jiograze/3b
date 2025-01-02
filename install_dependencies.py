import subprocess
import sys
import pkg_resources
import platform
import os

def check_gpu():
    """GPU tipini ve kullanılabilirliğini kontrol eder."""
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "AMD"
        elif torch.cuda.is_available():
            return "NVIDIA"
        else:
            return "CPU"
    except ImportError:
        return "CPU"

def install_pytorch3d_from_source():
    """PyTorch3D'yi kaynak koddan derler ve yükler."""
    print("PyTorch3D kaynak koddan derleniyor...")
    
    # Gerekli bağımlılıkları yükle
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fvcore", "iopath"])
    
    # CXX11 ABI uyumluluğu için
    env = os.environ.copy()
    env["FORCE_CUDA"] = "0"  # AMD GPU için CUDA'yı devre dışı bırak
    env["PYTORCH3D_NO_CUDA_EXTENSION"] = "1"
    
    # PyTorch3D'yi klonla ve derle
    subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/pytorch3d.git"])
    
    # AMD için özel derleme seçenekleri
    env["HIP_PATH"] = "/opt/rocm/hip"  # ROCm yolu
    env["ROCM_PATH"] = "/opt/rocm"
    
    try:
        # Derleme ve kurulum
        subprocess.check_call([sys.executable, "setup.py", "install"], cwd="pytorch3d", env=env)
    except subprocess.CalledProcessError:
        print("PyTorch3D derleme hatası. CPU versiyonu yüklenecek...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-index",
            "--no-deps",
            "pytorch3d"
        ])
    finally:
        # Geçici dosyaları temizle
        subprocess.check_call(["rm", "-rf", "pytorch3d"])

def install_package(package):
    """Paketi pip ile yükler."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"UYARI: {package} yüklenemedi")

def check_and_install_dependencies():
    """Gerekli bağımlılıkları kontrol eder ve eksik olanları yükler."""
    required_packages = {
        "numpy": "numpy>=1.21.0",
        "pillow": "pillow>=8.3.1",
        "trimesh": "trimesh>=3.9.29",
        "matplotlib": "matplotlib>=3.4.3",
        "wandb": "wandb>=0.12.0",
        "tqdm": "tqdm>=4.62.2",
        "transformers": "transformers>=4.11.3",
        "pytorch3d-cpu": "pytorch3d-cpu"  # CPU versiyonu
    }
    
    print("Bağımlılıklar kontrol ediliyor...")
    
    # GPU kontrolü
    gpu_type = check_gpu()
    print(f"Tespit edilen GPU tipi: {gpu_type}")
    
    if gpu_type == "AMD":
        print("AMD GPU tespit edildi. ROCm uyumlu kurulum yapılacak...")
        # ROCm uyumlu PyTorch ve torchvision
        install_package("--extra-index-url https://download.pytorch.org/whl/rocm5.4.2 torch")
        install_package("--extra-index-url https://download.pytorch.org/whl/rocm5.4.2 torchvision")
    else:
        # CPU versiyonu
        install_package("torch")
        install_package("torchvision")
    
    # Diğer paketleri yükle
    for package, requirement in required_packages.items():
        try:
            pkg_resources.require(requirement)
            print(f"✓ {package} zaten yüklü")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"× {package} yükleniyor...")
            install_package(requirement)
    
    # PyTorch3D kurulumu
    try:
        import pytorch3d
        print("✓ pytorch3d zaten yüklü")
    except ImportError:
        print("× pytorch3d yükleniyor...")
        if gpu_type == "AMD":
            install_pytorch3d_from_source()
        else:
            install_package("pytorch3d-cpu")
    
    print("\nTüm bağımlılıklar yüklendi!")

if __name__ == "__main__":
    check_and_install_dependencies() 