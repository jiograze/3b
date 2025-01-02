import os
import sys
import subprocess
import pkg_resources

def check_and_install_dependencies():
    """Gerekli bağımlılıkları kontrol et ve eksik olanları yükle"""
    
    # Gerekli paketler ve sürümleri
    required_packages = {
        'torch': '2.0.0',
        'torchvision': '0.15.0',
        'transformers': '4.30.0',
        'wandb': '0.15.0',
        'open3d': '0.17.0',
        'numpy': '1.24.0',
        'scipy': '1.10.0',
        'tqdm': '4.65.0',
        'matplotlib': '3.7.0',
        'streamlit': '1.25.0',
        'pyyaml': '6.0',
        'pillow': '9.5.0',
        'requests': '2.31.0'
    }
    
    print("Bağımlılıklar kontrol ediliyor...")
    
    # Mevcut paketleri kontrol et
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Eksik veya eski sürüm paketleri belirle
    packages_to_install = []
    for package, version in required_packages.items():
        if package not in installed_packages:
            packages_to_install.append(f"{package}=={version}")
        elif pkg_resources.parse_version(installed_packages[package]) < pkg_resources.parse_version(version):
            packages_to_install.append(f"{package}=={version}")
    
    # Eksik paketleri yükle
    if packages_to_install:
        print("\nEksik paketler yükleniyor:")
        for package in packages_to_install:
            print(f"- {package}")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *packages_to_install])
            print("\nTüm paketler başarıyla yüklendi!")
        except subprocess.CalledProcessError as e:
            print(f"\nPaket yükleme hatası: {str(e)}")
            sys.exit(1)
    else:
        print("\nTüm gerekli paketler zaten yüklü!")

def setup_environment():
    """Çalışma ortamını hazırla"""
    
    # Proje kök dizinini belirle
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Gerekli dizinleri oluştur
    directories = [
        "data/datasets",
        "models/checkpoints",
        "outputs/samples",
        "logs"
    ]
    
    print("\nDizin yapısı oluşturuluyor...")
    for directory in directories:
        path = os.path.join(project_root, directory)
        os.makedirs(path, exist_ok=True)
        print(f"- {directory} oluşturuldu")
    
    # Python yoluna proje kök dizinini ekle
    if project_root not in sys.path:
        sys.path.append(project_root)
        print("\nProje kök dizini Python yoluna eklendi")

def main():
    """Ana kurulum fonksiyonu"""
    print("Kurulum başlatılıyor...\n")
    
    # Bağımlılıkları kontrol et ve yükle
    check_and_install_dependencies()
    
    # Ortamı hazırla
    setup_environment()
    
    print("\nKurulum tamamlandı!")

if __name__ == "__main__":
    main() 