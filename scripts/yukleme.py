#!/usr/bin/env python3
"""
Modül Yükleme ve Ortam Hazırlama Scripti
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Python versiyonunu kontrol eder"""
    if sys.version_info < (3, 9):
        print("Hata: Python 3.9 veya üstü gerekli")
        sys.exit(1)
    print(f"Python versiyonu: {sys.version}")

def install_requirements():
    """Gerekli paketleri yükler"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("Pip güncellendi")
        
        requirements_path = Path(__file__).parent.parent / "requirements.txt"
        if not requirements_path.exists():
            print("Hata: requirements.txt dosyası bulunamadı")
            sys.exit(1)
            
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        print("Gerekli paketler yüklendi")
    except subprocess.CalledProcessError as e:
        print(f"Hata: Paket yükleme başarısız: {e}")
        sys.exit(1)

def check_environment():
    """Çalışma ortamını kontrol eder"""
    # İşletim sistemi kontrolü
    os_name = platform.system()
    print(f"İşletim sistemi: {os_name}")
    
    # GPU kontrolü (NVIDIA için)
    if os_name == "Linux":
        try:
            subprocess.check_output(["nvidia-smi"])
            print("NVIDIA GPU bulundu")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("NVIDIA GPU bulunamadı veya nvidia-smi yüklü değil")
    
    # Gerekli dizinlerin varlığını kontrol et
    required_dirs = ["data", "logs", "models"]
    project_root = Path(__file__).parent.parent
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"{dir_name} dizini oluşturuldu")
        else:
            print(f"{dir_name} dizini mevcut")

def setup_environment_variables():
    """Çevre değişkenlerini ayarlar"""
    env_file = Path(__file__).parent.parent / ".env"
    
    if not env_file.exists():
        default_env = """
# Uygulama ayarları
APP_NAME=Otuken3D
APP_VERSION=1.0.0
DEBUG=True

# Sunucu ayarları
HOST=localhost
PORT=8000
WORKERS=4

# Güvenlik
ALLOWED_ORIGINS=["*"]

# Loglama
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
        with open(env_file, "w") as f:
            f.write(default_env.strip())
        print(".env dosyası oluşturuldu")
    else:
        print(".env dosyası mevcut")

def main():
    """Ana fonksiyon"""
    print("Ortam hazırlanıyor...")
    
    check_python_version()
    install_requirements()
    check_environment()
    setup_environment_variables()
    
    print("\nKurulum tamamlandı!")
    print("\nUygulamayı başlatmak için:")
    print("python src/main.py")

if __name__ == "__main__":
    main() 