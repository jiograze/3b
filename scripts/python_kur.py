#!/usr/bin/env python3
"""
Python 3.9 Kurulum ve Ortam Hazırlama Scripti
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def remove_old_python():
    """Eski Python sürümünü kaldırır"""
    try:
        # Conda ortamını kontrol et
        if "conda" in sys.version:
            print("Conda ortamı tespit edildi")
            subprocess.check_call(["conda", "deactivate"])
            subprocess.check_call(["conda", "env", "remove", "--name", "base", "--all"])
            print("Conda base ortamı kaldırıldı")
    except Exception as e:
        print(f"Uyarı: Conda ortamı kaldırılırken hata: {e}")

def install_python39():
    """Python 3.9'u kurar"""
    os_name = platform.system().lower()
    
    if os_name == "linux":
        try:
            # Gerekli paketleri kur
            subprocess.check_call(["sudo", "apt", "update"])
            subprocess.check_call(["sudo", "apt", "install", "-y", "python3.9", "python3.9-venv", "python3.9-dev"])
            print("Python 3.9 kuruldu")
            
            # Python 3.9'u varsayılan yap
            subprocess.check_call(["sudo", "update-alternatives", "--install", 
                                 "/usr/bin/python3", "python3", "/usr/bin/python3.9", "1"])
            subprocess.check_call(["sudo", "update-alternatives", "--set", "python3", "/usr/bin/python3.9"])
            print("Python 3.9 varsayılan sürüm yapıldı")
            
        except subprocess.CalledProcessError as e:
            print(f"Hata: Python 3.9 kurulumu başarısız: {e}")
            sys.exit(1)
    else:
        print(f"Bu işletim sistemi ({os_name}) için otomatik kurulum desteklenmiyor.")
        print("Lütfen Python 3.9'u manuel olarak kurun: https://www.python.org/downloads/")
        sys.exit(1)

def setup_virtualenv():
    """Virtual environment oluşturur"""
    try:
        venv_path = Path.home() / "otuken3d_venv"
        if not venv_path.exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
            print(f"Virtual environment oluşturuldu: {venv_path}")
            
            # Aktivasyon komutlarını göster
            print("\nVirtual environment'ı aktive etmek için:")
            print(f"source {venv_path}/bin/activate")
        else:
            print(f"Virtual environment zaten mevcut: {venv_path}")
    except Exception as e:
        print(f"Hata: Virtual environment oluşturulamadı: {e}")
        sys.exit(1)

def main():
    """Ana fonksiyon"""
    print("Python 3.9 kurulumu başlıyor...")
    
    remove_old_python()
    install_python39()
    setup_virtualenv()
    
    print("\nKurulum tamamlandı!")
    print("\nSıradaki adımlar:")
    print("1. Terminal'i yeniden başlatın")
    print("2. Virtual environment'ı aktive edin:")
    print("   source ~/otuken3d_venv/bin/activate")
    print("3. Modülleri yükleyin:")
    print("   python scripts/yukleme.py")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Bu script root yetkileri gerektirir.")
        print("Lütfen 'sudo python scripts/python_kur.py' şeklinde çalıştırın")
        sys.exit(1)
    main() 