#!/bin/bash

echo -e "\e[32mÖtüken3D projesi başlatılıyor...\e[0m"

# Python kontrolü
if ! command -v python3 &> /dev/null; then
    echo -e "\e[31mPython bulunamadı! Lütfen Python'u yükleyin.\e[0m"
    echo "Debian/Ubuntu için: sudo apt install python3"
    echo "Pop!_OS için: sudo apt install python3"
    exit 1
fi

# Pip kontrolü
if ! command -v pip3 &> /dev/null; then
    echo -e "\e[31mPip bulunamadı! Lütfen pip'i yükleyin.\e[0m"
    echo "Debian/Ubuntu için: sudo apt install python3-pip"
    exit 1
fi

# Temel dizin yapısını oluştur
echo "1/4 Proje yapısı oluşturuluyor..."
python3 scripts/create_structure.py

# Bağımlılıkları yükle
echo "2/4 Bağımlılıklar yükleniyor..."
pip3 install -r requirements.txt

# Bağımlılıkları kontrol et
echo "3/4 Bağımlılıklar kontrol ediliyor..."
python3 scripts/check_dependencies.py

# Veri setlerini indir
echo "4/4 Veri setleri indiriliyor..."
python3 scripts/dataset_manager.py

echo -e "\e[32mKurulum tamamlandı!\e[0m"
