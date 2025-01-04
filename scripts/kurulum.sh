#!/bin/bash

# Hata durumunda scripti durdur
set -e

echo "Conda'yı kaldırma..."
rm -rf ~/miniconda ~/.conda ~/anaconda ~/.anaconda

echo "Sistem paketlerini güncelleme..."
sudo apt update
sudo apt install -y software-properties-common

echo "Python 3.9 deposunu ekleme..."
sudo add-apt-repository -y ppa:deadsnakes/ppa

echo "Python 3.9 kurulumu..."
sudo apt install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    python3.9-distutils \
    python3.9-lib2to3 \
    python3.9-gdbm \
    python3.9-tk

echo "pip kurulumu..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py --user
rm get-pip.py

echo "Virtual environment oluşturma..."
VENV_PATH=~/otuken3d_venv
rm -rf $VENV_PATH
python3.9 -m venv $VENV_PATH --clear

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Hata: Virtual environment oluşturulamadı!"
    exit 1
fi

echo "Virtual environment izinlerini düzeltme..."
sudo chown -R $USER:$USER $VENV_PATH

echo "Kurulum tamamlandı!"
echo
echo "Sıradaki adımlar:"
echo "1. Terminal'i kapatıp yeni bir terminal açın"
echo "2. Virtual environment'ı aktive edin:"
echo "   source ~/otuken3d_venv/bin/activate"
echo "3. Modülleri yükleyin:"
echo "   pip install wheel"
echo "   pip install -r requirements.txt"
echo "4. Uygulamayı çalıştırın:"
echo "   PYTHONPATH=$PWD python src/main.py" 