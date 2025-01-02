@echo off
chcp 65001 > nul
echo Ötüken3D bağımlılıkları yükleniyor...

:: Python yolunu kontrol et
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python bulunamadı! Lütfen Python'u yükleyin ve PATH'e ekleyin.
    echo Python indirme sayfası: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: pip'i güncelle
python -m pip install --upgrade pip

:: Temel paketleri yükle
pip install -e .

:: Eksik paketleri kontrol et ve yükle
echo Gerekli paketler kontrol ediliyor...

:: PyYAML
python -c "import yaml" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo PyYAML yükleniyor...
    pip install pyyaml
)

:: Wandb
python -c "import wandb" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Wandb yükleniyor...
    pip install wandb
)

:: Diğer paketler
python -c "import torch; import trimesh; import open3d" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Diğer gerekli paketler yükleniyor...
    pip install -r requirements.txt
)

echo Kurulum tamamlandı!
pause 