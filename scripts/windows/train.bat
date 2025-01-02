@echo off
chcp 65001 > nul
echo Ötüken3D model eğitimi başlatılıyor...

:: Python yolunu kontrol et
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python bulunamadı! Lütfen Python'u yükleyin ve PATH'e ekleyin.
    echo Python indirme sayfası: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Python versiyonunu kontrol et
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Python versiyonu kontrol edilemedi!
    pause
    exit /b 1
)

:: Gerekli paketleri kontrol et
echo Gerekli paketler kontrol ediliyor...
python -c "import torch; import wandb" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Eksik paketler bulundu. Paketler yükleniyor...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Paket yüklemesi başarısız oldu!
        pause
        exit /b 1
    )
)

:: Hugging Face uyarısını devre dışı bırak
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

:: AMD ROCm için gerekli ortam değişkenlerini ayarla
set PYTORCH_ROCM_ARCH=gfx1030
set HSA_OVERRIDE_GFX_VERSION=10.3.0

:: GPU kontrolü
python -c "import torch; gpu_available = torch.cuda.is_available() or (hasattr(torch, 'hip') and torch.hip.is_available()); print('GPU Kullanılabilir:', gpu_available)"

:: Eğitimi başlat
echo.
echo Eğitim başlatılıyor...
python scripts/train.py --config config/training_config.yaml

if %ERRORLEVEL% NEQ 0 (
    echo Eğitim sırasında bir hata oluştu!
)

pause 