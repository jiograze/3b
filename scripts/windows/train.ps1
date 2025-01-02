Write-Host "Ötüken3D model eğitimi başlatılıyor..." -ForegroundColor Green

# Python yolunu kontrol et
$pythonPath = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonPath) {
    Write-Host "Python bulunamadı! Lütfen Python'u yükleyin ve PATH'e ekleyin." -ForegroundColor Red
    Write-Host "Python indirme sayfası: https://www.python.org/downloads/"
    pause
    exit 1
}

# Python versiyonunu kontrol et
Write-Host "Python versiyonu:" -NoNewline
python --version

# Gerekli paketleri kontrol et
Write-Host "Gerekli paketler kontrol ediliyor..."
$packages = python -c "import torch; import wandb" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Eksik paketler bulundu. Paketler yükleniyor..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Paket yüklemesi başarısız oldu!" -ForegroundColor Red
        pause
        exit 1
    }
}

# Hugging Face uyarısını devre dışı bırak
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# CUDA kontrolü
python -c "import torch; print('CUDA Kullanılabilir:', torch.cuda.is_available())"

# CUDA görünür cihazları ayarla (isteğe bağlı)
$env:CUDA_VISIBLE_DEVICES = "0"

# Eğitimi başlat
Write-Host "`nEğitim başlatılıyor..." -ForegroundColor Green
python scripts/train.py --config config/training_config.yaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "Eğitim sırasında bir hata oluştu!" -ForegroundColor Red
}

pause 