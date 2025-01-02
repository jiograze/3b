Write-Host "Ötüken3D bağımlılıkları yükleniyor..." -ForegroundColor Green

# Python yolunu kontrol et
$pythonPath = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonPath) {
    Write-Host "Python bulunamadı! Lütfen Python'u yükleyin ve PATH'e ekleyin." -ForegroundColor Red
    Write-Host "Python indirme sayfası: https://www.python.org/downloads/"
    pause
    exit 1
}

# pip'i güncelle
python -m pip install --upgrade pip

# Temel paketleri yükle
pip install -e .

# Eksik paketleri kontrol et ve yükle
Write-Host "Gerekli paketler kontrol ediliyor..."

# PyYAML
$yaml = python -c "import yaml" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyYAML yükleniyor..." -ForegroundColor Yellow
    pip install pyyaml
}

# Wandb
$wandb = python -c "import wandb" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Wandb yükleniyor..." -ForegroundColor Yellow
    pip install wandb
}

# Diğer paketler
$others = python -c "import torch; import trimesh; import open3d" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Diğer gerekli paketler yükleniyor..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "Kurulum tamamlandı!" -ForegroundColor Green
pause 