Write-Host "Ötüken3D kurulumu başlıyor..."

# Ana dizinleri oluştur
$directories = @(
    "data/images",
    "data/3d_models",
    "data/text_prompts",
    "data/datasets/COCO",
    "data/datasets/ImageNet",
    "data/datasets/ShapeNet",
    "data/datasets/Pix3D",
    "data/feedback",
    "models/checkpoints",
    "models/pretrained",
    "models/generated",
    "models/scripts",
    "modules/core",
    "modules/data_management",
    "modules/nlp",
    "modules/image_processing",
    "modules/model_generation",
    "modules/training",
    "modules/evaluation",
    "modules/ui",
    "modules/security",
    "modules/deployment",
    "utils/helpers",
    "utils/config",
    "utils/logging",
    "tests/unit",
    "tests/integration",
    "tests/end_to_end",
    "docs/user_guide",
    "docs/api_docs",
    "docs/developer_guide",
    "config",
    "scripts"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir
    Write-Host "Created directory: $dir"
}

# Python paketlerini yükle
Write-Host "Python paketleri yükleniyor..."
python -m pip install -r requirements.txt

# Hugging Face cache dizinini ayarla
$env:HF_HOME = "models/pretrained"
$env:TORCH_HOME = "models/pretrained"

# Önceden eğitilmiş modelleri indir
Write-Host "Önceden eğitilmiş modeller indiriliyor..."

# CLIP modelini indir
python -c "from transformers import CLIPTextModel, CLIPTokenizer; CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='models/pretrained'); CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', cache_dir='models/pretrained')"

# Point-E modelini indir
python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('openai/point-e-base', cache_dir='models/pretrained')"

# MiDaS modelini indir
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS', pretrained=True)"

# Veri seti yöneticisini kullanarak veri setlerini indir
Write-Host "`nVeri setleri yönetimi"
Write-Host "-------------------"

# Python script'ini çalıştır
$dataset_script = @"
from modules.data_management.dataset_manager import DatasetManager

dm = DatasetManager()
dm.list_datasets()

print('\nHangi veri setlerini indirmek istersiniz?')
print('1: Sadece temel veri setleri (önerilen, ~1.2GB)')
print('2: Orta seviye veri setleri (~3.2GB)')
print('3: Tüm veri setleri (~33GB)')
print('0: Veri seti indirme')

choice = input('Seçiminiz (0-3): ')
if choice.isdigit():
    dm.download_by_priority(int(choice))
"@

python -c $dataset_script

# Yapılandırma dosyasını oluştur
if (Test-Path "config/config.yaml.example") {
    Copy-Item "config/config.yaml.example" -Destination "config/config.yaml"
}

Write-Host "Ötüken3D kurulumu tamamlandı!"