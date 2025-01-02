#!/bin/bash

echo "Ötüken3D kurulumu başlıyor..."

# Ana dizinleri oluştur
mkdir -p data/{images,3d_models,text_prompts,datasets/{COCO,ImageNet,ShapeNet,Pix3D},feedback}
mkdir -p models/{checkpoints,pretrained,generated,scripts}
mkdir -p modules/{core,data_management,nlp,image_processing,model_generation,training,evaluation,ui,security,deployment}
mkdir -p utils/{helpers,config,logging}
mkdir -p tests/{unit,integration,end_to_end}
mkdir -p docs/{user_guide,api_docs,developer_guide}
mkdir -p config
mkdir -p scripts

# Python paketlerini yükle
echo "Python paketleri yükleniyor..."
python3 -m pip install -r requirements.txt

# Hugging Face cache dizinini ayarla
export HF_HOME="models/pretrained"
export TORCH_HOME="models/pretrained"

# Önceden eğitilmiş modelleri indir
echo "Önceden eğitilmiş modeller indiriliyor..."

# CLIP modelini indir
python3 -c "from transformers import CLIPTextModel, CLIPTokenizer; CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32', cache_dir='models/pretrained'); CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', cache_dir='models/pretrained')"

# Point-E modelini indir
python3 -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('openai/point-e-base', cache_dir='models/pretrained')"

# MiDaS modelini indir
python3 -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS', pretrained=True)"

# Veri seti yöneticisini kullanarak veri setlerini indir
echo -e "\nVeri setleri yönetimi"
echo "-------------------"

# Python script'ini çalıştır
python3 -c "
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
"

# Yapılandırma dosyasını oluştur
if [ -f "config/config.yaml.example" ]; then
    cp config/config.yaml.example config/config.yaml
fi

echo "Ötüken3D kurulumu tamamlandı!" 