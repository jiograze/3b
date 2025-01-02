# 🎨 Ötüken3D - 3D Voxel Autoencoder

Ötüken3D, 3D voxel modellerini işlemek ve üretmek için geliştirilmiş bir deep learning projesidir.

## 🚀 Özellikler

- 3D Voxel Autoencoder mimarisi
- GPU/CPU otomatik algılama ve optimizasyon
- AMD GPU (ROCm) desteği
- Google Colab Pro desteği
- Otomatik bellek yönetimi
- Gelişmiş görselleştirme araçları
- Wandb entegrasyonu

## 📦 Gereksinimler

```bash
# Temel paketler
numpy>=1.21.0
matplotlib>=3.4.0
wandb>=0.12.0
tqdm>=4.62.0
psutil>=5.8.0
gputil>=1.4.0

# PyTorch (AMD ROCm)
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
```

## 🛠️ Kurulum

1. Repository'yi klonlayın:
```bash
git clone https://github.com/jiograze/otuken3d.git
cd otuken3d
```

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## 💻 Kullanım

### Google Colab'da Çalıştırma

1. `otuken3d_colab_pro.ipynb` dosyasını Google Colab'a yükleyin
2. Runtime > Change runtime type seçeneğinden GPU'yu seçin
3. Notebook'taki hücreleri sırayla çalıştırın

### Yerel Makinede Çalıştırma

```bash
# Örnek veri oluştur
python modules/training/create_sample_data.py

# Modeli eğit
python modules/training/train.py --data_dir data --output_dir outputs
```

## 📊 Eğitim Sonuçları

Eğitim metriklerini Weights & Biases üzerinden takip edebilirsiniz:
[Wandb Project Link](https://wandb.ai/jiograze/otuken3d)

## 📁 Proje Yapısı

```
otuken3d/
├── modules/
│   ├── training/
│   │   ├── otuken3d_model.py
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   └── create_sample_data.py
│   └── utils/
├── notebooks/
│   └── otuken3d_colab_pro.ipynb
├── data/
│   ├── train/
│   └── val/
├── outputs/
│   └── checkpoints/
├── requirements.txt
└── README.md
```

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Bir Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## ✨ Teşekkürler

- Google Colab ekibine ücretsiz GPU kaynakları için
- Weights & Biases ekibine eğitim takip araçları için
- AMD ROCm ekibine GPU desteği için
