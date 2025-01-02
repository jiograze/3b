# Yardımcı Araçlar (Utils)

Bu dizin, Ötüken3D projesinde kullanılan yardımcı araçları ve fonksiyonları içerir.

## 📦 Yapı

```
utils/
├── helpers/           # Genel yardımcı fonksiyonlar
├── config/           # Yapılandırma yönetimi
└── logging/          # Loglama araçları
```

## 🔧 Araçlar

### Helpers

Genel amaçlı yardımcı fonksiyonlar ve sınıflar:

- `decorators.py`: Özel dekoratörler
  - `@timer`: Fonksiyon çalışma süresini ölçer
  - `@retry`: Başarısız işlemleri tekrar dener
  - `@cache`: Sonuçları önbelleğe alır

- `validators.py`: Veri doğrulama araçları
  - `validate_input`: Kullanıcı girdilerini doğrular
  - `validate_model`: Model çıktılarını doğrular
  - `validate_config`: Yapılandırma dosyalarını doğrular

- `converters.py`: Format dönüştürücüler
  - `obj_to_stl`: OBJ formatından STL'ye dönüştürür
  - `stl_to_obj`: STL formatından OBJ'ye dönüştürür
  - `gltf_to_obj`: GLTF formatından OBJ'ye dönüştürür

### Config

Yapılandırma yönetimi araçları:

- `config_loader.py`: Yapılandırma yükleme
  - `load_config`: YAML yapılandırma dosyasını yükler
  - `update_config`: Yapılandırmayı günceller
  - `validate_config`: Yapılandırmayı doğrular

- `defaults.py`: Varsayılan değerler
  - `DEFAULT_CONFIG`: Varsayılan yapılandırma
  - `DEFAULT_PATHS`: Varsayılan dizin yolları
  - `DEFAULT_PARAMS`: Varsayılan model parametreleri

### Logging

Loglama ve izleme araçları:

- `logger.py`: Merkezi loglama sistemi
  - `setup_logging`: Loglama sistemini yapılandırır
  - `get_logger`: Logger örneği döndürür
  - `log_metrics`: Model metriklerini loglar

- `handlers.py`: Özel log işleyicileri
  - `FileHandler`: Dosya tabanlı loglama
  - `ConsoleHandler`: Konsol tabanlı loglama
  - `MetricsHandler`: Metrik loglama

## 💡 Kullanım Örnekleri

### Dekoratör Kullanımı

```python
from utils.helpers.decorators import timer

@timer
def process_model(model_data):
    # İşlem kodu
    pass
```

### Yapılandırma Yükleme

```python
from utils.config.config_loader import load_config

config = load_config('config/model_config.yaml')
```

### Loglama

```python
from utils.logging.logger import get_logger

logger = get_logger(__name__)
logger.info('İşlem başladı')
```

## 🔄 Güncellemeler

Yeni bir yardımcı araç eklerken:

1. İlgili alt dizinde uygun bir modül oluşturun
2. Birim testlerini yazın
3. Dokümantasyonu güncelleyin
4. Bu README'yi güncelleyin

## 🧪 Test

```bash
# Tüm utils testlerini çalıştır
pytest tests/utils/

# Belirli bir modülün testlerini çalıştır
pytest tests/utils/helpers/
```

## 📝 Stil Kılavuzu

1. PEP 8 kurallarına uyun
2. Docstring'leri Google formatında yazın
3. Tip bilgilerini (type hints) kullanın
4. Fonksiyon ve değişken isimlerini açıklayıcı yapın
5. Karmaşık işlemleri yorumlarla açıklayın

## 🤝 Katkıda Bulunma

1. Yeni bir yardımcı araç eklemek için:
   - İlgili alt dizinde modül oluşturun
   - Testleri yazın
   - Dokümantasyonu güncelleyin

2. Mevcut bir aracı güncellemek için:
   - Değişiklikleri CHANGELOG.md'ye ekleyin
   - Testleri güncelleyin
   - Dokümantasyonu güncelleyin 