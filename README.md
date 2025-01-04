# Otuken3D

3D model işleme ve dönüştürme API'si.

## Özellikler

- 3D model optimizasyonu ve onarımı
- Doku haritası işleme ve dönüştürme
- Format dönüşümü (OBJ, STL, PLY, GLTF, GLB)
- Toplu işlem desteği
- RESTful API arayüzü

## Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/jiograze/3b.git
cd 3b
```

2. Sanal ortam oluşturun ve etkinleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# veya
.\venv\Scripts\activate  # Windows
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

1. Sunucuyu başlatın:
```bash
cd src
uvicorn main:app --reload
```

2. API belgelerine erişin:
```
http://localhost:8000/docs
```

## API Uç Noktaları

### Mesh İşleme

- `POST /api/v1/mesh/optimize`: 3D modeli optimize eder
- `POST /api/v1/mesh/repair`: 3D modeldeki hataları onarır

### Doku İşleme

- `POST /api/v1/texture/process`: Doku haritasını işler

### Format Dönüştürme

- `POST /api/v1/convert`: Tekil model dönüştürme
- `POST /api/v1/batch/convert`: Toplu model dönüştürme

### Bilgi

- `GET /api/v1/formats`: Desteklenen formatları listeler

## Yapılandırma

Yapılandırma ayarları `config.yml` dosyasında veya çevre değişkenleriyle belirtilebilir:

```yaml
app:
  name: "Otuken3D"
  version: "0.1.0"

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

storage:
  temp_dir: "/tmp/otuken3d"
  max_upload_size: 104857600  # 100MB
```

## Geliştirme

1. Test çalıştırma:
```bash
pytest
```

2. Kod formatı:
```bash
black src/
```

3. Lint kontrolü:
```bash
flake8 src/
```

4. Tip kontrolü:
```bash
mypy src/
```

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.
