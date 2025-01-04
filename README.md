# Otuken3D

3B model işleme ve dönüştürme API'si.

## Özellikler

- 3B model optimizasyonu
- Mesh onarımı
- Doku haritası işleme
- Format dönüştürme
- Toplu işleme desteği

## Kurulum

1. Python 3.8+ gereklidir.

2. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

3. Uygulamayı başlatın:
```bash
python src/main.py
```

Uygulama varsayılan olarak `http://0.0.0.0:8000` adresinde çalışacaktır.

## API Kullanımı

### Mesh Optimizasyonu

```http
POST /api/mesh/optimize
```

Parametreler:
- `file`: 3B model dosyası
- `target_faces`: Hedef üçgen sayısı (opsiyonel)
- `preserve_uv`: UV koordinatlarını koru (varsayılan: true)

### Mesh Onarımı

```http
POST /api/mesh/repair
```

Parametreler:
- `file`: 3B model dosyası
- `fix_normals`: Normalleri düzelt (varsayılan: true)
- `remove_duplicates`: Tekrarlanan noktaları sil (varsayılan: true)

### Doku İşleme

```http
POST /api/texture/process
```

Parametreler:
- `file`: Doku dosyası
- `width`: Yeni genişlik (opsiyonel)
- `height`: Yeni yükseklik (opsiyonel)
- `quality`: Çıktı kalitesi (0-100, opsiyonel)

### Format Dönüştürme

```http
POST /api/convert
```

Parametreler:
- `file`: 3B model dosyası
- `output_format`: Çıktı formatı
- `preserve_materials`: Materyalleri koru (varsayılan: true)
- `optimize_mesh`: Mesh'i optimize et (varsayılan: true)

### Toplu Dönüştürme

```http
POST /api/batch/convert
```

Parametreler:
- `files`: 3B model dosyaları
- `output_format`: Çıktı formatı
- `preserve_materials`: Materyalleri koru (varsayılan: true)
- `optimize_mesh`: Mesh'i optimize et (varsayılan: true)

### Desteklenen Formatlar

```http
GET /api/formats
```

Desteklenen dosya formatlarını listeler.

## Yapılandırma

Uygulama ayarları `config.yml` dosyasından veya çevre değişkenlerinden yüklenebilir:

```yaml
app:
  name: Otuken3D
  version: 0.1.0
  description: 3B Model İşleme API

server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 60

storage:
  upload_dir: uploads
  output_dir: outputs
  temp_dir: temp
  max_file_size: 104857600  # 100MB

processing:
  max_vertices: 100000
  max_faces: 50000
  texture_size: 2048
  texture_quality: 90

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/otuken3d.log

security:
  allowed_formats: [".obj", ".stl", ".ply", ".glb", ".gltf", ".fbx", ".dae"]
  allowed_origins: ["*"]
  max_batch_size: 10
```

## Lisans

MIT
