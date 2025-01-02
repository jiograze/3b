# 📚 Ötüken3D API Referansı

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Kimlik Doğrulama](#kimlik-doğrulama)
- [Endpoints](#endpoints)
- [Modeller](#modeller)
- [Hata Kodları](#hata-kodları)

## Genel Bakış

Ötüken3D API'si, RESTful prensiplerini takip eden ve JSON formatında iletişim kuran bir API'dir. Tüm istekler HTTPS üzerinden yapılmalıdır.

**Base URL**: `https://api.otuken3d.ai/v1`

## Kimlik Doğrulama

API, JWT (JSON Web Token) tabanlı kimlik doğrulama kullanır.

```bash
curl -X POST https://api.otuken3d.ai/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "your-username", "password": "your-password"}'
```

Başarılı kimlik doğrulama sonrası alınan token, sonraki isteklerde `Authorization` header'ında kullanılmalıdır:

```bash
curl -X GET https://api.otuken3d.ai/v1/models \
  -H "Authorization: Bearer your-token"
```

## Endpoints

### Text-to-3D

#### Yeni Model Oluşturma

```http
POST /text-to-3d
```

**Request Body**:
```json
{
  "text": "Antik Türk motifli vazo",
  "style": "realistic",
  "resolution": "high",
  "format": "glb"
}
```

**Response**:
```json
{
  "task_id": "t_1234567890",
  "status": "processing",
  "estimated_time": 120
}
```

#### Model Durumu Sorgulama

```http
GET /text-to-3d/{task_id}
```

**Response**:
```json
{
  "task_id": "t_1234567890",
  "status": "completed",
  "model_url": "https://storage.otuken3d.ai/models/1234567890.glb",
  "preview_url": "https://storage.otuken3d.ai/previews/1234567890.png"
}
```

### Image-to-3D

#### Yeni Model Oluşturma

```http
POST /image-to-3d
```

**Request Body** (multipart/form-data):
```yaml
images: [file1.jpg, file2.jpg]  # En az 1, en fazla 8 görsel
options:
  resolution: "high"
  format: "glb"
  optimize: true
```

**Response**:
```json
{
  "task_id": "t_0987654321",
  "status": "processing",
  "estimated_time": 180
}
```

## Modeller

### Task

```typescript
interface Task {
  task_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  created_at: string;
  updated_at: string;
  estimated_time?: number;
  progress?: number;
  result?: {
    model_url: string;
    preview_url: string;
    format: string;
    size: number;
  };
  error?: {
    code: string;
    message: string;
  };
}
```

### Model Options

```typescript
interface ModelOptions {
  resolution: "low" | "medium" | "high";
  format: "obj" | "glb" | "fbx" | "stl";
  optimize: boolean;
  style?: string;
  scale?: number;
}
```

## Hata Kodları

| Kod | Açıklama |
|-----|-----------|
| 400 | Bad Request - İstek formatı hatalı |
| 401 | Unauthorized - Kimlik doğrulama gerekli |
| 403 | Forbidden - Yetkisiz erişim |
| 404 | Not Found - Kaynak bulunamadı |
| 429 | Too Many Requests - Rate limit aşıldı |
| 500 | Internal Server Error - Sunucu hatası |

### Hata Yanıtı Formatı

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again in 60 seconds.",
    "details": {
      "limit": 100,
      "remaining": 0,
      "reset": 1609459200
    }
  }
}
```

## Rate Limiting

- Anonim istekler: 100 istek/saat
- Kimlik doğrulamalı istekler: 1000 istek/saat
- Model oluşturma: 10 istek/saat

Rate limit bilgileri yanıt header'larında döner:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

## Webhook Entegrasyonu

Model oluşturma tamamlandığında bildirim almak için webhook URL'i tanımlayabilirsiniz:

```http
POST /text-to-3d
```

```json
{
  "text": "Antik Türk motifli vazo",
  "webhook_url": "https://your-domain.com/webhook"
}
```

Webhook payload'ı:

```json
{
  "task_id": "t_1234567890",
  "status": "completed",
  "model_url": "https://storage.otuken3d.ai/models/1234567890.glb",
  "preview_url": "https://storage.otuken3d.ai/previews/1234567890.png",
  "metadata": {
    "processing_time": 120,
    "format": "glb",
    "size": 1048576
  }
}
``` 