# 🛠️ Ötüken3D Geliştirici Kılavuzu

## İçindekiler
- [Geliştirme Ortamı](#geliştirme-ortamı)
- [Kod Standartları](#kod-standartları)
- [Test Yazımı](#test-yazımı)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)
- [Monitoring](#monitoring)

## Geliştirme Ortamı

### Gereksinimler

- Python 3.9.16
- CUDA 11.7+ (GPU geliştirmesi için)
- Docker ve Docker Compose
- Git
- VS Code (önerilen)

### Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/otuken3d/otuken3d.git
cd otuken3d
```

2. Virtual environment oluşturun:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# veya
.\venv\Scripts\activate  # Windows
```

3. Geliştirme bağımlılıklarını yükleyin:
```bash
pip install -e ".[dev]"
```

4. Pre-commit hooks'ları kurun:
```bash
pre-commit install
```

### VS Code Ayarları

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

## Kod Standartları

### Python Stil Kılavuzu

- [Black](https://black.readthedocs.io/) kod formatı
- [isort](https://pycqa.github.io/isort/) import düzeni
- [Flake8](https://flake8.pycqa.org/) linting
- [mypy](http://mypy-lang.org/) tip kontrolü
- [Google docstring](https://google.github.io/styleguide/pyguide.html) formatı

### Örnek Kod

```python
from typing import List, Optional

from otuken3d.core.base import BaseModel
from otuken3d.utils.logger import get_logger

logger = get_logger(__name__)

class TextToShapeModel(BaseModel):
    """Text-to-3D model sınıfı.

    Args:
        model_path: Model ağırlıklarının yolu
        device: Çalıştırılacak cihaz ('cuda' veya 'cpu')
        config: Model yapılandırması

    Attributes:
        model: Yüklenmiş PyTorch modeli
        tokenizer: Text tokenizer
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        config: Optional[dict] = None
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.config = config or {}
        self._load_model()

    def predict(self, text: str) -> List[float]:
        """Metinden 3D model embeddingi oluşturur.

        Args:
            text: Girdi metni

        Returns:
            3D model embeddingi

        Raises:
            ModelError: Model yüklenemediğinde
            ValueError: Geçersiz girdi
        """
        try:
            embedding = self._generate_embedding(text)
            return self._embedding_to_mesh(embedding)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
```

### Commit Mesajları

[Conventional Commits](https://www.conventionalcommits.org/) standardını kullanın:

- `feat`: Yeni özellik
- `fix`: Hata düzeltmesi
- `docs`: Dokümantasyon değişiklikleri
- `style`: Kod formatı değişiklikleri
- `refactor`: Kod yeniden düzenleme
- `test`: Test değişiklikleri
- `chore`: Yapılandırma değişiklikleri

Örnek:
```bash
git commit -m "feat(model): add text-to-shape transformer"
```

## Test Yazımı

### Unit Test Örneği

```python
import pytest
from otuken3d.models import TextToShapeModel

@pytest.fixture
def model():
    return TextToShapeModel(
        model_path="tests/fixtures/test_model.pt",
        device="cpu"
    )

def test_model_prediction(model):
    text = "Antik Türk motifli vazo"
    result = model.predict(text)
    
    assert isinstance(result, list)
    assert len(result) == model.embedding_dim
    assert all(isinstance(x, float) for x in result)

def test_invalid_input(model):
    with pytest.raises(ValueError):
        model.predict("")
```

### Integration Test Örneği

```python
import pytest
from otuken3d.pipeline import ModelPipeline

@pytest.mark.integration
def test_end_to_end_pipeline():
    pipeline = ModelPipeline()
    
    # Test input
    text = "Antik Türk motifli vazo"
    
    # Pipeline çalıştırma
    task = pipeline.create_task(text)
    result = pipeline.wait_for_result(task.id)
    
    # Sonuçları kontrol etme
    assert result.status == "completed"
    assert result.model_url.endswith(".glb")
    assert result.preview_url.endswith(".png")
```

## CI/CD Pipeline

### GitHub Actions Workflow

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9.16
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest --cov=otuken3d tests/
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/models/text2shape.pt
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker:
    build: .
    command: celery -A otuken3d.tasks worker
    environment:
      - CUDA_VISIBLE_DEVICES=1
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
```

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

MODEL_REQUESTS = Counter(
    'model_requests_total',
    'Total number of model requests',
    ['model_type', 'status']
)

PROCESSING_TIME = Histogram(
    'model_processing_seconds',
    'Time spent processing model requests',
    ['model_type']
)

def process_request(text: str) -> None:
    try:
        with PROCESSING_TIME.labels('text2shape').time():
            result = model.predict(text)
        MODEL_REQUESTS.labels('text2shape', 'success').inc()
    except Exception:
        MODEL_REQUESTS.labels('text2shape', 'error').inc()
        raise
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Ötüken3D Metrics",
    "panels": [
      {
        "title": "Model Requests",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(model_requests_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Processing Time",
        "type": "heatmap",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(model_processing_seconds_bucket[5m])",
            "legendFormat": "{{le}}"
          }
        ]
      }
    ]
  }
}
``` 