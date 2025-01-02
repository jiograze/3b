# Ötüken3D Service Mesh (Istio) Dokümantasyonu

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Bileşenler](#bileşenler)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [Kullanım](#kullanım)
- [Sorun Giderme](#sorun-giderme)

## Genel Bakış

Ötüken3D projesi, mikroservis mimarisinin karmaşıklığını yönetmek için Istio service mesh çözümünü kullanmaktadır. Bu dokümantasyon, sistemin service mesh yapılandırmasını ve kullanımını detaylı olarak açıklamaktadır.

## Bileşenler

### 1. Virtual Service
- **Amaç**: Trafik yönlendirme kurallarını tanımlar
- **Özellikler**:
  - Canary deployment desteği
  - A/B testing
  - Fault injection
  - Timeout yapılandırması
  - Retry mekanizması

### 2. Destination Rules
- **Amaç**: Trafik politikalarını tanımlar
- **Özellikler**:
  - Load balancing
  - Circuit breaking
  - Connection pool
  - Outlier detection

### 3. Gateway
- **Amaç**: Dış trafiği yönetir
- **Özellikler**:
  - TLS terminasyonu
  - Port yapılandırması
  - Host routing

### 4. Authorization Policy
- **Amaç**: Güvenlik politikalarını tanımlar
- **Özellikler**:
  - RBAC kuralları
  - JWT doğrulama
  - Namespace izolasyonu

## Kurulum

1. Istio kurulumu:
```bash
istioctl install --set profile=demo
```

2. Namespace etiketleme:
```bash
kubectl label namespace default istio-injection=enabled
```

3. Yapılandırmaları uygulama:
```bash
kubectl apply -f deployment/istio/
```

## Yapılandırma

### Virtual Service Yapılandırması

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: otuken3d
spec:
  hosts:
  - "api.otuken3d.ai"
  gateways:
  - otuken3d-gateway
  http:
  - route:
    - destination:
        host: otuken3d
        subset: production
```

### Destination Rule Yapılandırması

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: otuken3d
spec:
  host: otuken3d
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
```

## Kullanım

### Trafik Yönetimi

1. Canary Deployment:
```bash
kubectl apply -f deployment/istio/canary.yaml
```

2. Circuit Breaking:
```bash
kubectl apply -f deployment/istio/circuit-breaker.yaml
```

### Monitoring

1. Kiali Dashboard:
```bash
istioctl dashboard kiali
```

2. Grafana:
```bash
istioctl dashboard grafana
```

## Sorun Giderme

### Sık Karşılaşılan Sorunlar

1. **503 Service Unavailable**
- Çözüm: Circuit breaker ayarlarını kontrol edin
- Destination rule yapılandırmasını gözden geçirin

2. **401 Unauthorized**
- Çözüm: JWT token'ı kontrol edin
- Authorization policy'i gözden geçirin

3. **Gateway Sync Hatası**
- Çözüm: Gateway yapılandırmasını kontrol edin
- SSL sertifikalarını kontrol edin

### Logging ve Debugging

1. Proxy loglarını görüntüleme:
```bash
kubectl logs -l app=otuken3d -c istio-proxy
```

2. Envoy konfigürasyonunu kontrol etme:
```bash
istioctl proxy-config all pod-name.default
```

### Performans İyileştirme

1. Connection Pool Ayarları:
```yaml
connectionPool:
  tcp:
    maxConnections: 100
  http:
    http2MaxRequests: 1000
```

2. Circuit Breaker Ayarları:
```yaml
outlierDetection:
  consecutive5xxErrors: 5
  interval: 30s
  baseEjectionTime: 30s
``` 