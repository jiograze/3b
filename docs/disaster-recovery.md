# Ötüken3D Disaster Recovery Dokümantasyonu

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Backup Stratejisi](#backup-stratejisi)
- [Restore Prosedürleri](#restore-prosedürleri)
- [Failover Mekanizması](#failover-mekanizması)
- [Veri Replikasyonu](#veri-replikasyonu)

## Genel Bakış

Ötüken3D projesi için kapsamlı bir disaster recovery stratejisi uygulanmıştır. Bu dokümantasyon, sistemin yedekleme, geri yükleme ve felaket kurtarma süreçlerini detaylı olarak açıklamaktadır.

## Backup Stratejisi

### 1. Velero Kurulumu

```bash
velero install \
    --provider aws \
    --plugins velero/velero-plugin-for-aws:v1.2.0 \
    --bucket otuken3d-backup \
    --backup-location-config region=eu-central-1 \
    --snapshot-location-config region=eu-central-1 \
    --secret-file ./credentials-velero
```

### 2. Backup Yapılandırması

```yaml
apiVersion: velero.io/v1
kind: BackupStorageLocation
metadata:
  name: otuken3d-backup
spec:
  provider: aws
  objectStorage:
    bucket: otuken3d-backup
    prefix: cluster-backup
```

### 3. Backup Planı

1. **Günlük Full Backup**:
```yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: daily-backup
spec:
  schedule: "0 1 * * *"
  template:
    includedNamespaces:
    - default
    - monitoring
```

2. **Saatlik Incremental Backup**:
```yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: hourly-backup
spec:
  schedule: "0 * * * *"
  template:
    includedNamespaces:
    - default
```

## Restore Prosedürleri

### 1. Tam Sistem Restore

```bash
velero restore create --from-backup backup-name
```

### 2. Seçici Restore

```bash
velero restore create --from-backup backup-name \
  --include-namespaces default \
  --include-resources deployments,services
```

### 3. Restore Doğrulama

```bash
velero restore describe restore-name
velero restore logs restore-name
```

## Failover Mekanizması

### 1. Multi-Region Deployment

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: otuken3d-failover
spec:
  host: otuken3d
  trafficPolicy:
    loadBalancer:
      simple: ROUND_ROBIN
    outlierDetection:
      consecutive5xxErrors: 3
      interval: 10s
```

### 2. Otomatik Failover

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: otuken3d-failover
spec:
  hosts:
  - otuken3d
  http:
  - route:
    - destination:
        host: otuken3d-primary
      weight: 100
    - destination:
        host: otuken3d-secondary
      weight: 0
```

### 3. Manuel Failover

```bash
# Primary'den Secondary'ye geçiş
kubectl apply -f deployment/disaster-recovery/failover-to-secondary.yaml

# Secondary'den Primary'ye geri dönüş
kubectl apply -f deployment/disaster-recovery/failback-to-primary.yaml
```

## Veri Replikasyonu

### 1. Database Replikasyonu

```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: otuken3d-db
spec:
  instances: 3
  postgresql:
    parameters:
      max_connections: "100"
      shared_buffers: 256MB
  bootstrap:
    recovery:
      source: otuken3d-db-primary
```

### 2. Storage Replikasyonu

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: otuken3d-replicated-storage
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  replication-type: synchronous
```

## Recovery Testing

### 1. Test Planı

1. **Haftalık Testler**:
- Database restore testi
- Failover testi
- Veri tutarlılığı kontrolü

2. **Aylık Testler**:
- Tam sistem restore testi
- Multi-region failover testi
- Performans testi

### 2. Test Senaryoları

1. **Database Failure**:
```bash
# Test başlatma
kubectl exec -it otuken3d-db-0 -- pg_ctl stop
# Recovery kontrolü
kubectl get pods -l app=otuken3d-db -w
```

2. **Network Partition**:
```bash
# Network policy uygulama
kubectl apply -f test/network-partition.yaml
# Failover kontrolü
kubectl get pods -o wide
```

3. **Region Failure**:
```bash
# Region isolation
kubectl taint nodes -l topology.kubernetes.io/zone=eu-central-1a NoSchedule
# Failover kontrolü
kubectl get pods -o wide
```

## Metrikler ve Monitoring

### 1. Recovery Metrikleri

- Recovery Time Objective (RTO)
- Recovery Point Objective (RPO)
- Failover başarı oranı
- Backup başarı oranı

### 2. Alerting

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: disaster-recovery-alerts
spec:
  groups:
  - name: backup
    rules:
    - alert: BackupFailed
      expr: backup_status{job="velero"} != 1
      for: 1h
```

## Best Practices

1. **Backup Yönetimi**:
- Düzenli backup doğrulama
- Backup rotasyonu
- Encryption kullanımı

2. **Recovery Prosedürleri**:
- Dokümante edilmiş recovery planı
- Düzenli test ve güncelleme
- Otomatizasyon kullanımı

3. **Monitoring**:
- Proaktif monitoring
- Detaylı logging
- Anomali tespiti 