# Ötüken3D Maliyet Optimizasyonu Dokümantasyonu

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [Spot Instance Stratejisi](#spot-instance-stratejisi)
- [Resource Yönetimi](#resource-yönetimi)
- [Storage Optimizasyonu](#storage-optimizasyonu)
- [Network Optimizasyonu](#network-optimizasyonu)

## Genel Bakış

Ötüken3D projesi için kapsamlı bir maliyet optimizasyonu stratejisi uygulanmıştır. Bu dokümantasyon, sistemin kaynak kullanımını optimize etme ve maliyetleri düşürme stratejilerini detaylı olarak açıklamaktadır.

## Spot Instance Stratejisi

### 1. Karpenter Kurulumu

```bash
helm repo add karpenter https://charts.karpenter.sh
helm repo update
helm upgrade --install karpenter karpenter/karpenter \
  --namespace karpenter \
  --create-namespace \
  --set serviceAccount.annotations."eks.amazonaws.com/role-arn"=arn:aws:iam::ACCOUNT_ID:role/karpenter-controller
```

### 2. Spot Instance Yapılandırması

```yaml
apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: otuken3d-spot
spec:
  requirements:
    - key: karpenter.sh/capacity-type
      operator: In
      values: ["spot"]
    - key: node.kubernetes.io/instance-type
      operator: In
      values: ["t3.large", "t3.xlarge"]
```

### 3. Spot Instance Monitoring

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: spot-monitor
spec:
  selector:
    matchLabels:
      app: spot-termination-handler
  endpoints:
  - port: metrics
```

## Resource Yönetimi

### 1. Resource Quotas

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: otuken3d-quota
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
```

### 2. Limit Ranges

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: otuken3d-limits
spec:
  limits:
  - type: Container
    default:
      cpu: 500m
      memory: 512Mi
    defaultRequest:
      cpu: 200m
      memory: 256Mi
```

### 3. Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: otuken3d-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: otuken3d
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Storage Optimizasyonu

### 1. Storage Classes

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: otuken3d-standard
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
```

### 2. Volume Lifecycle

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: otuken3d-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: otuken3d-standard
```

### 3. Backup Storage

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

## Network Optimizasyonu

### 1. CDN Yapılandırması

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: otuken3d-cdn
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/proxy-buffering: "on"
    nginx.ingress.kubernetes.io/proxy-buffer-size: "128k"
```

### 2. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: otuken3d-network-policy
spec:
  podSelector:
    matchLabels:
      app: otuken3d
  policyTypes:
  - Ingress
  - Egress
```

## Maliyet İzleme

### 1. Cost Explorer Entegrasyonu

```yaml
apiVersion: aws.upbound.io/v1beta1
kind: CostExplorer
metadata:
  name: otuken3d-cost-explorer
spec:
  forProvider:
    region: eu-central-1
    timeUnit: MONTHLY
    metrics:
      - BlendedCost
      - UnblendedCost
```

### 2. Maliyet Alarmları

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: cost-alerts
spec:
  groups:
  - name: costs
    rules:
    - alert: HighCostSpike
      expr: aws_billing_estimated_charges > 1000
      for: 1h
```

## Best Practices

### 1. Instance Seçimi
- Workload analizi
- Instance ailesi optimizasyonu
- Rezervasyon stratejisi

### 2. Resource Planlama
- Capacity planning
- Scaling politikaları
- Resource request/limit optimizasyonu

### 3. Storage Yönetimi
- Lifecycle policies
- Snapshot stratejisi
- Storage class seçimi

### 4. Network Optimizasyonu
- CDN kullanımı
- Cache stratejisi
- Bandwidth optimizasyonu

## Monitoring ve Raporlama

### 1. Maliyet Metrikleri
- Günlük/Aylık harcama
- Resource kullanım oranları
- Spot savings

### 2. Optimizasyon Önerileri
- Instance right-sizing
- Reserved instance coverage
- Unused resource tespiti

### 3. Raporlama
- Maliyet dağılımı
- Trend analizi
- ROI hesaplaması 