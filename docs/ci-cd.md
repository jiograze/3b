# Ötüken3D CI/CD Pipeline Dokümantasyonu

## İçindekiler
- [Genel Bakış](#genel-bakış)
- [ArgoCD Entegrasyonu](#argocd-entegrasyonu)
- [GitOps Workflow](#gitops-workflow)
- [Deployment Stratejileri](#deployment-stratejileri)
- [Monitoring ve Alerting](#monitoring-ve-alerting)

## Genel Bakış

Ötüken3D projesi, sürekli entegrasyon ve sürekli dağıtım (CI/CD) için ArgoCD kullanmaktadır. Bu dokümantasyon, CI/CD pipeline'ının kurulumunu, yapılandırmasını ve kullanımını detaylı olarak açıklamaktadır.

## ArgoCD Entegrasyonu

### Kurulum

1. ArgoCD kurulumu:
```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

2. ArgoCD CLI kurulumu:
```bash
curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
sudo install -m 555 argocd-linux-amd64 /usr/local/bin/argocd
rm argocd-linux-amd64
```

### Yapılandırma

1. Application tanımı:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: otuken3d
  namespace: argocd
spec:
  project: default
  source:
    repoURL: 'https://github.com/otuken3d/otuken3d.git'
    path: deployment/kubernetes
    targetRevision: HEAD
```

2. Project tanımı:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: otuken3d
  namespace: argocd
spec:
  description: Otuken3D Project
```

## GitOps Workflow

### Branch Stratejisi

1. **Main Branch**: Production ortamı
2. **Staging Branch**: Staging ortamı
3. **Feature Branches**: Yeni özellikler için

### Deployment Süreci

1. Feature branch oluşturma:
```bash
git checkout -b feature/yeni-ozellik
```

2. Değişiklikleri commit etme:
```bash
git add .
git commit -m "feat: yeni özellik eklendi"
```

3. Pull request oluşturma ve review süreci

4. Staging'e deploy:
```bash
argocd app sync otuken3d-staging
```

5. Production'a deploy:
```bash
argocd app sync otuken3d-production
```

## Deployment Stratejileri

### 1. Canary Deployment

```yaml
spec:
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 1h}
      - setWeight: 40
      - pause: {duration: 1h}
      - setWeight: 60
      - pause: {duration: 1h}
      - setWeight: 80
      - pause: {duration: 1h}
```

### 2. Blue/Green Deployment

```yaml
spec:
  strategy:
    blueGreen:
      activeService: otuken3d-active
      previewService: otuken3d-preview
      autoPromotionEnabled: false
```

### 3. Rolling Update

```yaml
spec:
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
```

## Monitoring ve Alerting

### 1. Notifications

```yaml
apiVersion: notifications.argoproj.io/v1alpha1
kind: NotificationConfig
metadata:
  name: otuken3d-notifications
spec:
  templates:
    - name: deployment-status
      body: |
        Application: {{.app.metadata.name}}
        Status: {{.app.status.sync.status}}
```

### 2. Metrikler

- Deployment süresi
- Başarı oranı
- Rollback sayısı
- Servis kesinti süresi

### 3. Alerting Kuralları

1. **Deployment Başarısızlığı**:
```yaml
triggers:
  - name: deployment-failed
    condition: app.status.sync.status == 'Failed'
    template: deployment-status
```

2. **Senkronizasyon Gecikmesi**:
```yaml
triggers:
  - name: sync-delayed
    condition: app.status.sync.status == 'OutOfSync'
    template: sync-status
```

## Rollback Stratejisi

### 1. Manuel Rollback

```bash
argocd app history otuken3d
argocd app rollback otuken3d --to-revision=2
```

### 2. Otomatik Rollback

```yaml
spec:
  rollback:
    enabled: true
    limit: 5
    steps:
    - setWeight: 0
    - pause: {duration: 5m}
```

### 3. Rollback Tetikleyicileri

- Health check başarısızlığı
- Metrik anomalileri
- Error rate artışı

## Best Practices

1. **Versiyonlama**:
- Semantic versioning kullanımı
- Git tag'leri ile sürüm takibi

2. **Güvenlik**:
- Secrets yönetimi
- RBAC yapılandırması
- Image scanning

3. **Testing**:
- Unit testler
- Integration testler
- End-to-end testler

4. **Monitoring**:
- Prometheus metrics
- Grafana dashboards
- Alert yapılandırması 