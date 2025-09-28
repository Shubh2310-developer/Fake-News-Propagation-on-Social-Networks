# Infrastructure as Code (IaC) for Fake News Game Theory Platform

This directory contains the complete Kubernetes infrastructure configuration for deploying the fake news detection and game theory simulation platform in a scalable, production-ready manner.

## 📁 Directory Structure

```
infrastructure/
├── kubernetes/
│   ├── base/                   # Base Kubernetes manifests
│   │   ├── namespace.yaml      # Namespace definitions
│   │   ├── backend-deployment.yaml
│   │   ├── frontend-deployment.yaml
│   │   ├── postgres-deployment.yaml
│   │   ├── redis-deployment.yaml
│   │   ├── service.yaml        # Service definitions
│   │   ├── ingress.yaml        # Ingress configurations
│   │   ├── configmap.yaml      # Configuration data
│   │   ├── secrets.yaml        # Secret templates
│   │   ├── persistent-volumes.yaml
│   │   └── hpa.yaml           # Horizontal Pod Autoscaler
│   ├── staging/               # Staging environment overlays
│   │   ├── kustomization.yaml
│   │   └── deployment-patch.yaml
│   ├── production/            # Production environment overlays
│   │   ├── kustomization.yaml
│   │   └── deployment-patch.yaml
│   └── monitoring/            # Monitoring components
│       ├── prometheus.yaml
│       └── grafana.yaml
├── scripts/
│   ├── setup-cluster.sh       # Initial cluster setup
│   └── deploy.sh              # Deployment script
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster**: EKS, GKE, AKS, or local (kind/minikube)
2. **kubectl**: Kubernetes CLI tool
3. **kustomize**: Configuration management tool
4. **helm**: Package manager for Kubernetes (for cluster setup)

### 1. Initial Cluster Setup

```bash
# Set up cluster with required components
./scripts/setup-cluster.sh

# This installs:
# - NGINX Ingress Controller
# - cert-manager for TLS certificates
# - Metrics server for HPA
# - Prometheus Operator (optional)
# - Application namespaces
# - RBAC configuration
```

### 2. Deploy to Staging

```bash
# Deploy to staging environment
./scripts/deploy.sh staging

# With dry-run to preview changes
./scripts/deploy.sh staging --dry-run
```

### 3. Deploy to Production

```bash
# Deploy to production (with confirmation prompt)
./scripts/deploy.sh production

# Force deployment without prompts
./scripts/deploy.sh production --force
```

## 🏗️ Architecture Overview

### Environment Isolation

The infrastructure uses Kubernetes namespaces for environment isolation:

- **fakenews-staging**: Staging environment
- **fakenews-production**: Production environment
- **fakenews-monitoring**: Shared monitoring infrastructure

### Components

#### Application Services

1. **Backend API** (`fake-news-backend`)
   - FastAPI application with ML models
   - Horizontal Pod Autoscaler for scaling
   - Health checks and readiness probes
   - Persistent storage for models and data

2. **Frontend** (`fake-news-frontend`)
   - Next.js React application
   - Served through NGINX Ingress
   - Auto-scaling based on CPU/memory

3. **Database** (`postgres`)
   - PostgreSQL with persistent storage
   - Optimized for production workloads
   - Automated backups (via snapshots)

4. **Cache** (`redis`)
   - Redis for session and API caching
   - Persistent storage for durability

#### Infrastructure Services

1. **NGINX Ingress Controller**
   - External traffic routing
   - TLS termination with Let's Encrypt
   - Rate limiting and security headers

2. **cert-manager**
   - Automatic TLS certificate management
   - Let's Encrypt integration

3. **Monitoring Stack**
   - Prometheus for metrics collection
   - Grafana for visualization
   - Custom dashboards for the platform

### Scaling Strategy

#### Horizontal Pod Autoscaler (HPA)

- **Backend**: Scales 2-20 pods based on CPU (70%) and memory (80%)
- **Frontend**: Scales 1-10 pods based on CPU (70%) and memory (80%)

#### Vertical Pod Autoscaler (VPA)

- **Database**: Automatic resource adjustment for PostgreSQL

#### Resource Limits

**Staging Environment**:
- Backend: 256Mi-1Gi memory, 125m-500m CPU
- Frontend: 128Mi-512Mi memory, 50m-250m CPU

**Production Environment**:
- Backend: 1Gi-4Gi memory, 500m-2000m CPU
- Frontend: 512Mi-2Gi memory, 250m-1000m CPU

## 🔧 Configuration Management

### Kustomize Structure

The infrastructure uses Kustomize for configuration management:

- **Base**: Common manifests shared across environments
- **Overlays**: Environment-specific patches and configurations

### Environment Variables

#### Backend Configuration

```yaml
# Production settings
environment: "production"
log-level: "INFO"
api-workers: "4"

# Staging settings
environment: "staging"
log-level: "DEBUG"
api-workers: "2"
```

#### Frontend Configuration

```yaml
# API endpoints
api-url: "https://api.fake-news-platform.com"          # Production
api-url: "https://api-staging.fake-news-platform.com"  # Staging
```

### Secrets Management

🔐 **Important**: The `secrets.yaml` file contains template secrets with base64-encoded dummy values. For production:

1. Use proper secret management (HashiCorp Vault, AWS Secrets Manager)
2. Never commit real secrets to version control
3. Rotate secrets regularly

#### Creating Real Secrets

```bash
# Database URL
kubectl create secret generic backend-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --namespace=fakenews-production

# TLS certificates (managed by cert-manager)
# Will be automatically created when ingress is deployed
```

## 🌐 Network Configuration

### Ingress Routes

#### Production Domains

- **Frontend**: `https://fake-news-platform.com`
- **API**: `https://api.fake-news-platform.com`
- **Admin**: `https://admin.fake-news-platform.com` (IP-restricted)

#### Staging Domains

- **Frontend**: `https://staging.fake-news-platform.com`
- **API**: `https://api-staging.fake-news-platform.com`

### TLS Configuration

- Automatic certificate provisioning via Let's Encrypt
- TLS 1.2 and 1.3 support
- HSTS headers for security

### Security Features

- Rate limiting (5 requests/second, 10 connections)
- CORS configuration
- Security headers (CSP, X-Frame-Options, etc.)
- IP whitelisting for admin access

## 💾 Storage Configuration

### Persistent Volumes

1. **PostgreSQL**: 100Gi SSD storage
2. **Redis**: 20Gi SSD storage
3. **ML Models**: 50Gi standard storage (ReadWriteMany)
4. **Training Data**: 200Gi standard storage (ReadWriteMany)
5. **Backups**: 500Gi standard storage

### Storage Classes

- **fast-ssd**: High-performance SSD for databases
- **standard-storage**: Standard storage for data and backups

### Backup Strategy

- Volume snapshots for point-in-time recovery
- Cross-region replication for disaster recovery
- Automated backup scheduling

## 📊 Monitoring and Observability

### Prometheus Metrics

Custom metrics for the platform:

- `http_requests_total`: API request counts
- `classification_accuracy`: Model accuracy metrics
- `active_simulations_total`: Running simulations
- `database_connections`: PostgreSQL connections

### Grafana Dashboards

Pre-configured dashboards for:

- Application performance monitoring
- Infrastructure resource usage
- Business metrics (classifications, simulations)
- Error rates and latency

### Alerting Rules

- High error rates (>10% 5xx responses)
- High latency (>500ms 95th percentile)
- Pod crash looping
- Resource exhaustion

### Log Aggregation

- Centralized logging with structured JSON logs
- Log retention policies
- Error tracking and alerting

## 🔄 CI/CD Integration

### GitOps Workflow

1. **Code Changes**: Push to Git repository
2. **Build**: CI pipeline builds Docker images
3. **Test**: Automated testing in staging
4. **Deploy**: Automatic deployment via ArgoCD/Flux

### Deployment Strategies

#### Rolling Updates

- Zero-downtime deployments
- Gradual traffic shifting
- Automatic rollback on failure

#### Blue-Green Deployments

```bash
# Deploy new version to blue environment
./scripts/deploy.sh production --blue

# Switch traffic after validation
./scripts/switch-traffic.sh blue
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl get pods -n fakenews-production

# Check pod logs
kubectl logs -f deployment/fake-news-backend -n fakenews-production

# Describe pod for events
kubectl describe pod <pod-name> -n fakenews-production
```

#### 2. Ingress Not Working

```bash
# Check ingress status
kubectl get ingress -n fakenews-production

# Check NGINX controller logs
kubectl logs -f deployment/nginx-ingress-controller -n ingress-nginx

# Verify certificates
kubectl get certificates -n fakenews-production
```

#### 3. Database Connection Issues

```bash
# Check PostgreSQL status
kubectl get pods -l app=postgres -n fakenews-production

# Test database connection
kubectl exec -it deployment/postgres -n fakenews-production -- psql -U app_user -d fake_news_db -c "SELECT 1;"
```

### Health Checks

```bash
# Backend health
curl -k https://api.fake-news-platform.com/api/health

# Frontend health
curl -k https://fake-news-platform.com/api/health

# Database health
kubectl exec deployment/postgres -n fakenews-production -- pg_isready
```

## 🔧 Maintenance

### Updating Dependencies

```bash
# Update Docker images
./scripts/deploy.sh production --image-tag=v1.3.0

# Update Kubernetes components
helm upgrade nginx-ingress ingress-nginx/ingress-nginx -n ingress-nginx
helm upgrade cert-manager jetstack/cert-manager -n cert-manager
```

### Scaling Operations

```bash
# Manual scaling
kubectl scale deployment fake-news-backend --replicas=10 -n fakenews-production

# Update HPA limits
kubectl patch hpa fake-news-backend-hpa -n fakenews-production -p '{"spec":{"maxReplicas":30}}'
```

### Database Maintenance

```bash
# Create database backup
kubectl exec deployment/postgres -n fakenews-production -- pg_dump fake_news_db > backup.sql

# Create volume snapshot
kubectl create volumesnapshot postgres-snapshot --claim=postgres-pvc -n fakenews-production
```

## 🔐 Security Best Practices

### Implemented Security Measures

1. **Network Policies**: Restrict inter-pod communication
2. **RBAC**: Least-privilege access control
3. **Pod Security Standards**: Enforce security contexts
4. **Secret Management**: Encrypted secrets with rotation
5. **Image Scanning**: Vulnerability scanning in CI/CD
6. **Ingress Security**: Rate limiting, IP whitelisting

### Security Checklist

- [ ] All secrets are properly encrypted and rotated
- [ ] Network policies are in place
- [ ] RBAC is configured with minimal permissions
- [ ] Images are scanned for vulnerabilities
- [ ] TLS is enforced everywhere
- [ ] Monitoring and alerting are active
- [ ] Backup and disaster recovery are tested

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kustomize Documentation](https://kustomize.io/)
- [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/)
- [cert-manager Documentation](https://cert-manager.io/docs/)
- [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator)

## 🤝 Contributing

When making changes to the infrastructure:

1. Test changes in staging first
2. Use `--dry-run` to preview changes
3. Update documentation for new components
4. Follow the principle of least privilege
5. Ensure backwards compatibility

## 📞 Support

For infrastructure issues:

1. Check the troubleshooting section
2. Review logs and events
3. Consult monitoring dashboards
4. Create an issue with detailed information

---

**⚠️ Important**: Always test infrastructure changes in staging before applying to production. The deployment scripts include safety checks, but human review is essential for production deployments.