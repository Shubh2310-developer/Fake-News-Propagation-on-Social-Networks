# Monitoring Setup - Fake News Game Theory Platform

This directory contains a comprehensive monitoring and observability stack using Prometheus for metrics collection and Grafana for visualization and alerting. The setup provides deep insights into application health, performance, and business metrics.

## 📁 Directory Structure

```
monitoring/
├── prometheus.yml                  # Prometheus main configuration
├── docker-compose.yml             # Local development stack
├── rules/                          # Alerting rules
│   └── application-alerts.yml      # Application-specific alerts
├── alertmanager/                   # Alertmanager configuration
│   └── alertmanager.yml           # Alert routing and notifications
└── grafana/                        # Grafana configuration
    ├── datasources/                # Data source definitions
    │   └── prometheus.yml          # Prometheus datasource config
    ├── provisioning/               # Dashboard provisioning
    │   └── dashboards.yml          # Dashboard discovery config
    └── dashboards/                 # Dashboard definitions
        ├── application/            # Application dashboards
        │   └── application-health.json
        ├── business/               # Business metrics dashboards
        │   └── business-metrics.json
        ├── kubernetes/             # Infrastructure dashboards
        │   └── cluster-overview.json
        ├── infrastructure/         # System dashboards
        └── sla/                    # SLA and alerting dashboards
```

## 🎯 Monitoring Philosophy

### Golden Signals (SRE Methodology)

The monitoring setup focuses on the four golden signals:

1. **Latency** - Response time of requests
2. **Traffic** - Number of requests per second
3. **Errors** - Rate of failed requests
4. **Saturation** - Resource utilization (CPU, memory, disk)

### Business Metrics

Beyond infrastructure monitoring, we track business-critical KPIs:

- ML model accuracy and performance
- Fake news detection rates
- Game theory simulation success rates
- User engagement and session metrics
- Data processing pipeline health

## 🚀 Quick Start

### Local Development with Docker Compose

1. **Start the monitoring stack:**
   ```bash
   docker-compose up -d
   ```

2. **Access the services:**
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000 (admin/admin123)
   - **Alertmanager**: http://localhost:9093

3. **Stop the stack:**
   ```bash
   docker-compose down
   ```

### Kubernetes Deployment

1. **Deploy with existing Kubernetes manifests:**
   ```bash
   kubectl apply -f ../kubernetes/monitoring/
   ```

2. **Access via port-forward:**
   ```bash
   # Prometheus
   kubectl port-forward -n fakenews-monitoring svc/prometheus-server 9090:9090

   # Grafana
   kubectl port-forward -n fakenews-monitoring svc/grafana-service 3000:3000
   ```

## 📊 Dashboard Overview

### Application Health Dashboard

**Location:** `grafana/dashboards/application/application-health.json`

**Purpose:** Monitors the golden signals across all application services

**Key Panels:**
- **Golden Signals Overview** - High-level KPI summary
- **Traffic** - Request rates by service and status code
- **Latency** - Response time percentiles (50th, 95th, 99th)
- **Error Rate** - 4xx and 5xx error percentages
- **Saturation** - CPU and memory usage by pod

**Variables:**
- `$namespace` - Kubernetes namespace filter
- `$service` - Service name filter
- `$interval` - Metrics aggregation interval

### Business Metrics Dashboard

**Location:** `grafana/dashboards/business/business-metrics.json`

**Purpose:** Tracks application-specific KPIs and business value

**Key Panels:**
- **Business KPIs Overview** - Total predictions, simulations, user sessions
- **ML Model Performance** - Accuracy, precision, recall, F1 score
- **Classifier Distribution** - Fake vs real news detection rates
- **Game Theory Simulations** - Simulation rates and player engagement
- **User Engagement** - Session metrics and interaction rates
- **Model Training & Deployment** - ML pipeline health
- **Data Processing Pipeline** - Queue sizes and processing times
- **Business Value Metrics** - Platform reliability and trust scores

### Kubernetes Cluster Overview

**Location:** `grafana/dashboards/kubernetes/cluster-overview.json`

**Purpose:** Infrastructure and cluster-level monitoring

**Key Panels:**
- **Cluster Overview** - Node count, pod status, deployments
- **Node Resource Usage** - CPU, memory, disk usage per node
- **Pod Status Distribution** - Running, pending, failed pods
- **Network I/O** - Network traffic by node and interface
- **Storage I/O** - Disk read/write operations
- **Container Resources** - Per-container CPU and memory usage
- **Kubernetes Events** - Recent cluster events table
- **Node Information** - Node labels, capacity, and specifications

## 🚨 Alerting Strategy

### Alert Severity Levels

1. **Critical** - Immediate action required, system impact
2. **Warning** - Attention needed, potential issues
3. **Info** - Informational, trend monitoring

### Alert Categories

#### Golden Signals Alerts
- **HighErrorRate** - 5xx error rate > 5% (Warning), > 10% (Critical)
- **HighLatency** - 95th percentile > 500ms (Warning), > 2000ms (Critical)
- **LowTraffic** - Unusually low request rates
- **ServiceDown** - Service unavailable

#### Resource Alerts
- **HighMemoryUsage** - Memory usage > 85% (Warning), > 95% (Critical)
- **HighCPUUsage** - CPU usage > 80% for 10 minutes
- **PodCrashLooping** - Frequent pod restarts

#### Business Logic Alerts
- **LowModelAccuracy** - ML model accuracy < 75% (Warning), < 60% (Critical)
- **HighPredictionErrors** - Classification errors > 0.1/sec
- **HighSimulationFailures** - Simulation failure rate > 10%
- **NoPredictions** - No predictions for 10 minutes

#### Data Processing Alerts
- **HighQueueSize** - Processing queues > 1000 items
- **SlowDataProcessing** - Processing time > 30 seconds

### Notification Routing

Alerts are routed based on:
- **Severity** - Critical alerts go to on-call rotation
- **Team** - Platform, Data Science, Infrastructure, Product teams
- **Environment** - Production gets immediate attention, staging is batched
- **Component** - Different notification channels per component

### Notification Channels

- **Slack** - Real-time team notifications
- **Email** - Detailed alert information
- **PagerDuty** - Critical production alerts only
- **Webhook** - Integration with incident management

## 🔧 Configuration

### Prometheus Configuration

**File:** `prometheus.yml`

**Key Features:**
- **Service Discovery** - Kubernetes-native service discovery
- **Job-Based Organization** - Logical grouping of targets
- **Metric Relabeling** - Consistent labeling across services
- **Multiple Scrape Intervals** - Optimized for different metric types

**Scrape Jobs:**
- `fake-news-backend` - Application API metrics (10s interval)
- `fake-news-frontend` - Frontend metrics (15s interval)
- `kubelet` - Node metrics (30s interval)
- `cadvisor` - Container metrics (30s interval)
- `business-metrics` - Custom business KPIs (10s interval)
- `ml-model-metrics` - Model performance (30s interval)

### Grafana Provisioning

**Datasources:** `grafana/datasources/prometheus.yml`
- Prometheus (primary metrics)
- Alertmanager (alert management)
- CloudWatch (AWS native metrics)
- TestData (development/testing)

**Dashboards:** `grafana/provisioning/dashboards.yml`
- Automatic dashboard discovery
- Organized into folders by category
- Hot-reload on file changes

### Alertmanager Configuration

**File:** `alertmanager/alertmanager.yml`

**Features:**
- **Route Tree** - Hierarchical alert routing
- **Inhibition Rules** - Suppress redundant alerts
- **Template System** - Consistent notification formatting
- **Multi-Channel Delivery** - Slack, email, PagerDuty

## 📈 Metrics Reference

### Application Metrics

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Latency percentiles
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Model accuracy
model_accuracy{model_type="fake_news_classifier"}
```

### Business Metrics

```promql
# Prediction volume
rate(classifier_predictions_total[5m])

# Fake news detection rate
rate(classifier_predictions_total{prediction="fake"}[5m]) / rate(classifier_predictions_total[5m])

# Simulation success rate
rate(successful_simulations_total[5m]) / rate(simulations_run_total[5m])

# Active users
sum(active_users)
```

### Infrastructure Metrics

```promql
# CPU usage
rate(container_cpu_usage_seconds_total[5m])

# Memory usage
container_memory_working_set_bytes / container_spec_memory_limit_bytes

# Pod restart rate
rate(kube_pod_container_status_restarts_total[5m])
```

## 🛠️ Customization

### Adding New Dashboards

1. **Create JSON file** in appropriate folder:
   ```bash
   grafana/dashboards/application/my-dashboard.json
   ```

2. **Dashboard will auto-load** via provisioning

3. **Use variables** for reusability:
   ```json
   "templating": {
     "list": [
       {
         "name": "service",
         "query": "label_values(up, service)"
       }
     ]
   }
   ```

### Adding New Alerts

1. **Edit alert rules:**
   ```yaml
   # rules/application-alerts.yml
   - alert: MyCustomAlert
     expr: my_metric > 100
     for: 5m
     labels:
       severity: warning
     annotations:
       summary: "Custom alert fired"
   ```

2. **Prometheus auto-reloads** rules

### Adding New Metrics

1. **Application instrumentation:**
   ```python
   from prometheus_client import Counter, Histogram

   REQUEST_COUNT = Counter('app_requests_total', 'Total requests')
   REQUEST_LATENCY = Histogram('app_request_duration_seconds', 'Request latency')
   ```

2. **Update Prometheus config** if needed for new targets

## 🚀 Production Considerations

### Resource Requirements

- **Prometheus**: 4GB RAM, 100GB SSD (30 days retention)
- **Grafana**: 1GB RAM, 10GB storage
- **Alertmanager**: 512MB RAM, 1GB storage

### High Availability

- **Prometheus** - Use federation or Thanos for scale
- **Grafana** - External database for clustering
- **Alertmanager** - Cluster mode for reliability

### Security

- **Authentication** - LDAP/OAuth integration
- **Authorization** - Role-based access control
- **Network** - TLS encryption, private networks
- **Secrets** - External secret management

### Backup Strategy

- **Prometheus** - Snapshot automation
- **Grafana** - Database backup + dashboard export
- **Configuration** - Version control (this repo)

## 🔍 Troubleshooting

### Common Issues

1. **Prometheus targets down:**
   ```bash
   # Check service discovery
   kubectl get endpoints -n fakenews-production

   # Verify pod labels
   kubectl get pods --show-labels
   ```

2. **Grafana dashboards not loading:**
   ```bash
   # Check provisioning logs
   kubectl logs -n fakenews-monitoring deployment/grafana
   ```

3. **Alerts not firing:**
   ```bash
   # Check rule evaluation
   http://prometheus:9090/rules

   # Verify alertmanager config
   http://alertmanager:9093/#/status
   ```

### Debug Queries

```promql
# Check scrape success
up{job="fake-news-backend"}

# Verify metric availability
{__name__=~"http_.*"}

# Alert rule evaluation
ALERTS{alertname="HighErrorRate"}
```

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Best Practices](https://grafana.com/docs/grafana/latest/best-practices/)

---

**⚡ Quick Commands:**

```bash
# Start local stack
docker-compose up -d

# View metrics
curl http://localhost:9090/api/v1/query?query=up

# Test alerts
curl -X POST http://localhost:9093/api/v1/alerts

# Export dashboard
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:3000/api/dashboards/uid/dashboard-uid
```