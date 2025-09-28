#!/bin/bash

# Kubernetes Cluster Setup Script for Fake News Game Theory Platform
# This script sets up the initial cluster requirements including NGINX Ingress,
# cert-manager, metrics server, and other essential components.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    local color=$1
    shift
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}"
}

info() { log "$BLUE" "$@"; }
success() { log "$GREEN" "$@"; }
warn() { log "$YELLOW" "$@"; }
error() { log "$RED" "$@"; }

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
        exit 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        error "helm is not installed"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check cluster permissions
    if ! kubectl auth can-i create namespace &> /dev/null; then
        error "Insufficient permissions to create namespaces"
        exit 1
    fi

    success "Prerequisites check passed"
}

# Install NGINX Ingress Controller
install_nginx_ingress() {
    info "Installing NGINX Ingress Controller..."

    # Add NGINX Ingress Helm repository
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update

    # Check if already installed
    if helm list -n ingress-nginx | grep -q nginx-ingress; then
        warn "NGINX Ingress already installed, upgrading..."
        helm upgrade nginx-ingress ingress-nginx/ingress-nginx \
            --namespace ingress-nginx \
            --set controller.service.type=LoadBalancer \
            --set controller.metrics.enabled=true \
            --set controller.podAnnotations."prometheus\.io/scrape"="true" \
            --set controller.podAnnotations."prometheus\.io/port"="10254"
    else
        # Create namespace
        kubectl create namespace ingress-nginx --dry-run=client -o yaml | kubectl apply -f -

        # Install NGINX Ingress
        helm install nginx-ingress ingress-nginx/ingress-nginx \
            --namespace ingress-nginx \
            --set controller.service.type=LoadBalancer \
            --set controller.metrics.enabled=true \
            --set controller.podAnnotations."prometheus\.io/scrape"="true" \
            --set controller.podAnnotations."prometheus\.io/port"="10254"
    fi

    # Wait for the controller to be ready
    info "Waiting for NGINX Ingress Controller to be ready..."
    kubectl wait --namespace ingress-nginx \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/component=controller \
        --timeout=300s

    success "NGINX Ingress Controller installed"
}

# Install cert-manager
install_cert_manager() {
    info "Installing cert-manager..."

    # Add Jetstack Helm repository
    helm repo add jetstack https://charts.jetstack.io
    helm repo update

    # Check if already installed
    if helm list -n cert-manager | grep -q cert-manager; then
        warn "cert-manager already installed, upgrading..."
        helm upgrade cert-manager jetstack/cert-manager \
            --namespace cert-manager \
            --set installCRDs=true \
            --set prometheus.enabled=true
    else
        # Create namespace
        kubectl create namespace cert-manager --dry-run=client -o yaml | kubectl apply -f -

        # Install cert-manager
        helm install cert-manager jetstack/cert-manager \
            --namespace cert-manager \
            --set installCRDs=true \
            --set prometheus.enabled=true
    fi

    # Wait for cert-manager to be ready
    info "Waiting for cert-manager to be ready..."
    kubectl wait --namespace cert-manager \
        --for=condition=ready pod \
        --selector=app.kubernetes.io/instance=cert-manager \
        --timeout=300s

    success "cert-manager installed"
}

# Create ClusterIssuers for Let's Encrypt
create_cluster_issuers() {
    info "Creating Let's Encrypt ClusterIssuers..."

    # Create staging ClusterIssuer
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-staging
spec:
  acme:
    server: https://acme-staging-v02.api.letsencrypt.org/directory
    email: admin@fake-news-platform.com
    privateKeySecretRef:
      name: letsencrypt-staging
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

    # Create production ClusterIssuer
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@fake-news-platform.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

    success "ClusterIssuers created"
}

# Install metrics server
install_metrics_server() {
    info "Installing metrics server..."

    # Check if already installed
    if kubectl get deployment metrics-server -n kube-system &> /dev/null; then
        warn "Metrics server already installed"
    else
        kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

        # Wait for metrics server to be ready
        info "Waiting for metrics server to be ready..."
        kubectl wait --namespace kube-system \
            --for=condition=ready pod \
            --selector=k8s-app=metrics-server \
            --timeout=300s
    fi

    success "Metrics server installed"
}

# Install Prometheus Operator (optional)
install_prometheus_operator() {
    info "Installing Prometheus Operator..."

    # Add Prometheus community Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    # Check if already installed
    if helm list -n monitoring | grep -q prometheus-operator; then
        warn "Prometheus Operator already installed, upgrading..."
        helm upgrade prometheus-operator prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --set prometheus.prometheusSpec.retention=15d \
            --set grafana.adminPassword=admin123
    else
        # Create namespace
        kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

        # Install Prometheus Operator
        helm install prometheus-operator prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --set prometheus.prometheusSpec.retention=15d \
            --set grafana.adminPassword=admin123
    fi

    success "Prometheus Operator installed"
}

# Create namespaces
create_namespaces() {
    info "Creating application namespaces..."

    # Apply namespace configuration
    kubectl apply -f "$(dirname "$(dirname "$0")")/kubernetes/base/namespace.yaml"

    success "Namespaces created"
}

# Set up RBAC
setup_rbac() {
    info "Setting up RBAC..."

    # Create service accounts and roles for the application
    cat <<EOF | kubectl apply -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fake-news-backend
  namespace: fakenews-staging
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fake-news-backend
  namespace: fakenews-production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: fakenews-staging
  name: fake-news-backend-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: fakenews-production
  name: fake-news-backend-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: fake-news-backend-binding
  namespace: fakenews-staging
subjects:
- kind: ServiceAccount
  name: fake-news-backend
  namespace: fakenews-staging
roleRef:
  kind: Role
  name: fake-news-backend-role
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: fake-news-backend-binding
  namespace: fakenews-production
subjects:
- kind: ServiceAccount
  name: fake-news-backend
  namespace: fakenews-production
roleRef:
  kind: Role
  name: fake-news-backend-role
  apiGroup: rbac.authorization.k8s.io
EOF

    success "RBAC configured"
}

# Install storage classes
install_storage_classes() {
    info "Installing storage classes..."

    # Apply storage class configuration
    kubectl apply -f "$(dirname "$(dirname "$0")")/kubernetes/base/persistent-volumes.yaml"

    success "Storage classes configured"
}

# Show cluster info
show_cluster_info() {
    info "Cluster Setup Complete!"
    echo "========================"

    echo "🔧 Installed Components:"
    echo "  ✅ NGINX Ingress Controller"
    echo "  ✅ cert-manager"
    echo "  ✅ Metrics Server"
    echo "  ✅ Prometheus Operator (optional)"
    echo "  ✅ Application Namespaces"
    echo "  ✅ RBAC Configuration"
    echo "  ✅ Storage Classes"

    echo
    echo "📊 Cluster Information:"
    kubectl cluster-info

    echo
    echo "🌐 NGINX Ingress External IP:"
    kubectl get service nginx-ingress-ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
    echo

    echo
    echo "📂 Namespaces:"
    kubectl get namespaces | grep fakenews

    echo
    echo "🔐 ClusterIssuers:"
    kubectl get clusterissuers

    echo
    echo "💾 Storage Classes:"
    kubectl get storageclass

    echo
    echo "Next Steps:"
    echo "1. Update DNS records to point to the NGINX Ingress IP"
    echo "2. Update domain names in ingress configurations"
    echo "3. Run ./deploy.sh staging to deploy the application"
    echo "4. Configure monitoring and alerting"
}

# Main execution
main() {
    info "🚀 Starting Kubernetes cluster setup for Fake News Game Theory Platform"

    check_root
    check_prerequisites

    install_nginx_ingress
    install_cert_manager
    create_cluster_issuers
    install_metrics_server

    # Uncomment to install Prometheus Operator
    # install_prometheus_operator

    create_namespaces
    setup_rbac
    install_storage_classes

    show_cluster_info

    success "🎉 Cluster setup completed successfully!"
}

# Run main function
main "$@"