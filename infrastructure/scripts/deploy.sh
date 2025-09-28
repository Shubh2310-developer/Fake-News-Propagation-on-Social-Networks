#!/bin/bash

# Kubernetes Deployment Script for Fake News Game Theory Platform
# Usage: ./deploy.sh [staging|production] [--dry-run]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
DRY_RUN=false
VERBOSE=false
FORCE=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"
KUBE_DIR="$INFRA_DIR/kubernetes"

# Function to print colored output
log() {
    local color=$1
    shift
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] $*${NC}"
}

info() { log "$BLUE" "$@"; }
success() { log "$GREEN" "$@"; }
warn() { log "$YELLOW" "$@"; }
error() { log "$RED" "$@"; }

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENT:
    staging         Deploy to staging environment
    production      Deploy to production environment

OPTIONS:
    --dry-run       Show what would be deployed without making changes
    --verbose       Enable verbose output
    --force         Force deployment without confirmation prompts
    --help          Show this help message

Examples:
    $0 staging                  # Deploy to staging
    $0 production --dry-run     # Preview production deployment
    $0 staging --force          # Deploy to staging without prompts

Environment Variables:
    KUBECONFIG                  Path to kubernetes config file
    DOCKER_REGISTRY            Docker registry URL (default: docker.io)
    IMAGE_TAG                   Docker image tag (default: latest)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                error "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done

    if [[ -z "$ENVIRONMENT" ]]; then
        error "Environment must be specified (staging or production)"
        usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
        exit 1
    fi

    # Check if kustomize is installed
    if ! command -v kustomize &> /dev/null; then
        error "kustomize is not installed or not in PATH"
        exit 1
    fi

    # Check kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        error "Please check your KUBECONFIG and cluster connection"
        exit 1
    fi

    # Check if we can access the target namespace
    local namespace="fakenews-$ENVIRONMENT"
    if ! kubectl get namespace "$namespace" &> /dev/null; then
        warn "Namespace '$namespace' does not exist. It will be created."
    fi

    success "Prerequisites check passed"
}

# Confirm deployment
confirm_deployment() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi

    local cluster_info
    cluster_info=$(kubectl cluster-info | head -n 1)

    warn "You are about to deploy to $ENVIRONMENT environment"
    warn "Cluster: $cluster_info"
    warn "Namespace: fakenews-$ENVIRONMENT"

    if [[ "$ENVIRONMENT" == "production" ]]; then
        error "⚠️  PRODUCTION DEPLOYMENT ⚠️"
        error "This will deploy to the production environment!"
        echo
    fi

    read -p "Are you sure you want to continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Deployment cancelled"
        exit 0
    fi
}

# Deploy function
deploy() {
    local env_dir="$KUBE_DIR/$ENVIRONMENT"
    local base_dir="$KUBE_DIR/base"

    info "Starting deployment to $ENVIRONMENT environment..."

    # Check if environment directory exists
    if [[ ! -d "$env_dir" ]]; then
        error "Environment directory '$env_dir' does not exist"
        exit 1
    fi

    # Build kustomization
    info "Building Kubernetes manifests..."
    local manifest_file="/tmp/k8s-manifests-$ENVIRONMENT.yaml"

    if [[ "$VERBOSE" == "true" ]]; then
        kustomize build "$env_dir" | tee "$manifest_file"
    else
        kustomize build "$env_dir" > "$manifest_file"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run mode - showing what would be deployed:"
        echo "----------------------------------------"
        cat "$manifest_file"
        echo "----------------------------------------"
        info "Dry run completed. No changes were made."
        return 0
    fi

    # Apply the manifests
    info "Applying Kubernetes manifests..."
    kubectl apply -f "$manifest_file"

    # Wait for deployments to be ready
    info "Waiting for deployments to be ready..."
    local namespace="fakenews-$ENVIRONMENT"

    # Get all deployments in the namespace
    local deployments
    deployments=$(kubectl get deployments -n "$namespace" -o name 2>/dev/null || echo "")

    if [[ -n "$deployments" ]]; then
        for deployment in $deployments; do
            info "Waiting for $deployment to be ready..."
            kubectl wait --for=condition=available --timeout=300s -n "$namespace" "$deployment"
        done
    fi

    success "Deployment completed successfully!"

    # Show deployment status
    show_status
}

# Show deployment status
show_status() {
    local namespace="fakenews-$ENVIRONMENT"

    info "Deployment Status for $ENVIRONMENT:"
    echo "=================================="

    # Show deployments
    echo "📦 Deployments:"
    kubectl get deployments -n "$namespace" -o wide 2>/dev/null || echo "No deployments found"

    echo
    echo "🔗 Services:"
    kubectl get services -n "$namespace" -o wide 2>/dev/null || echo "No services found"

    echo
    echo "🌐 Ingresses:"
    kubectl get ingresses -n "$namespace" -o wide 2>/dev/null || echo "No ingresses found"

    echo
    echo "💾 Persistent Volume Claims:"
    kubectl get pvc -n "$namespace" -o wide 2>/dev/null || echo "No PVCs found"

    # Show pods with status
    echo
    echo "🚀 Pods:"
    kubectl get pods -n "$namespace" -o wide 2>/dev/null || echo "No pods found"

    # Show any unhealthy pods
    local unhealthy_pods
    unhealthy_pods=$(kubectl get pods -n "$namespace" --field-selector=status.phase!=Running -o name 2>/dev/null || echo "")

    if [[ -n "$unhealthy_pods" ]]; then
        warn "⚠️  Unhealthy pods detected:"
        for pod in $unhealthy_pods; do
            echo "  - $pod"
        done
    fi

    # Show endpoints
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo
        echo "🌍 External Endpoints:"
        echo "  Frontend: https://fake-news-platform.com"
        echo "  API: https://api.fake-news-platform.com"
        echo "  Admin: https://admin.fake-news-platform.com"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        echo
        echo "🌍 External Endpoints:"
        echo "  Frontend: https://staging.fake-news-platform.com"
        echo "  API: https://api-staging.fake-news-platform.com"
    fi
}

# Rollback function
rollback() {
    local namespace="fakenews-$ENVIRONMENT"

    warn "Rolling back deployments in $namespace..."

    # Get all deployments and rollback
    local deployments
    deployments=$(kubectl get deployments -n "$namespace" -o name 2>/dev/null || echo "")

    if [[ -n "$deployments" ]]; then
        for deployment in $deployments; do
            info "Rolling back $deployment..."
            kubectl rollout undo -n "$namespace" "$deployment"
        done

        # Wait for rollback to complete
        for deployment in $deployments; do
            kubectl rollout status -n "$namespace" "$deployment" --timeout=300s
        done

        success "Rollback completed"
    else
        warn "No deployments found to rollback"
    fi
}

# Cleanup function
cleanup() {
    info "Cleaning up temporary files..."
    rm -f "/tmp/k8s-manifests-$ENVIRONMENT.yaml"
}

# Main function
main() {
    # Set up cleanup trap
    trap cleanup EXIT

    # Parse arguments
    parse_args "$@"

    # Enable verbose mode if requested
    if [[ "$VERBOSE" == "true" ]]; then
        set -x
    fi

    # Check prerequisites
    check_prerequisites

    # Confirm deployment
    confirm_deployment

    # Deploy
    deploy

    success "🎉 Deployment to $ENVIRONMENT completed successfully!"

    if [[ "$ENVIRONMENT" == "production" ]]; then
        info "🚀 Your fake news detection platform is now live!"
        info "📊 Monitor the deployment at: https://admin.fake-news-platform.com/grafana"
    fi
}

# Run main function
main "$@"