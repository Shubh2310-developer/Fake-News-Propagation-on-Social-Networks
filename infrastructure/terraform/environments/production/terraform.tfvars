# =============================================================================
# Production Environment Configuration - Fake News Game Theory Platform
# =============================================================================
# This file contains production-specific variable values optimized for
# high availability, performance, and security.

# =============================================================================
# General Configuration
# =============================================================================
project_name = "fake-news-game-theory"
environment  = "production"
cost_center  = "platform-operations"

# =============================================================================
# AWS Configuration
# =============================================================================
aws_region               = "us-west-2"
terraform_state_bucket   = "fake-news-terraform-state-production"
terraform_lock_table     = "terraform-state-lock-production"
az_count                 = 3  # Full 3-AZ deployment for high availability

# =============================================================================
# Network Configuration
# =============================================================================
vpc_cidr = "10.0.0.0/16"

# Production subnets - properly sized for growth
public_subnet_cidrs   = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
private_subnet_cidrs  = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
database_subnet_cidrs = ["10.0.21.0/24", "10.0.22.0/24", "10.0.23.0/24"]

enable_nat_gateway = true  # Required for private subnet internet access
enable_vpn_gateway = true  # Enabled for secure management access

# =============================================================================
# EKS Cluster Configuration
# =============================================================================
cluster_version = "1.28"

cluster_endpoint_config = {
  private_access      = true
  public_access       = true
  public_access_cidrs = [
    "203.0.113.0/24",  # Replace with actual office/management IPs
    "198.51.100.0/24"  # Add additional trusted CIDR blocks
  ]
}

cluster_addons = {
  coredns = {
    version = "v1.10.1-eksbuild.1"
  }
  kube-proxy = {
    version = "v1.28.1-eksbuild.1"
  }
  vpc-cni = {
    version = "v1.13.4-eksbuild.1"
  }
  aws-ebs-csi-driver = {
    version = "v1.21.0-eksbuild.1"
  }
}

# Production node groups - optimized for performance and reliability
node_groups = {
  general = {
    instance_types = ["t3.large", "t3.xlarge"]  # Larger instances for stable performance
    capacity_type  = "ON_DEMAND"                # On-demand for guaranteed availability
    min_size      = 3                           # Minimum for high availability
    max_size      = 20                          # Higher max for scaling
    desired_size  = 6                           # Higher baseline for production load
    disk_size     = 100                         # Larger disk for production workloads
    labels = {
      role        = "general"
      environment = "production"
    }
    taints = []
  }
  compute = {
    instance_types = ["c5.xlarge", "c5.2xlarge", "c5.4xlarge"]  # High-performance instances
    capacity_type  = "ON_DEMAND"                                # Reliable capacity for ML workloads
    min_size      = 2                                           # Maintain minimum capacity
    max_size      = 30                                          # High max for ML scaling
    desired_size  = 4                                           # Higher baseline for production
    disk_size     = 200                                         # Large disk for model storage
    labels = {
      role        = "compute"
      workload    = "ml-training"
      environment = "production"
    }
    taints = [{
      key    = "compute-optimized"
      value  = "true"
      effect = "NO_SCHEDULE"
    }]
  }
  memory-optimized = {
    instance_types = ["r5.large", "r5.xlarge", "r5.2xlarge"]  # Memory-optimized for large models
    capacity_type  = "ON_DEMAND"
    min_size      = 0
    max_size      = 10
    desired_size  = 0                                          # Scale up only when needed
    disk_size     = 150
    labels = {
      role        = "memory-optimized"
      workload    = "large-model-inference"
      environment = "production"
    }
    taints = [{
      key    = "memory-optimized"
      value  = "true"
      effect = "NO_SCHEDULE"
    }]
  }
}

# =============================================================================
# Database Configuration (RDS PostgreSQL)
# =============================================================================
database_engine                 = "postgres"
database_engine_version         = "15.3"
database_instance_class         = "db.r5.large"     # Performance-optimized instance
database_allocated_storage      = 500               # Large initial storage
database_max_allocated_storage  = 5000              # High max for growth
database_name                   = "fake_news_db"
database_master_username        = "app_user"
database_backup_retention_period = 30               # Long retention for production
database_backup_window          = "03:00-04:00"
database_maintenance_window     = "sun:04:00-sun:05:00"

# =============================================================================
# Redis Configuration (ElastiCache)
# =============================================================================
redis_node_type          = "cache.r5.large"  # Performance-optimized instance
redis_num_cache_nodes    = 3                 # Multi-node for high availability

# =============================================================================
# Monitoring and Logging Configuration
# =============================================================================
log_retention_days           = 365   # Long retention for compliance
enable_container_insights    = true  # Full monitoring enabled
enable_performance_insights  = true  # Database performance monitoring

# =============================================================================
# Security Configuration
# =============================================================================
allowed_management_cidrs = [
  "203.0.113.0/24",   # Office network - replace with actual IPs
  "198.51.100.0/24",  # VPN network - replace with actual IPs
  "192.0.2.0/24"      # Management network - replace with actual IPs
]

enable_secret_rotation = true   # Enabled for production security
secret_rotation_days   = 30     # Frequent rotation for production

# =============================================================================
# Feature Flags
# =============================================================================
enable_backups        = true
enable_monitoring     = true
enable_autoscaling    = true
enable_spot_instances = false  # Disabled for production reliability

# =============================================================================
# Resource Limits and Sizing
# =============================================================================
max_nodes_per_az = 25  # Higher limit for production scaling