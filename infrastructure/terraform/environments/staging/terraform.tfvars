# =============================================================================
# Staging Environment Configuration - Fake News Game Theory Platform
# =============================================================================
# This file contains staging-specific variable values for cost optimization
# and development-friendly settings.

# =============================================================================
# General Configuration
# =============================================================================
project_name = "fake-news-game-theory"
environment  = "staging"
cost_center  = "engineering"

# =============================================================================
# AWS Configuration
# =============================================================================
aws_region               = "us-west-2"
terraform_state_bucket   = "fake-news-terraform-state-staging"
terraform_lock_table     = "terraform-state-lock-staging"
az_count                 = 2  # Reduced for cost savings

# =============================================================================
# Network Configuration
# =============================================================================
vpc_cidr = "10.1.0.0/16"

# Staging subnets - smaller CIDR blocks
public_subnet_cidrs   = ["10.1.1.0/24", "10.1.2.0/24"]
private_subnet_cidrs  = ["10.1.11.0/24", "10.1.12.0/24"]
database_subnet_cidrs = ["10.1.21.0/24", "10.1.22.0/24"]

enable_nat_gateway = true   # Needed for private subnet internet access
enable_vpn_gateway = false  # Not needed for staging

# =============================================================================
# EKS Cluster Configuration
# =============================================================================
cluster_version = "1.28"

cluster_endpoint_config = {
  private_access      = true
  public_access       = true
  public_access_cidrs = ["0.0.0.0/0"]  # More permissive for staging
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

# Staging node groups - cost-optimized
node_groups = {
  general = {
    instance_types = ["t3.small", "t3.medium"]  # Smaller instances for staging
    capacity_type  = "SPOT"                     # Use spot instances for cost savings
    min_size      = 1
    max_size      = 5                          # Lower max for cost control
    desired_size  = 2
    disk_size     = 30                         # Smaller disk for cost savings
    labels = {
      role        = "general"
      environment = "staging"
    }
    taints = []
  }
  compute = {
    instance_types = ["c5.large"]               # Single instance type for simplicity
    capacity_type  = "SPOT"                     # Spot instances for non-critical workloads
    min_size      = 0                          # Can scale to zero
    max_size      = 3                          # Lower max for cost control
    desired_size  = 0                          # Start with zero nodes
    disk_size     = 50
    labels = {
      role        = "compute"
      workload    = "ml-training"
      environment = "staging"
    }
    taints = [{
      key    = "compute-optimized"
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
database_instance_class         = "db.t3.micro"  # Smallest instance for staging
database_allocated_storage      = 20             # Minimum storage
database_max_allocated_storage  = 100            # Lower max for cost control
database_name                   = "fake_news_db"
database_master_username        = "app_user"
database_backup_retention_period = 3             # Shorter retention for staging
database_backup_window          = "03:00-04:00"
database_maintenance_window     = "sun:04:00-sun:05:00"

# =============================================================================
# Redis Configuration (ElastiCache)
# =============================================================================
redis_node_type          = "cache.t3.micro"  # Smallest instance for staging
redis_num_cache_nodes    = 1                 # Single node for cost savings

# =============================================================================
# Monitoring and Logging Configuration
# =============================================================================
log_retention_days           = 7      # Shorter retention for cost savings
enable_container_insights    = false  # Disabled for cost savings
enable_performance_insights  = false  # Disabled for cost savings

# =============================================================================
# Security Configuration
# =============================================================================
allowed_management_cidrs = [
  "0.0.0.0/0"  # More permissive for staging (consider restricting to office IPs)
]

enable_secret_rotation = false  # Disabled for staging simplicity
secret_rotation_days   = 90

# =============================================================================
# Feature Flags
# =============================================================================
enable_backups      = true
enable_monitoring   = false  # Basic monitoring only
enable_autoscaling  = true
enable_spot_instances = true  # Enabled for cost savings

# =============================================================================
# Resource Limits and Sizing
# =============================================================================
max_nodes_per_az = 5  # Lower limit for staging