# =============================================================================
# Input Variables for Fake News Game Theory Platform Infrastructure
# =============================================================================
# This file defines all input variables that can be customized for deployments,
# promoting reusability and security by decoupling configuration from code.

# =============================================================================
# General Configuration
# =============================================================================

variable "project_name" {
  description = "Name of the project used for resource naming and tagging"
  type        = string
  default     = "fake-news-game-theory"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (staging, production, development)"
  type        = string

  validation {
    condition     = contains(["staging", "production", "development"], var.environment)
    error_message = "Environment must be one of: staging, production, development."
  }
}

variable "cost_center" {
  description = "Cost center for resource billing and tracking"
  type        = string
  default     = "engineering"
}

# =============================================================================
# AWS Configuration
# =============================================================================

variable "aws_region" {
  description = "AWS region where resources will be deployed"
  type        = string
  default     = "us-west-2"

  validation {
    condition = can(regex("^[a-z]{2}-[a-z]+-[0-9]$", var.aws_region))
    error_message = "AWS region must be in the format: us-west-2, eu-central-1, etc."
  }
}

variable "terraform_state_bucket" {
  description = "S3 bucket name for storing Terraform state"
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]*[a-z0-9]$", var.terraform_state_bucket))
    error_message = "S3 bucket name must be lowercase and contain only letters, numbers, and hyphens."
  }
}

variable "terraform_lock_table" {
  description = "DynamoDB table name for Terraform state locking"
  type        = string
  default     = "terraform-state-lock"
}

variable "az_count" {
  description = "Number of availability zones to use (must be 2 or more for production)"
  type        = number
  default     = 3

  validation {
    condition     = var.az_count >= 2 && var.az_count <= 6
    error_message = "Number of availability zones must be between 2 and 6."
  }
}

# =============================================================================
# Network Configuration
# =============================================================================

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets (one per AZ)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]

  validation {
    condition = alltrue([
      for cidr in var.public_subnet_cidrs : can(cidrhost(cidr, 0))
    ])
    error_message = "All public subnet CIDRs must be valid IPv4 CIDR blocks."
  }
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets (one per AZ)"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]

  validation {
    condition = alltrue([
      for cidr in var.private_subnet_cidrs : can(cidrhost(cidr, 0))
    ])
    error_message = "All private subnet CIDRs must be valid IPv4 CIDR blocks."
  }
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for database subnets (one per AZ)"
  type        = list(string)
  default     = ["10.0.21.0/24", "10.0.22.0/24", "10.0.23.0/24"]

  validation {
    condition = alltrue([
      for cidr in var.database_subnet_cidrs : can(cidrhost(cidr, 0))
    ])
    error_message = "All database subnet CIDRs must be valid IPv4 CIDR blocks."
  }
}

variable "enable_nat_gateway" {
  description = "Whether to create NAT gateways for private subnet internet access"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Whether to create a VPN gateway for the VPC"
  type        = bool
  default     = false
}

# =============================================================================
# EKS Cluster Configuration
# =============================================================================

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.28"

  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+$", var.cluster_version))
    error_message = "Cluster version must be in format 'x.y' (e.g., '1.28')."
  }
}

variable "cluster_endpoint_config" {
  description = "EKS cluster endpoint configuration"
  type = object({
    private_access = bool
    public_access  = bool
    public_access_cidrs = list(string)
  })
  default = {
    private_access = true
    public_access  = true
    public_access_cidrs = ["0.0.0.0/0"]
  }
}

variable "cluster_addons" {
  description = "Map of EKS cluster addon configurations"
  type = map(object({
    version = string
    configuration_values = optional(string)
  }))
  default = {
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
}

variable "node_groups" {
  description = "Map of EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    min_size      = number
    max_size      = number
    desired_size  = number
    disk_size     = number
    labels        = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {
    general = {
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      min_size      = 1
      max_size      = 10
      desired_size  = 3
      disk_size     = 50
      labels = {
        role = "general"
      }
      taints = []
    }
    compute = {
      instance_types = ["c5.large", "c5.xlarge"]
      capacity_type  = "SPOT"
      min_size      = 0
      max_size      = 20
      desired_size  = 2
      disk_size     = 100
      labels = {
        role = "compute"
        workload = "ml-training"
      }
      taints = [{
        key    = "compute-optimized"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }
}

# =============================================================================
# Database Configuration (RDS PostgreSQL)
# =============================================================================

variable "database_engine" {
  description = "Database engine type"
  type        = string
  default     = "postgres"

  validation {
    condition     = contains(["postgres", "mysql"], var.database_engine)
    error_message = "Database engine must be either 'postgres' or 'mysql'."
  }
}

variable "database_engine_version" {
  description = "Database engine version"
  type        = string
  default     = "15.3"
}

variable "database_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"

  validation {
    condition     = can(regex("^db\\.[a-z0-9]+\\.[a-z0-9]+$", var.database_instance_class))
    error_message = "Database instance class must be in format 'db.type.size'."
  }
}

variable "database_allocated_storage" {
  description = "Initial allocated storage for the database (GB)"
  type        = number
  default     = 20

  validation {
    condition     = var.database_allocated_storage >= 20 && var.database_allocated_storage <= 65536
    error_message = "Database allocated storage must be between 20 and 65536 GB."
  }
}

variable "database_max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling (GB)"
  type        = number
  default     = 1000

  validation {
    condition     = var.database_max_allocated_storage >= 20 && var.database_max_allocated_storage <= 65536
    error_message = "Database max allocated storage must be between 20 and 65536 GB."
  }
}

variable "database_name" {
  description = "Name of the initial database to create"
  type        = string
  default     = "fake_news_db"

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]*$", var.database_name))
    error_message = "Database name must start with a letter and contain only letters, numbers, and underscores."
  }
}

variable "database_master_username" {
  description = "Master username for the database"
  type        = string
  default     = "app_user"

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]*$", var.database_master_username))
    error_message = "Database username must start with a letter and contain only letters, numbers, and underscores."
  }
}

variable "database_backup_retention_period" {
  description = "Number of days to retain database backups"
  type        = number
  default     = 7

  validation {
    condition     = var.database_backup_retention_period >= 0 && var.database_backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "database_backup_window" {
  description = "Preferred backup window (UTC)"
  type        = string
  default     = "03:00-04:00"

  validation {
    condition     = can(regex("^[0-9]{2}:[0-9]{2}-[0-9]{2}:[0-9]{2}$", var.database_backup_window))
    error_message = "Backup window must be in format 'HH:MM-HH:MM'."
  }
}

variable "database_maintenance_window" {
  description = "Preferred maintenance window (UTC)"
  type        = string
  default     = "sun:04:00-sun:05:00"

  validation {
    condition     = can(regex("^[a-z]{3}:[0-9]{2}:[0-9]{2}-[a-z]{3}:[0-9]{2}:[0-9]{2}$", var.database_maintenance_window))
    error_message = "Maintenance window must be in format 'ddd:HH:MM-ddd:HH:MM'."
  }
}

# =============================================================================
# Redis Configuration (ElastiCache)
# =============================================================================

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"

  validation {
    condition     = can(regex("^cache\\.[a-z0-9]+\\.[a-z0-9]+$", var.redis_node_type))
    error_message = "Redis node type must be in format 'cache.type.size'."
  }
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in the Redis cluster"
  type        = number
  default     = 2

  validation {
    condition     = var.redis_num_cache_nodes >= 1 && var.redis_num_cache_nodes <= 20
    error_message = "Number of Redis cache nodes must be between 1 and 20."
  }
}

# =============================================================================
# Monitoring and Logging Configuration
# =============================================================================

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 30

  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention days must be one of the valid CloudWatch retention periods."
  }
}

variable "enable_container_insights" {
  description = "Whether to enable CloudWatch Container Insights for EKS"
  type        = bool
  default     = true
}

variable "enable_performance_insights" {
  description = "Whether to enable Performance Insights for RDS"
  type        = bool
  default     = true
}

# =============================================================================
# Security Configuration
# =============================================================================

variable "allowed_management_cidrs" {
  description = "CIDR blocks allowed to access management interfaces"
  type        = list(string)
  default     = ["0.0.0.0/0"]

  validation {
    condition = alltrue([
      for cidr in var.allowed_management_cidrs : can(cidrhost(cidr, 0))
    ])
    error_message = "All management CIDRs must be valid IPv4 CIDR blocks."
  }
}

variable "enable_secret_rotation" {
  description = "Whether to enable automatic secret rotation"
  type        = bool
  default     = false
}

variable "secret_rotation_days" {
  description = "Number of days between automatic secret rotations"
  type        = number
  default     = 30

  validation {
    condition     = var.secret_rotation_days >= 1 && var.secret_rotation_days <= 365
    error_message = "Secret rotation days must be between 1 and 365."
  }
}

# =============================================================================
# Feature Flags
# =============================================================================

variable "enable_backups" {
  description = "Whether to enable automated backups"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Whether to enable comprehensive monitoring"
  type        = bool
  default     = true
}

variable "enable_autoscaling" {
  description = "Whether to enable cluster autoscaling"
  type        = bool
  default     = true
}

variable "enable_spot_instances" {
  description = "Whether to allow spot instances in node groups"
  type        = bool
  default     = false
}

# =============================================================================
# Resource Limits and Sizing
# =============================================================================

variable "max_nodes_per_az" {
  description = "Maximum number of nodes per availability zone"
  type        = number
  default     = 10

  validation {
    condition     = var.max_nodes_per_az >= 1 && var.max_nodes_per_az <= 50
    error_message = "Maximum nodes per AZ must be between 1 and 50."
  }
}