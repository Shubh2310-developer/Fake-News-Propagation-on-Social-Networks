# =============================================================================
# Terraform Configuration for Fake News Game Theory Platform
# =============================================================================
# This is the root module that orchestrates the entire infrastructure
# deployment across multiple cloud providers (AWS primary, with multi-cloud support)

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Remote State Backend Configuration
  # This ensures state is stored securely and enables team collaboration
  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "fake-news-platform/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = var.terraform_lock_table
  }
}

# =============================================================================
# Provider Configuration
# =============================================================================

# Primary AWS Provider
provider "aws" {
  region = var.aws_region

  # Default tags applied to all resources
  default_tags {
    tags = {
      Project     = "fake-news-game-theory"
      Environment = var.environment
      ManagedBy   = "terraform"
      CreatedBy   = "infrastructure-team"
      CostCenter  = var.cost_center
    }
  }
}

# Kubernetes Provider (configured after EKS cluster creation)
provider "kubernetes" {
  host                   = module.compute.cluster_endpoint
  cluster_ca_certificate = base64decode(module.compute.cluster_ca_certificate)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = [
      "eks",
      "get-token",
      "--cluster-name",
      module.compute.cluster_name
    ]
  }
}

# Helm Provider (for installing cluster components)
provider "helm" {
  kubernetes {
    host                   = module.compute.cluster_endpoint
    cluster_ca_certificate = base64decode(module.compute.cluster_ca_certificate)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args = [
        "eks",
        "get-token",
        "--cluster-name",
        module.compute.cluster_name
      ]
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

# Current AWS account information
data "aws_caller_identity" "current" {}

# Available availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# =============================================================================
# Local Values
# =============================================================================

locals {
  # Common naming prefix
  name_prefix = "${var.project_name}-${var.environment}"

  # Common tags
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    AccountId   = data.aws_caller_identity.current.account_id
  }

  # AZ configuration
  availability_zones = slice(data.aws_availability_zones.available.names, 0, var.az_count)

  # Database configuration
  database_config = {
    engine_version = var.database_engine_version
    instance_class = var.database_instance_class
    allocated_storage = var.database_allocated_storage
    backup_retention_period = var.database_backup_retention_period
    multi_az = var.environment == "production" ? true : false
  }
}

# =============================================================================
# Random Password Generation
# =============================================================================

# Database master password
resource "random_password" "db_master_password" {
  length  = 16
  special = true
}

# =============================================================================
# VPC Module
# =============================================================================

module "vpc" {
  source = "./modules/vpc"

  # Basic configuration
  name_prefix        = local.name_prefix
  environment        = var.environment
  vpc_cidr           = var.vpc_cidr
  availability_zones = local.availability_zones

  # Subnet configuration
  public_subnet_cidrs  = var.public_subnet_cidrs
  private_subnet_cidrs = var.private_subnet_cidrs
  database_subnet_cidrs = var.database_subnet_cidrs

  # Feature flags
  enable_nat_gateway = var.enable_nat_gateway
  enable_vpn_gateway = var.enable_vpn_gateway
  enable_dns_hostnames = true
  enable_dns_support = true

  # Tagging
  tags = local.common_tags
}

# =============================================================================
# Compute Module (EKS Cluster)
# =============================================================================

module "compute" {
  source = "./modules/compute"

  # Basic configuration
  name_prefix    = local.name_prefix
  environment    = var.environment
  cluster_version = var.cluster_version

  # VPC configuration
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  public_subnet_ids   = module.vpc.public_subnet_ids

  # Node group configuration
  node_groups = var.node_groups

  # Security configuration
  cluster_endpoint_config = var.cluster_endpoint_config

  # Addons
  cluster_addons = var.cluster_addons

  # Tagging
  tags = local.common_tags

  depends_on = [module.vpc]
}

# =============================================================================
# Database Module (RDS)
# =============================================================================

module "database" {
  source = "./modules/database"

  # Basic configuration
  name_prefix  = local.name_prefix
  environment  = var.environment

  # Database configuration
  engine               = var.database_engine
  engine_version       = local.database_config.engine_version
  instance_class       = local.database_config.instance_class
  allocated_storage    = local.database_config.allocated_storage
  max_allocated_storage = var.database_max_allocated_storage

  # Database credentials
  database_name = var.database_name
  master_username = var.database_master_username
  master_password = random_password.db_master_password.result

  # Network configuration
  vpc_id               = module.vpc.vpc_id
  database_subnet_ids  = module.vpc.database_subnet_ids
  allowed_cidr_blocks  = module.vpc.private_subnet_cidrs

  # Backup and maintenance
  backup_retention_period = local.database_config.backup_retention_period
  backup_window          = var.database_backup_window
  maintenance_window     = var.database_maintenance_window
  multi_az              = local.database_config.multi_az

  # Security
  storage_encrypted = true
  deletion_protection = var.environment == "production" ? true : false

  # Performance monitoring
  performance_insights_enabled = true
  monitoring_interval = 60

  # Tagging
  tags = local.common_tags

  depends_on = [module.vpc]
}

# =============================================================================
# Additional AWS Resources
# =============================================================================

# S3 Bucket for application data storage
resource "aws_s3_bucket" "app_data" {
  bucket = "${local.name_prefix}-app-data-${random_id.bucket_suffix.hex}"

  tags = merge(local.common_tags, {
    Purpose = "application-data-storage"
  })
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "app_data" {
  bucket = aws_s3_bucket.app_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "app_data" {
  bucket = aws_s3_bucket.app_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ElastiCache Redis cluster for caching
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.name_prefix}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnet_ids

  tags = local.common_tags
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name_prefix}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.compute.worker_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-redis-sg"
  })
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${local.name_prefix}-redis"
  description                = "Redis cluster for ${var.project_name} ${var.environment}"

  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"

  num_cache_clusters         = var.redis_num_cache_nodes

  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  automatic_failover_enabled = var.environment == "production" ? true : false
  multi_az_enabled          = var.environment == "production" ? true : false

  tags = local.common_tags
}

# CloudWatch Log Groups for centralized logging
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${module.compute.cluster_name}/application"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "cluster" {
  name              = "/aws/eks/${module.compute.cluster_name}/cluster"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# =============================================================================
# Secrets Manager for sensitive configuration
# =============================================================================

resource "aws_secretsmanager_secret" "database_credentials" {
  name                    = "${local.name_prefix}-database-credentials"
  description             = "Database credentials for ${var.project_name} ${var.environment}"
  recovery_window_in_days = var.environment == "production" ? 30 : 0

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "database_credentials" {
  secret_id = aws_secretsmanager_secret.database_credentials.id
  secret_string = jsonencode({
    username = module.database.master_username
    password = random_password.db_master_password.result
    endpoint = module.database.endpoint
    port     = module.database.port
    dbname   = module.database.database_name
  })
}

# =============================================================================
# IAM Roles for Application Pods
# =============================================================================

# IRSA (IAM Roles for Service Accounts) for the backend application
module "backend_irsa" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"

  role_name = "${local.name_prefix}-backend-irsa"

  attach_s3_policy = true
  s3_bucket_arns   = [aws_s3_bucket.app_data.arn]

  attach_secrets_manager_policy = true
  secrets_manager_arns         = [aws_secretsmanager_secret.database_credentials.arn]

  oidc_providers = {
    ex = {
      provider_arn               = module.compute.oidc_provider_arn
      namespace_service_accounts = ["fakenews-staging:fake-news-backend", "fakenews-production:fake-news-backend"]
    }
  }

  tags = local.common_tags
}