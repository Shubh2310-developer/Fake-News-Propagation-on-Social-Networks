# =============================================================================
# Output Values for Fake News Game Theory Platform Infrastructure
# =============================================================================
# This file exposes important information about the created infrastructure
# that can be used by other Terraform configurations or queried from the CLI.

# =============================================================================
# General Information
# =============================================================================

output "project_name" {
  description = "The name of the project"
  value       = var.project_name
}

output "environment" {
  description = "The environment name"
  value       = var.environment
}

output "aws_region" {
  description = "The AWS region where resources are deployed"
  value       = var.aws_region
}

output "account_id" {
  description = "The AWS account ID where resources are deployed"
  value       = data.aws_caller_identity.current.account_id
}

# =============================================================================
# VPC and Network Outputs
# =============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "List of IDs of the public subnets"
  value       = module.vpc.public_subnet_ids
}

output "private_subnet_ids" {
  description = "List of IDs of the private subnets"
  value       = module.vpc.private_subnet_ids
}

output "database_subnet_ids" {
  description = "List of IDs of the database subnets"
  value       = module.vpc.database_subnet_ids
}

output "nat_gateway_ids" {
  description = "List of IDs of the NAT gateways"
  value       = module.vpc.nat_gateway_ids
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = module.vpc.internet_gateway_id
}

# =============================================================================
# EKS Cluster Outputs
# =============================================================================

output "cluster_id" {
  description = "EKS cluster ID"
  value       = module.compute.cluster_id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.compute.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.compute.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.compute.cluster_security_group_id
}

output "cluster_ca_certificate" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.compute.cluster_ca_certificate
  sensitive   = true
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = module.compute.cluster_version
}

output "cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = module.compute.cluster_platform_version
}

output "cluster_status" {
  description = "Status of the EKS cluster"
  value       = module.compute.cluster_status
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.compute.cluster_oidc_issuer_url
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if one is created"
  value       = module.compute.oidc_provider_arn
}

output "node_security_group_id" {
  description = "ID of the EKS node shared security group"
  value       = module.compute.node_security_group_id
}

output "worker_security_group_id" {
  description = "ID of the worker node security group"
  value       = module.compute.worker_security_group_id
}

# =============================================================================
# Node Group Outputs
# =============================================================================

output "node_groups" {
  description = "Map of node group configurations and their statuses"
  value = {
    for name, config in var.node_groups : name => {
      arn           = module.compute.node_group_arns[name]
      status        = module.compute.node_group_status[name]
      capacity_type = config.capacity_type
      instance_types = config.instance_types
      scaling_config = {
        desired_size = config.desired_size
        max_size     = config.max_size
        min_size     = config.min_size
      }
    }
  }
}

# =============================================================================
# Database Outputs
# =============================================================================

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = module.database.endpoint
}

output "database_port" {
  description = "RDS instance port"
  value       = module.database.port
}

output "database_name" {
  description = "RDS database name"
  value       = module.database.database_name
}

output "database_username" {
  description = "RDS database master username"
  value       = module.database.master_username
  sensitive   = true
}

output "database_id" {
  description = "RDS instance ID"
  value       = module.database.id
}

output "database_arn" {
  description = "RDS instance ARN"
  value       = module.database.arn
}

output "database_availability_zone" {
  description = "RDS instance availability zone"
  value       = module.database.availability_zone
}

output "database_backup_retention_period" {
  description = "RDS backup retention period"
  value       = module.database.backup_retention_period
}

output "database_backup_window" {
  description = "RDS backup window"
  value       = module.database.backup_window
}

output "database_maintenance_window" {
  description = "RDS maintenance window"
  value       = module.database.maintenance_window
}

output "database_security_group_id" {
  description = "RDS security group ID"
  value       = module.database.security_group_id
}

# =============================================================================
# Redis Cache Outputs
# =============================================================================

output "redis_cluster_id" {
  description = "ElastiCache replication group ID"
  value       = aws_elasticache_replication_group.redis.replication_group_id
}

output "redis_primary_endpoint" {
  description = "ElastiCache primary endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_reader_endpoint" {
  description = "ElastiCache reader endpoint"
  value       = aws_elasticache_replication_group.redis.reader_endpoint_address
}

output "redis_port" {
  description = "ElastiCache port"
  value       = aws_elasticache_replication_group.redis.port
}

output "redis_security_group_id" {
  description = "ElastiCache security group ID"
  value       = aws_security_group.redis.id
}

# =============================================================================
# S3 Storage Outputs
# =============================================================================

output "app_data_bucket_id" {
  description = "S3 bucket ID for application data"
  value       = aws_s3_bucket.app_data.id
}

output "app_data_bucket_arn" {
  description = "S3 bucket ARN for application data"
  value       = aws_s3_bucket.app_data.arn
}

output "app_data_bucket_domain_name" {
  description = "S3 bucket domain name for application data"
  value       = aws_s3_bucket.app_data.bucket_domain_name
}

output "app_data_bucket_regional_domain_name" {
  description = "S3 bucket regional domain name for application data"
  value       = aws_s3_bucket.app_data.bucket_regional_domain_name
}

# =============================================================================
# IAM Role Outputs
# =============================================================================

output "backend_irsa_role_arn" {
  description = "ARN of the IAM role for the backend service account"
  value       = module.backend_irsa.iam_role_arn
}

output "backend_irsa_role_name" {
  description = "Name of the IAM role for the backend service account"
  value       = module.backend_irsa.iam_role_name
}

# =============================================================================
# Secrets Manager Outputs
# =============================================================================

output "database_credentials_secret_arn" {
  description = "ARN of the database credentials secret"
  value       = aws_secretsmanager_secret.database_credentials.arn
}

output "database_credentials_secret_name" {
  description = "Name of the database credentials secret"
  value       = aws_secretsmanager_secret.database_credentials.name
}

# =============================================================================
# CloudWatch Outputs
# =============================================================================

output "application_log_group_name" {
  description = "Name of the application CloudWatch log group"
  value       = aws_cloudwatch_log_group.application.name
}

output "application_log_group_arn" {
  description = "ARN of the application CloudWatch log group"
  value       = aws_cloudwatch_log_group.application.arn
}

output "cluster_log_group_name" {
  description = "Name of the cluster CloudWatch log group"
  value       = aws_cloudwatch_log_group.cluster.name
}

output "cluster_log_group_arn" {
  description = "ARN of the cluster CloudWatch log group"
  value       = aws_cloudwatch_log_group.cluster.arn
}

# =============================================================================
# Connection Information
# =============================================================================

output "kubeconfig_command" {
  description = "Command to configure kubectl for this cluster"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.compute.cluster_name}"
}

output "database_connection_string" {
  description = "PostgreSQL connection string (without password)"
  value       = "postgresql://${module.database.master_username}@${module.database.endpoint}:${module.database.port}/${module.database.database_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string"
  value       = "redis://${aws_elasticache_replication_group.redis.primary_endpoint_address}:${aws_elasticache_replication_group.redis.port}"
}

# =============================================================================
# Cost Tracking Outputs
# =============================================================================

output "resource_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (approximation)"
  value = {
    eks_cluster = "~$75/month (control plane)"
    node_groups = "~$100-500/month (depending on instance types and scaling)"
    rds_instance = "~$15-100/month (depending on instance class)"
    nat_gateways = "~$45/month (per NAT gateway)"
    elasticache = "~$15-50/month (depending on node type)"
    data_transfer = "Variable based on usage"
    cloudwatch_logs = "~$0.50/GB ingested"
    note = "Actual costs may vary based on usage patterns and AWS pricing changes"
  }
}

# =============================================================================
# Deployment Information
# =============================================================================

output "deployment_timestamp" {
  description = "Timestamp when the infrastructure was deployed"
  value       = timestamp()
}

output "terraform_workspace" {
  description = "Terraform workspace used for this deployment"
  value       = terraform.workspace
}

# =============================================================================
# Next Steps Information
# =============================================================================

output "next_steps" {
  description = "Recommended next steps after infrastructure deployment"
  value = [
    "1. Configure kubectl: ${local.kubeconfig_command}",
    "2. Deploy the application using Kubernetes manifests in ../kubernetes/",
    "3. Set up monitoring dashboards in CloudWatch or Grafana",
    "4. Configure DNS records to point to the load balancer",
    "5. Set up backup and disaster recovery procedures",
    "6. Configure alerting for critical metrics",
    "7. Review and adjust autoscaling policies based on actual usage"
  ]
}

# =============================================================================
# Local Values for Output Processing
# =============================================================================

locals {
  kubeconfig_command = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.compute.cluster_name}"
}