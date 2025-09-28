# =============================================================================
# Compute Module Outputs
# =============================================================================

output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "cluster_ca_certificate" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = aws_eks_cluster.main.version
}

output "cluster_platform_version" {
  description = "Platform version for the EKS cluster"
  value       = aws_eks_cluster.main.platform_version
}

output "cluster_status" {
  description = "Status of the EKS cluster"
  value       = aws_eks_cluster.main.status
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider"
  value       = aws_iam_openid_connect_provider.cluster.arn
}

output "node_security_group_id" {
  description = "ID of the EKS node shared security group"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "worker_security_group_id" {
  description = "ID of the worker node security group"
  value       = aws_security_group.worker_nodes.id
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by EKS"
  value       = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
}

output "node_group_arns" {
  description = "Map of node group ARNs"
  value = {
    for name, node_group in aws_eks_node_group.main : name => node_group.arn
  }
}

output "node_group_status" {
  description = "Map of node group statuses"
  value = {
    for name, node_group in aws_eks_node_group.main : name => node_group.status
  }
}

output "node_group_capacities" {
  description = "Map of node group capacity information"
  value = {
    for name, node_group in aws_eks_node_group.main : name => {
      capacity_type  = node_group.capacity_type
      instance_types = node_group.instance_types
      scaling_config = node_group.scaling_config
    }
  }
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "cluster_iam_role_name" {
  description = "IAM role name of the EKS cluster"
  value       = aws_iam_role.cluster.name
}

output "node_group_iam_role_arn" {
  description = "IAM role ARN of the EKS node groups"
  value       = aws_iam_role.node_group.arn
}

output "node_group_iam_role_name" {
  description = "IAM role name of the EKS node groups"
  value       = aws_iam_role.node_group.name
}

output "cluster_autoscaler_iam_role_arn" {
  description = "IAM role ARN for the cluster autoscaler"
  value       = aws_iam_role.cluster_autoscaler.arn
}

output "cluster_autoscaler_iam_role_name" {
  description = "IAM role name for the cluster autoscaler"
  value       = aws_iam_role.cluster_autoscaler.name
}

output "aws_load_balancer_controller_iam_role_arn" {
  description = "IAM role ARN for the AWS Load Balancer Controller"
  value       = aws_iam_role.aws_load_balancer_controller.arn
}

output "aws_load_balancer_controller_iam_role_name" {
  description = "IAM role name for the AWS Load Balancer Controller"
  value       = aws_iam_role.aws_load_balancer_controller.name
}

output "cluster_encryption_key_arn" {
  description = "The ARN of the KMS key used to encrypt EKS secrets"
  value       = aws_kms_key.eks.arn
}

output "cluster_encryption_key_id" {
  description = "The ID of the KMS key used to encrypt EKS secrets"
  value       = aws_kms_key.eks.key_id
}

output "cluster_log_group_name" {
  description = "Name of the CloudWatch log group for cluster logs"
  value       = aws_cloudwatch_log_group.cluster.name
}

output "cluster_log_group_arn" {
  description = "ARN of the CloudWatch log group for cluster logs"
  value       = aws_cloudwatch_log_group.cluster.arn
}

output "launch_template_ids" {
  description = "Map of launch template IDs for node groups"
  value = {
    for name, template in aws_launch_template.node_group : name => template.id
  }
}

output "launch_template_latest_versions" {
  description = "Map of latest versions for launch templates"
  value = {
    for name, template in aws_launch_template.node_group : name => template.latest_version
  }
}

output "cluster_addons" {
  description = "Map of cluster addon configurations and statuses"
  value = {
    for name, addon in aws_eks_addon.addons : name => {
      arn           = addon.arn
      status        = addon.status
      version       = addon.addon_version
      service_account_role_arn = addon.service_account_role_arn
    }
  }
}

# Useful outputs for integration
output "kubeconfig_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.main.certificate_authority[0].data
}

output "cluster_endpoint_private_access" {
  description = "Whether private API server endpoint is enabled"
  value       = aws_eks_cluster.main.vpc_config[0].endpoint_private_access
}

output "cluster_endpoint_public_access" {
  description = "Whether public API server endpoint is enabled"
  value       = aws_eks_cluster.main.vpc_config[0].endpoint_public_access
}

output "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the public API server endpoint"
  value       = aws_eks_cluster.main.vpc_config[0].public_access_cidrs
}

output "cluster_vpc_config" {
  description = "VPC configuration for the cluster"
  value = {
    vpc_id                   = aws_eks_cluster.main.vpc_config[0].vpc_id
    subnet_ids              = aws_eks_cluster.main.vpc_config[0].subnet_ids
    security_group_ids      = aws_eks_cluster.main.vpc_config[0].security_group_ids
    cluster_security_group_id = aws_eks_cluster.main.vpc_config[0].cluster_security_group_id
  }
}