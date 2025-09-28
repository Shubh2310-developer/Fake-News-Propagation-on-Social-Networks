# =============================================================================
# Database Module Outputs
# =============================================================================

output "id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.id
}

output "arn" {
  description = "RDS instance ARN"
  value       = aws_db_instance.main.arn
}

output "identifier" {
  description = "RDS instance identifier"
  value       = aws_db_instance.main.identifier
}

output "endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "database_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "master_username" {
  description = "Master username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "engine" {
  description = "Database engine"
  value       = aws_db_instance.main.engine
}

output "engine_version" {
  description = "Database engine version"
  value       = aws_db_instance.main.engine_version
}

output "instance_class" {
  description = "RDS instance class"
  value       = aws_db_instance.main.instance_class
}

output "allocated_storage" {
  description = "Allocated storage in GB"
  value       = aws_db_instance.main.allocated_storage
}

output "availability_zone" {
  description = "Availability zone of the RDS instance"
  value       = aws_db_instance.main.availability_zone
}

output "backup_retention_period" {
  description = "Backup retention period"
  value       = aws_db_instance.main.backup_retention_period
}

output "backup_window" {
  description = "Backup window"
  value       = aws_db_instance.main.backup_window
}

output "maintenance_window" {
  description = "Maintenance window"
  value       = aws_db_instance.main.maintenance_window
}

output "multi_az" {
  description = "Whether Multi-AZ is enabled"
  value       = aws_db_instance.main.multi_az
}

output "storage_encrypted" {
  description = "Whether storage is encrypted"
  value       = aws_db_instance.main.storage_encrypted
}

output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = aws_db_instance.main.kms_key_id
}

output "performance_insights_enabled" {
  description = "Whether Performance Insights is enabled"
  value       = aws_db_instance.main.performance_insights_enabled
}

output "monitoring_interval" {
  description = "Enhanced monitoring interval"
  value       = aws_db_instance.main.monitoring_interval
}

output "status" {
  description = "RDS instance status"
  value       = aws_db_instance.main.status
}

output "security_group_id" {
  description = "Security group ID for the RDS instance"
  value       = aws_security_group.rds.id
}

output "db_subnet_group_name" {
  description = "Database subnet group name"
  value       = aws_db_subnet_group.main.name
}

output "db_subnet_group_arn" {
  description = "Database subnet group ARN"
  value       = aws_db_subnet_group.main.arn
}

output "parameter_group_name" {
  description = "Database parameter group name"
  value       = aws_db_parameter_group.main.name
}

output "parameter_group_arn" {
  description = "Database parameter group ARN"
  value       = aws_db_parameter_group.main.arn
}

output "option_group_name" {
  description = "Database option group name"
  value       = aws_db_option_group.main.name
}

output "option_group_arn" {
  description = "Database option group ARN"
  value       = aws_db_option_group.main.arn
}

# Read Replica Outputs
output "read_replica_id" {
  description = "Read replica instance ID"
  value       = var.create_read_replica ? aws_db_instance.read_replica[0].id : null
}

output "read_replica_endpoint" {
  description = "Read replica endpoint"
  value       = var.create_read_replica ? aws_db_instance.read_replica[0].endpoint : null
}

output "read_replica_arn" {
  description = "Read replica ARN"
  value       = var.create_read_replica ? aws_db_instance.read_replica[0].arn : null
}

# CloudWatch Outputs
output "cloudwatch_log_groups" {
  description = "Map of CloudWatch log group names"
  value = {
    postgresql = aws_cloudwatch_log_group.postgresql.name
    upgrade    = aws_cloudwatch_log_group.upgrade.name
  }
}

output "cloudwatch_alarms" {
  description = "Map of CloudWatch alarm names"
  value = {
    cpu_utilization      = aws_cloudwatch_metric_alarm.database_cpu.alarm_name
    freeable_memory     = aws_cloudwatch_metric_alarm.database_freeable_memory.alarm_name
    free_storage_space  = aws_cloudwatch_metric_alarm.database_free_storage_space.alarm_name
    connection_count    = aws_cloudwatch_metric_alarm.database_connection_count.alarm_name
  }
}

# Secrets Manager Outputs
output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing database credentials"
  value       = var.store_credentials_in_secrets_manager ? aws_secretsmanager_secret.db_credentials[0].arn : null
}

output "secrets_manager_secret_name" {
  description = "Name of the Secrets Manager secret containing database credentials"
  value       = var.store_credentials_in_secrets_manager ? aws_secretsmanager_secret.db_credentials[0].name : null
}

# Enhanced Monitoring IAM Role
output "enhanced_monitoring_iam_role_arn" {
  description = "ARN of the IAM role for enhanced monitoring"
  value       = var.monitoring_interval > 0 ? aws_iam_role.rds_enhanced_monitoring[0].arn : null
}

# KMS Encryption Key
output "kms_key_arn" {
  description = "ARN of the KMS key used for encryption"
  value       = var.storage_encrypted ? aws_kms_key.rds[0].arn : null
}

output "kms_key_id" {
  description = "ID of the KMS key used for encryption"
  value       = var.storage_encrypted ? aws_kms_key.rds[0].key_id : null
}

# Connection Information
output "connection_string" {
  description = "PostgreSQL connection string (without password)"
  value       = "postgresql://${aws_db_instance.main.username}@${aws_db_instance.main.endpoint}:${aws_db_instance.main.port}/${aws_db_instance.main.db_name}"
  sensitive   = true
}

output "jdbc_connection_string" {
  description = "JDBC connection string (without password)"
  value       = "jdbc:postgresql://${aws_db_instance.main.endpoint}:${aws_db_instance.main.port}/${aws_db_instance.main.db_name}"
}

# Migration Commands Parameter
output "migration_commands_parameter_name" {
  description = "SSM parameter name containing database migration setup commands"
  value       = aws_ssm_parameter.db_migration_commands.name
}

# Performance and Monitoring URLs
output "performance_insights_url" {
  description = "URL to Performance Insights dashboard"
  value       = var.performance_insights_enabled ? "https://console.aws.amazon.com/rds/home?region=${data.aws_region.current.name}#performance-insights-v20206:resourceId=${aws_db_instance.main.resource_id}" : null
}

output "cloudwatch_dashboard_url" {
  description = "URL to CloudWatch metrics dashboard"
  value       = "https://console.aws.amazon.com/cloudwatch/home?region=${data.aws_region.current.name}#metricsV2:graph=~();query=AWS%2FRDS%20DBInstanceIdentifier%3D${aws_db_instance.main.id}"
}

# Backup Information
output "latest_restorable_time" {
  description = "Latest time to which a database can be restored with point-in-time restore"
  value       = aws_db_instance.main.latest_restorable_time
}

output "backup_retention_period_days" {
  description = "Number of days for which backups are retained"
  value       = aws_db_instance.main.backup_retention_period
}

# Storage Information
output "storage_type" {
  description = "Storage type of the RDS instance"
  value       = aws_db_instance.main.storage_type
}

output "iops" {
  description = "IOPS of the RDS instance"
  value       = aws_db_instance.main.iops
}

output "storage_throughput" {
  description = "Storage throughput of the RDS instance"
  value       = aws_db_instance.main.storage_throughput
}

# Data sources for additional context
data "aws_region" "current" {}