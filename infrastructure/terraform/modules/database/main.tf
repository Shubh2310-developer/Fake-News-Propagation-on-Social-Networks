# =============================================================================
# Database Module - RDS PostgreSQL for Fake News Game Theory Platform
# =============================================================================
# This module creates a production-ready RDS PostgreSQL instance with
# high availability, security, and monitoring features.

# =============================================================================
# Data Sources
# =============================================================================

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# =============================================================================
# Local Values
# =============================================================================

locals {
  db_identifier = "${var.name_prefix}-postgres"

  # Common tags for all database resources
  database_tags = merge(var.tags, {
    Component = "database"
    Module    = "rds"
    Engine    = var.engine
  })

  # Determine final configuration based on environment
  final_config = {
    multi_az               = var.multi_az
    backup_retention_period = var.backup_retention_period
    storage_encrypted      = var.storage_encrypted
    deletion_protection    = var.deletion_protection
    skip_final_snapshot    = var.environment == "production" ? false : true
    final_snapshot_identifier = var.environment == "production" ? "${local.db_identifier}-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}" : null
  }
}

# =============================================================================
# KMS Key for RDS Encryption
# =============================================================================

resource "aws_kms_key" "rds" {
  count = var.storage_encrypted ? 1 : 0

  description             = "KMS key for RDS encryption - ${local.db_identifier}"
  deletion_window_in_days = 7

  tags = merge(local.database_tags, {
    Name = "${local.db_identifier}-encryption-key"
  })
}

resource "aws_kms_alias" "rds" {
  count = var.storage_encrypted ? 1 : 0

  name          = "alias/${local.db_identifier}-encryption"
  target_key_id = aws_kms_key.rds[0].key_id
}

# =============================================================================
# DB Subnet Group
# =============================================================================

resource "aws_db_subnet_group" "main" {
  name       = "${local.db_identifier}-subnet-group"
  subnet_ids = var.database_subnet_ids

  tags = merge(local.database_tags, {
    Name = "${local.db_identifier}-subnet-group"
  })
}

# =============================================================================
# Security Group for RDS
# =============================================================================

resource "aws_security_group" "rds" {
  name_prefix = "${local.db_identifier}-"
  vpc_id      = var.vpc_id

  # PostgreSQL port from allowed CIDR blocks
  dynamic "ingress" {
    for_each = var.allowed_cidr_blocks
    content {
      description = "PostgreSQL from ${ingress.value}"
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  # Allow access from specific security groups if provided
  dynamic "ingress" {
    for_each = var.allowed_security_groups
    content {
      description     = "PostgreSQL from security group"
      from_port       = 5432
      to_port         = 5432
      protocol        = "tcp"
      security_groups = [ingress.value]
    }
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.database_tags, {
    Name = "${local.db_identifier}-sg"
  })
}

# =============================================================================
# DB Parameter Group
# =============================================================================

resource "aws_db_parameter_group" "main" {
  family = "postgres15"
  name   = "${local.db_identifier}-params"

  # Performance optimizations
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,pg_hint_plan"
  }

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"  # Log queries taking longer than 1 second
  }

  parameter {
    name  = "log_checkpoints"
    value = "1"
  }

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name  = "log_disconnections"
    value = "1"
  }

  parameter {
    name  = "log_lock_waits"
    value = "1"
  }

  parameter {
    name  = "log_temp_files"
    value = "0"
  }

  parameter {
    name  = "checkpoint_completion_target"
    value = "0.9"
  }

  parameter {
    name  = "wal_buffers"
    value = "16MB"
  }

  parameter {
    name  = "default_statistics_target"
    value = "100"
  }

  # Memory settings (will be calculated based on instance class)
  parameter {
    name  = "shared_buffers"
    value = "{DBInstanceClassMemory/4}"
  }

  parameter {
    name  = "effective_cache_size"
    value = "{DBInstanceClassMemory*3/4}"
  }

  parameter {
    name  = "maintenance_work_mem"
    value = "{DBInstanceClassMemory/16}"
  }

  parameter {
    name  = "work_mem"
    value = "{DBInstanceClassMemory/64}"
  }

  tags = local.database_tags
}

# =============================================================================
# DB Option Group (for additional features)
# =============================================================================

resource "aws_db_option_group" "main" {
  name                     = "${local.db_identifier}-options"
  option_group_description = "Option group for ${local.db_identifier}"
  engine_name              = var.engine
  major_engine_version     = split(".", var.engine_version)[0]

  tags = local.database_tags
}

# =============================================================================
# CloudWatch Log Groups for RDS logs
# =============================================================================

resource "aws_cloudwatch_log_group" "postgresql" {
  name              = "/aws/rds/instance/${local.db_identifier}/postgresql"
  retention_in_days = 30

  tags = local.database_tags
}

resource "aws_cloudwatch_log_group" "upgrade" {
  name              = "/aws/rds/instance/${local.db_identifier}/upgrade"
  retention_in_days = 30

  tags = local.database_tags
}

# =============================================================================
# RDS Instance
# =============================================================================

resource "aws_db_instance" "main" {
  # Basic configuration
  identifier     = local.db_identifier
  engine         = var.engine
  engine_version = var.engine_version
  instance_class = var.instance_class

  # Storage configuration
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = var.storage_encrypted
  kms_key_id           = var.storage_encrypted ? aws_kms_key.rds[0].arn : null

  # Database configuration
  db_name  = var.database_name
  username = var.master_username
  password = var.master_password

  # Network configuration
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  port                   = 5432

  # Backup configuration
  backup_retention_period = local.final_config.backup_retention_period
  backup_window          = var.backup_window
  maintenance_window     = var.maintenance_window
  copy_tags_to_snapshot  = true

  # High availability
  multi_az = local.final_config.multi_az

  # Parameter and option groups
  parameter_group_name = aws_db_parameter_group.main.name
  option_group_name    = aws_db_option_group.main.name

  # Monitoring and logging
  monitoring_interval             = var.monitoring_interval
  monitoring_role_arn            = var.monitoring_interval > 0 ? aws_iam_role.rds_enhanced_monitoring[0].arn : null
  performance_insights_enabled   = var.performance_insights_enabled
  performance_insights_retention_period = var.performance_insights_enabled ? 7 : null
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  # Security
  deletion_protection      = local.final_config.deletion_protection
  skip_final_snapshot     = local.final_config.skip_final_snapshot
  final_snapshot_identifier = local.final_config.final_snapshot_identifier

  # Maintenance
  auto_minor_version_upgrade = true
  apply_immediately         = false

  tags = merge(local.database_tags, {
    Name = local.db_identifier
  })

  depends_on = [
    aws_cloudwatch_log_group.postgresql,
    aws_cloudwatch_log_group.upgrade
  ]
}

# =============================================================================
# Enhanced Monitoring IAM Role
# =============================================================================

resource "aws_iam_role" "rds_enhanced_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  name = "${local.db_identifier}-enhanced-monitoring"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.database_tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  count = var.monitoring_interval > 0 ? 1 : 0

  role       = aws_iam_role.rds_enhanced_monitoring[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# =============================================================================
# Read Replica (for production environments)
# =============================================================================

resource "aws_db_instance" "read_replica" {
  count = var.create_read_replica ? 1 : 0

  identifier = "${local.db_identifier}-read-replica"

  # Replica configuration
  replicate_source_db = aws_db_instance.main.identifier

  # Instance configuration
  instance_class = var.read_replica_instance_class != "" ? var.read_replica_instance_class : var.instance_class

  # Storage configuration (inherited from source but can be modified)
  max_allocated_storage = var.max_allocated_storage

  # Network configuration
  publicly_accessible = false

  # Monitoring
  monitoring_interval             = var.monitoring_interval
  monitoring_role_arn            = var.monitoring_interval > 0 ? aws_iam_role.rds_enhanced_monitoring[0].arn : null
  performance_insights_enabled   = var.performance_insights_enabled

  # Security
  skip_final_snapshot = true

  tags = merge(local.database_tags, {
    Name = "${local.db_identifier}-read-replica"
    Type = "read-replica"
  })
}

# =============================================================================
# CloudWatch Alarms for Database Monitoring
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name          = "${local.db_identifier}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS CPU utilization"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = local.database_tags
}

resource "aws_cloudwatch_metric_alarm" "database_freeable_memory" {
  alarm_name          = "${local.db_identifier}-low-freeable-memory"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "FreeableMemory"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "134217728"  # 128 MB in bytes
  alarm_description   = "This metric monitors RDS freeable memory"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = local.database_tags
}

resource "aws_cloudwatch_metric_alarm" "database_free_storage_space" {
  alarm_name          = "${local.db_identifier}-low-free-storage"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "2147483648"  # 2 GB in bytes
  alarm_description   = "This metric monitors RDS free storage space"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = local.database_tags
}

resource "aws_cloudwatch_metric_alarm" "database_connection_count" {
  alarm_name          = "${local.db_identifier}-high-connection-count"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "50"
  alarm_description   = "This metric monitors RDS connection count"
  alarm_actions       = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }

  tags = local.database_tags
}

# =============================================================================
# Secrets Manager Integration (Optional)
# =============================================================================

resource "aws_secretsmanager_secret" "db_credentials" {
  count = var.store_credentials_in_secrets_manager ? 1 : 0

  name                    = "${local.db_identifier}-credentials"
  description             = "Database credentials for ${local.db_identifier}"
  recovery_window_in_days = var.environment == "production" ? 30 : 0

  tags = local.database_tags
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  count = var.store_credentials_in_secrets_manager ? 1 : 0

  secret_id = aws_secretsmanager_secret.db_credentials[0].id
  secret_string = jsonencode({
    username = aws_db_instance.main.username
    password = var.master_password
    endpoint = aws_db_instance.main.endpoint
    port     = aws_db_instance.main.port
    dbname   = aws_db_instance.main.db_name
    engine   = aws_db_instance.main.engine
  })
}

# =============================================================================
# Database Migration User (for application deployments)
# =============================================================================

# This would typically be managed by the application deployment process
# but we can create a placeholder for the migration user setup

resource "aws_ssm_parameter" "db_migration_commands" {
  name  = "/${var.name_prefix}/database/migration-commands"
  type  = "StringList"
  value = join(",", [
    "CREATE USER migration_user WITH PASSWORD 'CHANGE_ME';",
    "GRANT CONNECT ON DATABASE ${var.database_name} TO migration_user;",
    "GRANT USAGE ON SCHEMA public TO migration_user;",
    "GRANT CREATE ON SCHEMA public TO migration_user;",
    "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO migration_user;",
    "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO migration_user;",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO migration_user;",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO migration_user;"
  ])

  description = "Commands to set up database migration user"
  tags        = local.database_tags
}