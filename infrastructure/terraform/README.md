# Terraform Infrastructure - Fake News Game Theory Platform

This directory contains the complete Terraform configuration for provisioning and managing the cloud infrastructure required to run the Fake News Game Theory Platform on AWS.

## 📁 Directory Structure

```
terraform/
├── main.tf                    # Root module orchestrating all infrastructure
├── variables.tf               # Input variables and validation
├── outputs.tf                 # Output values from infrastructure
├── modules/                   # Reusable infrastructure components
│   ├── vpc/                   # Network infrastructure
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── compute/               # EKS cluster and node groups
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── userdata.sh
│   └── database/              # RDS PostgreSQL instance
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── environments/              # Environment-specific configurations
│   ├── staging/
│   │   └── terraform.tfvars
│   └── production/
│       └── terraform.tfvars
├── scripts/                   # Automation scripts
│   ├── deploy.sh             # Main deployment script
│   └── setup-backend.sh      # Backend setup script
└── README.md                 # This file
```

## 🏗️ Infrastructure Overview

### Core Components

1. **VPC Module** (`modules/vpc/`)
   - Creates isolated network environment with public, private, and database subnets
   - Multi-AZ deployment for high availability
   - NAT gateways for private subnet internet access
   - VPC endpoints for cost optimization
   - Security groups and NACLs for network security

2. **Compute Module** (`modules/compute/`)
   - EKS cluster with managed node groups
   - Auto-scaling capabilities with multiple instance types
   - IRSA (IAM Roles for Service Accounts) for secure pod access
   - Cluster autoscaler and AWS Load Balancer Controller setup
   - CloudWatch logging and monitoring

3. **Database Module** (`modules/database/`)
   - Production-ready RDS PostgreSQL instance
   - Multi-AZ deployment for high availability
   - Automated backups and point-in-time recovery
   - Performance Insights and enhanced monitoring
   - Security groups and encryption at rest

### Additional Resources

- **S3 Bucket**: Application data storage with versioning and encryption
- **ElastiCache Redis**: Caching layer for improved performance
- **CloudWatch**: Centralized logging and monitoring
- **Secrets Manager**: Secure credential storage
- **KMS**: Encryption key management

## 🚀 Quick Start

### Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.0 installed
3. **jq** for JSON processing
4. Appropriate AWS permissions for creating infrastructure

### 1. Setup Remote State Backend

First, create the S3 bucket and DynamoDB table for Terraform state:

```bash
# Setup staging backend
./scripts/setup-backend.sh staging

# Setup production backend
./scripts/setup-backend.sh production
```

### 2. Deploy Infrastructure

```bash
# Plan staging deployment
./scripts/deploy.sh staging plan

# Apply staging deployment
./scripts/deploy.sh staging apply

# Plan production deployment
./scripts/deploy.sh production plan

# Apply production deployment
./scripts/deploy.sh production apply --force
```

## 🌍 Environment Configurations

### Staging Environment

**Optimized for cost and development:**
- Smaller instance types (t3.small, t3.medium)
- Spot instances for cost savings
- Reduced storage allocations
- Single AZ for non-critical components
- Shorter log retention periods

```bash
# Key staging settings
database_instance_class = "db.t3.micro"
redis_node_type = "cache.t3.micro"
node_groups = {
  general = {
    instance_types = ["t3.small", "t3.medium"]
    capacity_type = "SPOT"
  }
}
```

### Production Environment

**Optimized for performance and reliability:**
- Performance-optimized instance types
- On-demand instances for reliability
- Multi-AZ deployment
- Enhanced monitoring and logging
- Long-term backup retention

```bash
# Key production settings
database_instance_class = "db.r5.large"
redis_node_type = "cache.r5.large"
node_groups = {
  general = {
    instance_types = ["t3.large", "t3.xlarge"]
    capacity_type = "ON_DEMAND"
  }
}
```

## 🔧 Configuration Management

### Variable Files

Environment-specific configurations are managed through `.tfvars` files:

- `environments/staging/terraform.tfvars` - Staging configuration
- `environments/production/terraform.tfvars` - Production configuration

### Backend Configuration

Remote state backend is configured per environment:

```bash
# Backend configuration files
environments/staging/backend.conf
environments/production/backend.conf
```

## 📊 Outputs and Connection Information

After deployment, important connection information is available via outputs:

```bash
# View all outputs
./scripts/deploy.sh staging output

# Key outputs include:
terraform output cluster_endpoint
terraform output database_endpoint
terraform output redis_primary_endpoint
```

### Connecting to EKS Cluster

```bash
# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name fake-news-game-theory-staging-eks

# Verify connection
kubectl get nodes
```

### Database Connection

```bash
# Get connection details
terraform output database_connection_string

# Connect using psql
psql "postgresql://app_user:PASSWORD@ENDPOINT:5432/fake_news_db"
```

## 🔐 Security Features

### Network Security
- VPC with private subnets for all application components
- Security groups with principle of least privilege
- Network ACLs for additional layer of security
- VPC endpoints to reduce data transfer costs

### Encryption
- EKS secrets encrypted with customer-managed KMS keys
- RDS encryption at rest with customer-managed keys
- S3 bucket encryption for application data
- ElastiCache encryption in transit and at rest

### Access Control
- IRSA for secure pod-to-AWS service communication
- IAM roles with minimal required permissions
- Secrets Manager for sensitive configuration
- Enhanced monitoring and audit logging

## 💰 Cost Optimization

### Staging Environment
- Spot instances where appropriate
- Smaller instance types
- Reduced storage allocations
- Shorter backup retention
- Minimal monitoring features

### Production Environment
- Right-sized instances for workload
- Reserved instances consideration
- Auto-scaling to match demand
- Lifecycle policies for data retention
- VPC endpoints to reduce data transfer

## 📈 Monitoring and Observability

### CloudWatch Integration
- EKS cluster logs (API, audit, authenticator, etc.)
- RDS Performance Insights
- Custom metrics for application performance
- CloudWatch alarms for critical thresholds

### Metrics Collected
- EKS cluster performance metrics
- Node and pod resource utilization
- Database performance metrics
- Application-specific metrics

## 🔄 CI/CD Integration

### GitOps Workflow
1. Infrastructure changes committed to Git
2. Terraform plan runs in CI/CD pipeline
3. Manual approval for production changes
4. Automated deployment after approval

### Deployment Scripts
- `deploy.sh` - Main deployment automation
- `setup-backend.sh` - Backend initialization
- Built-in safety checks and confirmations

## 🛠️ Maintenance and Operations

### Regular Tasks

1. **Update Terraform Modules**
   ```bash
   terraform init -upgrade
   ```

2. **Review and Apply Security Updates**
   ```bash
   ./scripts/deploy.sh production plan
   ```

3. **Monitor Resource Usage**
   ```bash
   aws cloudwatch get-metric-statistics --namespace AWS/EKS
   ```

### Backup and Recovery

- **Database**: Automated backups with point-in-time recovery
- **State Files**: Versioned in S3 with cross-region replication
- **Configuration**: Version controlled in Git

### Scaling Operations

```bash
# Update node group scaling
terraform apply -var="node_groups.general.max_size=20"

# Update database instance class
terraform apply -var="database_instance_class=db.r5.xlarge"
```

## 🚨 Troubleshooting

### Common Issues

1. **State Lock Issues**
   ```bash
   terraform force-unlock LOCK_ID
   ```

2. **EKS Node Group Issues**
   ```bash
   kubectl get nodes
   kubectl describe node NODE_NAME
   ```

3. **Database Connection Issues**
   ```bash
   aws rds describe-db-instances --db-instance-identifier INSTANCE_ID
   ```

### Debug Mode

Enable verbose logging:
```bash
./scripts/deploy.sh staging plan --verbose
export TF_LOG=DEBUG
```

## 📚 Additional Resources

- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [EKS Best Practices Guide](https://aws.github.io/aws-eks-best-practices/)
- [RDS PostgreSQL Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_PostgreSQL.html)
- [VPC Best Practices](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-security-best-practices.html)

## 🤝 Contributing

When making infrastructure changes:

1. Test in staging environment first
2. Use `--dry-run` to preview changes
3. Follow infrastructure as code best practices
4. Update documentation for new resources
5. Ensure backwards compatibility

---

**⚠️ Important**: Always test infrastructure changes in staging before applying to production. The deployment scripts include safety checks, but human review is essential for production deployments.