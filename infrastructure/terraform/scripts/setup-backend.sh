#!/bin/bash

# =============================================================================
# Terraform Backend Setup Script for Fake News Game Theory Platform
# =============================================================================
# This script creates the S3 bucket and DynamoDB table required for
# Terraform remote state storage and locking.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT=""
AWS_REGION="us-west-2"
FORCE=false
DRY_RUN=false

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
    staging         Setup backend for staging environment
    production      Setup backend for production environment

OPTIONS:
    --region=REGION AWS region (default: us-west-2)
    --force         Force creation without confirmation
    --dry-run       Show what would be created without making changes
    --help          Show this help message

Examples:
    $0 staging                           # Setup staging backend
    $0 production --region=us-east-1     # Setup production backend in us-east-1
    $0 staging --dry-run                 # Preview staging backend setup

Environment Variables:
    AWS_PROFILE     AWS profile to use
    AWS_REGION      AWS region override

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
            --region=*)
                AWS_REGION="${1#*=}"
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
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

    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed or not in PATH"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured or invalid"
        exit 1
    fi

    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        error "jq is not installed or not in PATH"
        exit 1
    fi

    local aws_account
    aws_account=$(aws sts get-caller-identity --query Account --output text)
    info "Using AWS Account: $aws_account"
    info "Using AWS Region: $AWS_REGION"

    success "Prerequisites check passed"
}

# Generate resource names
generate_names() {
    BUCKET_NAME="fake-news-terraform-state-$ENVIRONMENT"
    DYNAMODB_TABLE="terraform-state-lock-$ENVIRONMENT"
    KMS_ALIAS="alias/terraform-state-$ENVIRONMENT"

    info "Resource names:"
    info "  S3 Bucket: $BUCKET_NAME"
    info "  DynamoDB Table: $DYNAMODB_TABLE"
    info "  KMS Key Alias: $KMS_ALIAS"
}

# Check if resources already exist
check_existing_resources() {
    info "Checking for existing resources..."

    # Check S3 bucket
    if aws s3 ls "s3://$BUCKET_NAME" &> /dev/null; then
        warn "S3 bucket '$BUCKET_NAME' already exists"
        BUCKET_EXISTS=true
    else
        info "S3 bucket '$BUCKET_NAME' does not exist"
        BUCKET_EXISTS=false
    fi

    # Check DynamoDB table
    if aws dynamodb describe-table --table-name "$DYNAMODB_TABLE" &> /dev/null; then
        warn "DynamoDB table '$DYNAMODB_TABLE' already exists"
        TABLE_EXISTS=true
    else
        info "DynamoDB table '$DYNAMODB_TABLE' does not exist"
        TABLE_EXISTS=false
    fi

    # Check KMS key
    if aws kms describe-key --key-id "$KMS_ALIAS" &> /dev/null; then
        warn "KMS key '$KMS_ALIAS' already exists"
        KMS_EXISTS=true
    else
        info "KMS key '$KMS_ALIAS' does not exist"
        KMS_EXISTS=false
    fi
}

# Create KMS key for S3 encryption
create_kms_key() {
    if [[ "$KMS_EXISTS" == "true" ]]; then
        info "KMS key already exists, skipping creation"
        return 0
    fi

    info "Creating KMS key for S3 bucket encryption..."

    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would create KMS key with alias '$KMS_ALIAS'"
        return 0
    fi

    local key_policy
    key_policy=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):root"
            },
            "Action": "kms:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "s3.amazonaws.com"
            },
            "Action": [
                "kms:Decrypt",
                "kms:GenerateDataKey"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    )

    local key_id
    key_id=$(aws kms create-key \
        --policy "$key_policy" \
        --description "Terraform state encryption key for $ENVIRONMENT" \
        --query 'KeyMetadata.KeyId' \
        --output text)

    aws kms create-alias \
        --alias-name "$KMS_ALIAS" \
        --target-key-id "$key_id"

    aws kms tag-resource \
        --key-id "$key_id" \
        --tags TagKey=Environment,TagValue="$ENVIRONMENT" \
               TagKey=Purpose,TagValue="terraform-state" \
               TagKey=Project,TagValue="fake-news-game-theory"

    success "KMS key created with alias: $KMS_ALIAS"
}

# Create S3 bucket for state storage
create_s3_bucket() {
    if [[ "$BUCKET_EXISTS" == "true" ]]; then
        info "S3 bucket already exists, skipping creation"
        return 0
    fi

    info "Creating S3 bucket for Terraform state..."

    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would create S3 bucket '$BUCKET_NAME'"
        return 0
    fi

    # Create bucket
    if [[ "$AWS_REGION" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "$BUCKET_NAME"
    else
        aws s3api create-bucket \
            --bucket "$BUCKET_NAME" \
            --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi

    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket "$BUCKET_NAME" \
        --versioning-configuration Status=Enabled

    # Enable server-side encryption
    local kms_key_id
    kms_key_id=$(aws kms describe-key --key-id "$KMS_ALIAS" --query 'KeyMetadata.KeyId' --output text)

    aws s3api put-bucket-encryption \
        --bucket "$BUCKET_NAME" \
        --server-side-encryption-configuration '{
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "aws:kms",
                        "KMSMasterKeyID": "'$kms_key_id'"
                    },
                    "BucketKeyEnabled": true
                }
            ]
        }'

    # Block public access
    aws s3api put-public-access-block \
        --bucket "$BUCKET_NAME" \
        --public-access-block-configuration \
            BlockPublicAcls=true,\
            IgnorePublicAcls=true,\
            BlockPublicPolicy=true,\
            RestrictPublicBuckets=true

    # Add bucket policy
    local bucket_policy
    bucket_policy=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        }
    ]
}
EOF
    )

    aws s3api put-bucket-policy \
        --bucket "$BUCKET_NAME" \
        --policy "$bucket_policy"

    # Add lifecycle configuration
    aws s3api put-bucket-lifecycle-configuration \
        --bucket "$BUCKET_NAME" \
        --lifecycle-configuration '{
            "Rules": [
                {
                    "ID": "DeleteIncompleteMultipartUploads",
                    "Status": "Enabled",
                    "AbortIncompleteMultipartUpload": {
                        "DaysAfterInitiation": 1
                    }
                },
                {
                    "ID": "TransitionToIA",
                    "Status": "Enabled",
                    "Transitions": [
                        {
                            "Days": 30,
                            "StorageClass": "STANDARD_IA"
                        },
                        {
                            "Days": 90,
                            "StorageClass": "GLACIER"
                        }
                    ]
                }
            ]
        }'

    # Add tags
    aws s3api put-bucket-tagging \
        --bucket "$BUCKET_NAME" \
        --tagging 'TagSet=[
            {Key=Environment,Value='$ENVIRONMENT'},
            {Key=Purpose,Value=terraform-state},
            {Key=Project,Value=fake-news-game-theory}
        ]'

    success "S3 bucket created: $BUCKET_NAME"
}

# Create DynamoDB table for state locking
create_dynamodb_table() {
    if [[ "$TABLE_EXISTS" == "true" ]]; then
        info "DynamoDB table already exists, skipping creation"
        return 0
    fi

    info "Creating DynamoDB table for Terraform state locking..."

    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would create DynamoDB table '$DYNAMODB_TABLE'"
        return 0
    fi

    aws dynamodb create-table \
        --table-name "$DYNAMODB_TABLE" \
        --attribute-definitions AttributeName=LockID,AttributeType=S \
        --key-schema AttributeName=LockID,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST \
        --tags Key=Environment,Value="$ENVIRONMENT" \
               Key=Purpose,Value=terraform-state-lock \
               Key=Project,Value=fake-news-game-theory

    # Wait for table to be active
    info "Waiting for DynamoDB table to become active..."
    aws dynamodb wait table-exists --table-name "$DYNAMODB_TABLE"

    # Enable point-in-time recovery
    aws dynamodb put-backup-policy \
        --table-name "$DYNAMODB_TABLE" \
        --backup-policy PointInTimeRecoveryEnabled=true

    success "DynamoDB table created: $DYNAMODB_TABLE"
}

# Generate backend configuration files
generate_backend_config() {
    info "Generating backend configuration files..."

    local backend_dir="$(dirname "$(dirname "$(readlink -f "$0")")")/environments/$ENVIRONMENT"
    mkdir -p "$backend_dir"

    local backend_config="$backend_dir/backend.conf"

    if [[ "$DRY_RUN" == "true" ]]; then
        info "DRY RUN: Would create backend config at '$backend_config'"
        return 0
    fi

    cat > "$backend_config" <<EOF
# Terraform Backend Configuration for $ENVIRONMENT
# Generated on $(date)

bucket         = "$BUCKET_NAME"
key            = "fake-news-platform/terraform.tfstate"
region         = "$AWS_REGION"
encrypt        = true
dynamodb_table = "$DYNAMODB_TABLE"
kms_key_id     = "$KMS_ALIAS"

# Enable versioning and state locking
versioning = true
EOF

    # Also create a terraform backend block for reference
    local backend_tf="$backend_dir/backend.tf.example"
    cat > "$backend_tf" <<EOF
# Example Terraform backend configuration
# Copy this to your main.tf file

terraform {
  backend "s3" {
    bucket         = "$BUCKET_NAME"
    key            = "fake-news-platform/terraform.tfstate"
    region         = "$AWS_REGION"
    encrypt        = true
    dynamodb_table = "$DYNAMODB_TABLE"
    kms_key_id     = "$KMS_ALIAS"
  }
}
EOF

    success "Backend configuration files created in: $backend_dir"
    info "  - backend.conf (for terraform init -backend-config)"
    info "  - backend.tf.example (example backend block)"
}

# Verify backend setup
verify_backend() {
    info "Verifying backend setup..."

    # Test S3 bucket access
    if aws s3 ls "s3://$BUCKET_NAME" &> /dev/null; then
        success "✓ S3 bucket is accessible"
    else
        error "✗ S3 bucket is not accessible"
        return 1
    fi

    # Test DynamoDB table access
    if aws dynamodb describe-table --table-name "$DYNAMODB_TABLE" &> /dev/null; then
        success "✓ DynamoDB table is accessible"
    else
        error "✗ DynamoDB table is not accessible"
        return 1
    fi

    # Test KMS key access
    if aws kms describe-key --key-id "$KMS_ALIAS" &> /dev/null; then
        success "✓ KMS key is accessible"
    else
        error "✗ KMS key is not accessible"
        return 1
    fi

    success "Backend setup verification passed"
}

# Show setup summary
show_summary() {
    info "==================================="
    info "Terraform Backend Setup Complete!"
    info "==================================="
    echo
    info "Environment: $ENVIRONMENT"
    info "AWS Region: $AWS_REGION"
    echo
    info "Created Resources:"
    info "  📦 S3 Bucket: $BUCKET_NAME"
    info "  🗃️  DynamoDB Table: $DYNAMODB_TABLE"
    info "  🔑 KMS Key: $KMS_ALIAS"
    echo
    info "Next Steps:"
    info "  1. Update your terraform backend configuration"
    info "  2. Run 'terraform init' to migrate to remote state"
    info "  3. Use the deployment script: ./deploy.sh $ENVIRONMENT plan"
    echo
    info "Backend configuration saved to:"
    info "  environments/$ENVIRONMENT/backend.conf"
}

# Confirm setup
confirm_setup() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi

    local aws_account
    aws_account=$(aws sts get-caller-identity --query Account --output text)

    warn "You are about to create Terraform backend resources for $ENVIRONMENT"
    warn "AWS Account: $aws_account"
    warn "AWS Region: $AWS_REGION"

    if [[ "$ENVIRONMENT" == "production" ]]; then
        error "⚠️  PRODUCTION ENVIRONMENT ⚠️"
        error "This will create production backend resources!"
        echo
    fi

    read -p "Are you sure you want to continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Setup cancelled"
        exit 0
    fi
}

# Main execution function
main() {
    # Parse arguments
    parse_args "$@"

    info "🚀 Setting up Terraform backend for $ENVIRONMENT environment"

    # Check prerequisites
    check_prerequisites

    # Generate resource names
    generate_names

    # Check existing resources
    check_existing_resources

    # Confirm setup
    confirm_setup

    # Create resources
    create_kms_key
    create_s3_bucket
    create_dynamodb_table

    # Generate configuration files
    generate_backend_config

    # Verify setup
    verify_backend

    # Show summary
    show_summary

    success "🎉 Terraform backend setup completed successfully!"
}

# Run main function
main "$@"