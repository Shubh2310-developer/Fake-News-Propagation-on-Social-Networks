#!/bin/bash

# =============================================================================
# Terraform Deployment Script for Fake News Game Theory Platform
# =============================================================================
# Usage: ./deploy.sh [staging|production] [plan|apply|destroy] [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
ACTION=""
AUTO_APPROVE=false
VERBOSE=false
FORCE=false
DRY_RUN=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(dirname "$TERRAFORM_DIR")"

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
Usage: $0 [ENVIRONMENT] [ACTION] [OPTIONS]

ENVIRONMENT:
    staging         Deploy to staging environment
    production      Deploy to production environment

ACTION:
    plan            Generate and show an execution plan
    apply           Apply the execution plan
    destroy         Destroy the infrastructure
    output          Show output values
    import          Import existing resources
    validate        Validate the configuration

OPTIONS:
    --auto-approve  Skip interactive approval of plan before applying
    --verbose       Enable verbose output
    --force         Force the action without confirmation prompts
    --dry-run       Show what would be done without making changes
    --target=RES    Target specific resource for operation
    --var-file=FILE Use additional variable file
    --help          Show this help message

Examples:
    $0 staging plan                          # Plan staging deployment
    $0 production apply --auto-approve       # Apply production without prompts
    $0 staging destroy --force               # Force destroy staging
    $0 production output                     # Show production outputs

Environment Variables:
    TF_VAR_*                 Terraform variables
    AWS_PROFILE              AWS profile to use
    AWS_REGION               AWS region override
    TERRAFORM_STATE_BUCKET   S3 bucket for state storage

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
            plan|apply|destroy|output|import|validate)
                ACTION="$1"
                shift
                ;;
            --auto-approve)
                AUTO_APPROVE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
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
            --target=*)
                TARGET="${1#*=}"
                shift
                ;;
            --var-file=*)
                EXTRA_VAR_FILE="${1#*=}"
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

    if [[ -z "$ACTION" ]]; then
        error "Action must be specified (plan, apply, destroy, output, import, validate)"
        usage
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed or not in PATH"
        exit 1
    fi

    # Check terraform version
    local tf_version
    tf_version=$(terraform version -json | jq -r '.terraform_version')
    info "Using Terraform version: $tf_version"

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

    # Check if jq is installed (for JSON processing)
    if ! command -v jq &> /dev/null; then
        error "jq is not installed or not in PATH"
        exit 1
    fi

    success "Prerequisites check passed"
}

# Initialize Terraform
terraform_init() {
    info "Initializing Terraform..."

    cd "$TERRAFORM_DIR"

    # Backend configuration file
    local backend_config_file="$TERRAFORM_DIR/environments/$ENVIRONMENT/backend.conf"

    if [[ -f "$backend_config_file" ]]; then
        terraform init -backend-config="$backend_config_file" -upgrade
    else
        terraform init -upgrade
    fi

    success "Terraform initialized"
}

# Select or create workspace
setup_workspace() {
    info "Setting up Terraform workspace for $ENVIRONMENT..."

    cd "$TERRAFORM_DIR"

    # Create workspace if it doesn't exist
    if ! terraform workspace list | grep -q "$ENVIRONMENT"; then
        terraform workspace new "$ENVIRONMENT"
    else
        terraform workspace select "$ENVIRONMENT"
    fi

    local current_workspace
    current_workspace=$(terraform workspace show)
    info "Using workspace: $current_workspace"

    if [[ "$current_workspace" != "$ENVIRONMENT" ]]; then
        error "Failed to select workspace $ENVIRONMENT"
        exit 1
    fi
}

# Validate Terraform configuration
validate_config() {
    info "Validating Terraform configuration..."

    cd "$TERRAFORM_DIR"

    terraform validate

    # Additional validation with tfvars file
    local tfvars_file="$TERRAFORM_DIR/environments/$ENVIRONMENT/terraform.tfvars"
    if [[ -f "$tfvars_file" ]]; then
        terraform plan -var-file="$tfvars_file" -input=false -detailed-exitcode > /dev/null
        local exit_code=$?
        if [[ $exit_code -eq 1 ]]; then
            error "Configuration validation failed"
            exit 1
        elif [[ $exit_code -eq 2 ]]; then
            info "Configuration valid, changes detected"
        else
            info "Configuration valid, no changes"
        fi
    fi

    success "Configuration validation passed"
}

# Build Terraform command arguments
build_terraform_args() {
    local args=()

    # Add variable file
    local tfvars_file="$TERRAFORM_DIR/environments/$ENVIRONMENT/terraform.tfvars"
    if [[ -f "$tfvars_file" ]]; then
        args+=("-var-file=$tfvars_file")
    fi

    # Add extra variable file if specified
    if [[ -n "${EXTRA_VAR_FILE:-}" ]]; then
        if [[ -f "$EXTRA_VAR_FILE" ]]; then
            args+=("-var-file=$EXTRA_VAR_FILE")
        else
            error "Extra variable file not found: $EXTRA_VAR_FILE"
            exit 1
        fi
    fi

    # Add environment variables
    args+=("-var=environment=$ENVIRONMENT")

    # Add target if specified
    if [[ -n "${TARGET:-}" ]]; then
        args+=("-target=$TARGET")
    fi

    # Add common flags
    args+=("-input=false")

    if [[ "$VERBOSE" == "true" ]]; then
        args+=("-detailed-exitcode")
    fi

    echo "${args[@]}"
}

# Confirm action
confirm_action() {
    if [[ "$FORCE" == "true" ]] || [[ "$AUTO_APPROVE" == "true" ]]; then
        return 0
    fi

    local aws_account
    aws_account=$(aws sts get-caller-identity --query Account --output text)

    warn "You are about to $ACTION infrastructure in $ENVIRONMENT environment"
    warn "AWS Account: $aws_account"
    warn "AWS Region: $(aws configure get region)"

    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$ACTION" =~ ^(apply|destroy)$ ]]; then
        error "⚠️  PRODUCTION ENVIRONMENT ⚠️"
        error "This action will affect production infrastructure!"
        echo
    fi

    read -p "Are you sure you want to continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Operation cancelled"
        exit 0
    fi
}

# Execute Terraform plan
terraform_plan() {
    info "Running Terraform plan for $ENVIRONMENT..."

    cd "$TERRAFORM_DIR"

    local args
    args=($(build_terraform_args))

    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run mode - showing command that would be executed:"
        echo "terraform plan ${args[*]}"
        return 0
    fi

    # Create plan file
    local plan_file="/tmp/terraform-$ENVIRONMENT-$(date +%Y%m%d-%H%M%S).plan"

    terraform plan "${args[@]}" -out="$plan_file"

    if [[ -f "$plan_file" ]]; then
        info "Plan saved to: $plan_file"
        echo "export TERRAFORM_PLAN_FILE='$plan_file'" > "/tmp/terraform-$ENVIRONMENT-plan.env"
    fi

    success "Plan completed"
}

# Execute Terraform apply
terraform_apply() {
    info "Running Terraform apply for $ENVIRONMENT..."

    cd "$TERRAFORM_DIR"

    local args
    args=($(build_terraform_args))

    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run mode - showing command that would be executed:"
        echo "terraform apply ${args[*]}"
        return 0
    fi

    # Check for existing plan file
    local plan_env_file="/tmp/terraform-$ENVIRONMENT-plan.env"
    local plan_file=""

    if [[ -f "$plan_env_file" ]]; then
        source "$plan_env_file"
        plan_file="$TERRAFORM_PLAN_FILE"
    fi

    if [[ -n "$plan_file" ]] && [[ -f "$plan_file" ]]; then
        info "Using existing plan file: $plan_file"
        if [[ "$AUTO_APPROVE" == "true" ]]; then
            terraform apply "$plan_file"
        else
            terraform apply "$plan_file"
        fi
    else
        if [[ "$AUTO_APPROVE" == "true" ]]; then
            terraform apply "${args[@]}" -auto-approve
        else
            terraform apply "${args[@]}"
        fi
    fi

    success "Apply completed"

    # Show important outputs
    show_outputs
}

# Execute Terraform destroy
terraform_destroy() {
    warn "Running Terraform destroy for $ENVIRONMENT..."

    cd "$TERRAFORM_DIR"

    local args
    args=($(build_terraform_args))

    if [[ "$DRY_RUN" == "true" ]]; then
        info "Dry run mode - showing command that would be executed:"
        echo "terraform destroy ${args[*]}"
        return 0
    fi

    # Additional confirmation for destroy
    if [[ "$FORCE" != "true" ]]; then
        echo
        error "⚠️  DESTRUCTIVE OPERATION ⚠️"
        error "This will DESTROY all infrastructure in $ENVIRONMENT!"
        echo
        read -p "Type 'destroy' to confirm: " -r
        if [[ "$REPLY" != "destroy" ]]; then
            info "Destroy cancelled"
            exit 0
        fi
    fi

    if [[ "$AUTO_APPROVE" == "true" ]] || [[ "$FORCE" == "true" ]]; then
        terraform destroy "${args[@]}" -auto-approve
    else
        terraform destroy "${args[@]}"
    fi

    success "Destroy completed"
}

# Show Terraform outputs
show_outputs() {
    info "Terraform outputs for $ENVIRONMENT:"

    cd "$TERRAFORM_DIR"

    if terraform output -json &> /dev/null; then
        terraform output -json | jq -r '
            to_entries[] |
            "\(.key): \(.value.value // "null")"
        ' | while IFS=': ' read -r key value; do
            if [[ "$value" == "null" ]] || [[ "$value" == "" ]]; then
                echo "  $key: (empty)"
            else
                echo "  $key: $value"
            fi
        done
    else
        warn "No outputs available or Terraform state not initialized"
    fi
}

# Main execution function
main() {
    # Set up cleanup trap
    trap cleanup EXIT

    # Parse arguments
    parse_args "$@"

    # Enable verbose mode if requested
    if [[ "$VERBOSE" == "true" ]]; then
        set -x
    fi

    info "🚀 Starting Terraform deployment for $ENVIRONMENT environment"
    info "Action: $ACTION"

    # Check prerequisites
    check_prerequisites

    # Initialize Terraform
    terraform_init

    # Setup workspace
    setup_workspace

    # Validate configuration
    if [[ "$ACTION" != "output" ]]; then
        validate_config
    fi

    # Confirm action
    if [[ "$ACTION" =~ ^(apply|destroy)$ ]]; then
        confirm_action
    fi

    # Execute the requested action
    case "$ACTION" in
        plan)
            terraform_plan
            ;;
        apply)
            terraform_apply
            ;;
        destroy)
            terraform_destroy
            ;;
        output)
            show_outputs
            ;;
        validate)
            validate_config
            success "Validation completed"
            ;;
        *)
            error "Unsupported action: $ACTION"
            exit 1
            ;;
    esac

    success "🎉 Terraform $ACTION completed successfully for $ENVIRONMENT!"

    if [[ "$ENVIRONMENT" == "production" ]] && [[ "$ACTION" == "apply" ]]; then
        info "🚀 Production infrastructure is now live!"
        info "📊 Monitor your deployment:"
        show_outputs
    fi
}

# Cleanup function
cleanup() {
    if [[ "$VERBOSE" == "true" ]]; then
        set +x
    fi

    # Clean up temporary files
    rm -f "/tmp/terraform-$ENVIRONMENT-plan.env" 2>/dev/null || true
}

# Run main function
main "$@"