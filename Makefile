# Fake News Game Theory - Build Automation
# ===========================================

# Variables
PYTHON := python3
PIP := pip3
NODE := node
NPM := npm
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Project Configuration
PROJECT_NAME := fake-news-game-theory
BACKEND_DIR := backend
FRONTEND_DIR := frontend
CONFIG_DIR := config
DOCS_DIR := docs

# Docker Configuration
COMPOSE_DEV := $(CONFIG_DIR)/docker-compose.dev.yml
COMPOSE_PROD := $(CONFIG_DIR)/docker-compose.prod.yml
COMPOSE_TEST := $(CONFIG_DIR)/docker-compose.test.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[1;37m
NC := \033[0m # No Color

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)Fake News Game Theory - Build Commands$(NC)"
	@echo "========================================"
	@echo ""
	@echo "$(WHITE)=€ Quick Start:$(NC)"
	@echo "  make setup     - Complete project setup"
	@echo "  make dev       - Start development servers"
	@echo "  make test      - Run all tests"
	@echo ""
	@echo "$(WHITE)=Ý Available Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(NC) %s\n", $$1, $$2}'

# ===========================================
# =€ Quick Start Commands
# ===========================================

.PHONY: setup
setup: ## Complete project setup (dependencies, environment, database)
	@echo "$(GREEN)=€ Setting up Fake News Game Theory project...$(NC)"
	@$(MAKE) install-deps
	@$(MAKE) setup-env
	@$(MAKE) setup-db
	@echo "$(GREEN) Setup complete! Run 'make dev' to start development.$(NC)"

.PHONY: dev
dev: ## Start development servers (backend + frontend)
	@echo "$(BLUE)= Starting development servers...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) up --build

.PHONY: dev-detached
dev-detached: ## Start development servers in detached mode
	@echo "$(BLUE)= Starting development servers (detached)...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) up --build -d

.PHONY: stop
stop: ## Stop all running services
	@echo "$(YELLOW)=Ñ Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) down
	@$(DOCKER_COMPOSE) -f $(COMPOSE_PROD) down
	@$(DOCKER_COMPOSE) -f $(COMPOSE_TEST) down

# ===========================================
# =æ Installation & Dependencies
# ===========================================

.PHONY: install-deps
install-deps: install-backend install-frontend ## Install all dependencies

.PHONY: install-backend
install-backend: ## Install Python backend dependencies
	@echo "$(BLUE)=æ Installing backend dependencies...$(NC)"
	@cd $(BACKEND_DIR) && $(PYTHON) -m venv venv
	@cd $(BACKEND_DIR) && source venv/bin/activate && $(PIP) install -r requirements.txt
	@cd $(BACKEND_DIR) && source venv/bin/activate && $(PIP) install -r requirements-dev.txt
	@echo "$(GREEN) Backend dependencies installed$(NC)"

.PHONY: install-frontend
install-frontend: ## Install Node.js frontend dependencies
	@echo "$(BLUE)=æ Installing frontend dependencies...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) ci
	@echo "$(GREEN) Frontend dependencies installed$(NC)"

.PHONY: update-deps
update-deps: ## Update all dependencies
	@echo "$(BLUE)= Updating dependencies...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && $(PIP) install --upgrade -r requirements.txt
	@cd $(FRONTEND_DIR) && $(NPM) update
	@echo "$(GREEN) Dependencies updated$(NC)"

# ===========================================
# =' Environment Setup
# ===========================================

.PHONY: setup-env
setup-env: ## Set up environment files
	@echo "$(BLUE)=' Setting up environment files...$(NC)"
	@if [ ! -f $(BACKEND_DIR)/.env ]; then \
		cp $(BACKEND_DIR)/.env.example $(BACKEND_DIR)/.env; \
		echo "$(YELLOW)   Please configure $(BACKEND_DIR)/.env$(NC)"; \
	fi
	@if [ ! -f $(FRONTEND_DIR)/.env.local ]; then \
		cp $(FRONTEND_DIR)/.env.example $(FRONTEND_DIR)/.env.local; \
		echo "$(YELLOW)   Please configure $(FRONTEND_DIR)/.env.local$(NC)"; \
	fi
	@echo "$(GREEN) Environment files created$(NC)"

.PHONY: setup-db
setup-db: ## Set up database (requires Docker)
	@echo "$(BLUE)=Ä  Setting up database...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) up -d db
	@sleep 5
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) exec db psql -U postgres -c "CREATE DATABASE fakenews_dev;"
	@echo "$(GREEN) Database setup complete$(NC)"

# ===========================================
# >ê Testing
# ===========================================

.PHONY: test
test: test-backend test-frontend ## Run all tests

.PHONY: test-backend
test-backend: ## Run backend tests with coverage
	@echo "$(BLUE)>ê Running backend tests...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing
	@echo "$(GREEN) Backend tests completed$(NC)"

.PHONY: test-frontend
test-frontend: ## Run frontend tests
	@echo "$(BLUE)>ê Running frontend tests...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) test -- --coverage --watchAll=false
	@echo "$(GREEN) Frontend tests completed$(NC)"

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)>ê Running e2e tests...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_TEST) up --build --abort-on-container-exit
	@echo "$(GREEN) E2E tests completed$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(BLUE)>ê Running integration tests...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) run test:e2e
	@echo "$(GREEN) Integration tests completed$(NC)"

# ===========================================
# <¨ Code Quality & Linting
# ===========================================

.PHONY: lint
lint: lint-backend lint-frontend ## Run all linters

.PHONY: lint-backend
lint-backend: ## Lint Python code
	@echo "$(BLUE)<¨ Linting backend code...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && black app/ tests/
	@cd $(BACKEND_DIR) && source venv/bin/activate && isort app/ tests/
	@cd $(BACKEND_DIR) && source venv/bin/activate && flake8 app/ tests/
	@cd $(BACKEND_DIR) && source venv/bin/activate && mypy app/
	@echo "$(GREEN) Backend linting completed$(NC)"

.PHONY: lint-frontend
lint-frontend: ## Lint TypeScript/JavaScript code
	@echo "$(BLUE)<¨ Linting frontend code...$(NC)"
	@cd $(FRONTEND_DIR) && $(NPM) run lint
	@cd $(FRONTEND_DIR) && $(NPM) run type-check
	@echo "$(GREEN) Frontend linting completed$(NC)"

.PHONY: format
format: ## Format all code
	@echo "$(BLUE)<¨ Formatting code...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && black app/ tests/
	@cd $(BACKEND_DIR) && source venv/bin/activate && isort app/ tests/
	@cd $(FRONTEND_DIR) && $(NPM) run format
	@echo "$(GREEN) Code formatted$(NC)"

# ===========================================
# =¢ Production & Deployment
# ===========================================

.PHONY: build
build: ## Build production images
	@echo "$(BLUE)<×  Building production images...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_PROD) build
	@echo "$(GREEN) Production build completed$(NC)"

.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)=€ Deploying to staging...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_PROD) up -d
	@echo "$(GREEN) Staging deployment completed$(NC)"

.PHONY: deploy-prod
deploy-prod: ## Deploy to production (use with caution!)
	@echo "$(RED)=¨ Deploying to production...$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) -f $(COMPOSE_PROD) up -d; \
		echo "$(GREEN) Production deployment completed$(NC)"; \
	else \
		echo "$(YELLOW)Deployment cancelled$(NC)"; \
	fi

# ===========================================
# =Ä  Database Operations
# ===========================================

.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(BLUE)=Ä  Running database migrations...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && alembic upgrade head
	@echo "$(GREEN) Database migrations completed$(NC)"

.PHONY: db-reset
db-reset: ## Reset database (destructive!)
	@echo "$(RED)=Ä  Resetting database...$(NC)"
	@read -p "This will delete all data. Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) down -v; \
		$(MAKE) setup-db; \
		$(MAKE) db-migrate; \
		echo "$(GREEN) Database reset completed$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

.PHONY: db-seed
db-seed: ## Seed database with sample data
	@echo "$(BLUE)<1 Seeding database...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && python scripts/seed_database.py
	@echo "$(GREEN) Database seeded$(NC)"

# ===========================================
# =Ê ML Model Operations
# ===========================================

.PHONY: train-models
train-models: ## Train ML models
	@echo "$(BLUE)> Training ML models...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && python scripts/train_models.py
	@echo "$(GREEN) Model training completed$(NC)"

.PHONY: evaluate-models
evaluate-models: ## Evaluate ML models
	@echo "$(BLUE)=Ê Evaluating ML models...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && python scripts/evaluate_models.py
	@echo "$(GREEN) Model evaluation completed$(NC)"

# ===========================================
# <® Game Theory Simulations
# ===========================================

.PHONY: run-simulations
run-simulations: ## Run game theory simulations
	@echo "$(BLUE)<® Running game theory simulations...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && python scripts/run_simulation.py
	@echo "$(GREEN) Simulations completed$(NC)"

.PHONY: generate-results
generate-results: ## Generate analysis results
	@echo "$(BLUE)=È Generating analysis results...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && python scripts/generate_results.py
	@echo "$(GREEN) Results generated$(NC)"

# ===========================================
# =Ú Documentation
# ===========================================

.PHONY: docs
docs: ## Build documentation
	@echo "$(BLUE)=Ú Building documentation...$(NC)"
	@cd $(DOCS_DIR) && make html
	@echo "$(GREEN) Documentation built$(NC)"

.PHONY: docs-serve
docs-serve: ## Serve documentation locally
	@echo "$(BLUE)=Ú Serving documentation...$(NC)"
	@cd $(DOCS_DIR)/_build/html && python -m http.server 8080

# ===========================================
# >ù Cleanup & Maintenance
# ===========================================

.PHONY: clean
clean: ## Clean build artifacts and caches
	@echo "$(YELLOW)>ù Cleaning build artifacts...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type f -name "*.coverage" -delete
	@rm -rf $(BACKEND_DIR)/htmlcov/
	@rm -rf $(FRONTEND_DIR)/.next/
	@rm -rf $(FRONTEND_DIR)/coverage/
	@echo "$(GREEN) Cleanup completed$(NC)"

.PHONY: clean-docker
clean-docker: ## Clean Docker images and containers
	@echo "$(YELLOW)>ù Cleaning Docker resources...$(NC)"
	@$(DOCKER) system prune -f
	@$(DOCKER) volume prune -f
	@echo "$(GREEN) Docker cleanup completed$(NC)"

.PHONY: reset
reset: clean clean-docker ## Complete reset (clean + remove containers)
	@echo "$(RED)= Performing complete reset...$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) down -v --remove-orphans
	@$(DOCKER_COMPOSE) -f $(COMPOSE_PROD) down -v --remove-orphans
	@$(DOCKER_COMPOSE) -f $(COMPOSE_TEST) down -v --remove-orphans
	@echo "$(GREEN) Complete reset finished$(NC)"

# ===========================================
# =Ë Information & Status
# ===========================================

.PHONY: status
status: ## Show project status
	@echo "$(CYAN)=Ë Project Status$(NC)"
	@echo "=================="
	@echo "$(WHITE)Project:$(NC) $(PROJECT_NAME)"
	@echo "$(WHITE)Backend:$(NC) $(shell cd $(BACKEND_DIR) && python --version 2>/dev/null || echo 'Python not found')"
	@echo "$(WHITE)Frontend:$(NC) $(shell cd $(FRONTEND_DIR) && node --version 2>/dev/null || echo 'Node.js not found')"
	@echo "$(WHITE)Docker:$(NC) $(shell docker --version 2>/dev/null || echo 'Docker not found')"
	@echo ""
	@echo "$(WHITE)Services Status:$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) ps 2>/dev/null || echo "No services running"

.PHONY: logs
logs: ## Show logs from all services
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) logs -f

.PHONY: logs-backend
logs-backend: ## Show backend logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) logs -f backend

.PHONY: logs-frontend
logs-frontend: ## Show frontend logs
	@$(DOCKER_COMPOSE) -f $(COMPOSE_DEV) logs -f frontend

# ===========================================
# = Security & Auditing
# ===========================================

.PHONY: security-audit
security-audit: ## Run security audits
	@echo "$(BLUE)= Running security audits...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && safety check
	@cd $(FRONTEND_DIR) && $(NPM) audit
	@echo "$(GREEN) Security audit completed$(NC)"

.PHONY: dependency-check
dependency-check: ## Check for dependency updates
	@echo "$(BLUE)=æ Checking dependencies...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && pip list --outdated
	@cd $(FRONTEND_DIR) && $(NPM) outdated
	@echo "$(GREEN) Dependency check completed$(NC)"

# ===========================================
# =€ CI/CD Helpers
# ===========================================

.PHONY: ci-setup
ci-setup: ## Setup for CI environment
	@echo "$(BLUE)=' Setting up CI environment...$(NC)"
	@$(MAKE) install-deps
	@$(MAKE) setup-env
	@echo "$(GREEN) CI setup completed$(NC)"

.PHONY: ci-test
ci-test: ## Run CI test suite
	@echo "$(BLUE)>ê Running CI tests...$(NC)"
	@$(MAKE) lint
	@$(MAKE) test
	@$(MAKE) security-audit
	@echo "$(GREEN) CI tests completed$(NC)"

# ===========================================
# =Ê Monitoring & Performance
# ===========================================

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)=Ê Running benchmarks...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && python scripts/benchmark.py
	@echo "$(GREEN) Benchmarks completed$(NC)"

.PHONY: profile
profile: ## Profile application performance
	@echo "$(BLUE)=Ê Profiling application...$(NC)"
	@cd $(BACKEND_DIR) && source venv/bin/activate && python scripts/profile.py
	@echo "$(GREEN) Profiling completed$(NC)"