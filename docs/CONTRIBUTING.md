# Contributing to Fake News Game Theory

We welcome contributions to the Fake News Game Theory project! This document provides guidelines for contributing to our research platform that combines game theory, machine learning, and network analysis to understand and combat fake news propagation.

## <¯ How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **= Bug reports and fixes**
- **( New features and enhancements**
- **=Ú Documentation improvements**
- **>ê Test coverage expansion**
- **=, Research methodology improvements**
- **=Ê Dataset contributions**
- **<¨ UI/UX improvements**
- **¡ Performance optimizations**

### Areas of Focus

- **Game Theory**: Nash equilibrium algorithms, strategy analysis, evolutionary stability
- **Machine Learning**: Classification models, ensemble methods, evaluation metrics
- **Network Analysis**: Propagation models, centrality measures, visualization
- **Frontend Development**: React/Next.js components, data visualization, user experience
- **Backend Development**: FastAPI endpoints, database optimization, API design
- **DevOps**: Docker optimization, CI/CD improvements, deployment automation

## =€ Getting Started

### Prerequisites

Before contributing, ensure you have:

- **Python 3.8+** and **Node.js 18+**
- **Git** version control
- **Docker** and **Docker Compose**
- A **GitHub account**
- Basic understanding of our tech stack (FastAPI, Next.js, PyTorch)

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/fake-news-game-theory.git
   cd fake-news-game-theory
   ```

2. **Set Up Development Environment**
   ```bash
   # Run setup script
   ./scripts/setup.sh

   # Or manual setup:
   # Backend
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt

   # Frontend
   cd ../frontend
   npm install
   ```

3. **Start Development Servers**
   ```bash
   # Using Docker (recommended)
   docker-compose -f config/docker-compose.dev.yml up

   # Or manual startup:
   # Terminal 1 - Backend
   cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

   # Terminal 2 - Frontend
   cd frontend && npm run dev
   ```

4. **Verify Setup**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## =Ë Contribution Workflow

### 1. Issue First Approach

Before starting work, please:

- **Check existing issues** to avoid duplicates
- **Create a new issue** for bugs, features, or questions
- **Discuss your approach** with maintainers for large changes
- **Get assignment** on issues you want to work on

### 2. Branch Strategy

```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# For bug fixes
git checkout -b fix/bug-description

# For documentation
git checkout -b docs/documentation-improvement
```

### 3. Development Process

#### Code Quality Standards

- **Python**: Follow PEP 8, use Black formatter, type hints required
- **TypeScript**: Use ESLint + Prettier, strict TypeScript configuration
- **Testing**: Maintain 80%+ coverage, write tests before implementation
- **Documentation**: Update docs for any API or behavior changes

#### Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: add Nash equilibrium visualization component
fix: resolve memory leak in network analysis
docs: update API documentation for classifier endpoints
test: add unit tests for game theory payoff calculations
refactor: optimize BERT model inference pipeline
```

### 4. Testing Requirements

Before submitting your PR:

```bash
# Backend testing
cd backend
pytest tests/ -v --coverage --cov-report=html

# Frontend testing
cd frontend
npm test
npm run test:e2e

# Integration testing
docker-compose -f config/docker-compose.test.yml up --abort-on-container-exit
```

### 5. Pull Request Process

1. **Update Documentation**
   - Update README.md if adding features
   - Add docstrings for new functions/classes
   - Update API documentation in `docs/api/`

2. **Ensure Quality**
   - All tests pass locally
   - Code follows style guidelines
   - No merge conflicts with main branch

3. **Create Pull Request**
   ```
   Title: Brief description of changes

   ## Description
   Detailed explanation of what this PR does and why.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No breaking changes (or marked as such)

   Fixes #issue_number
   ```

## >ê Research Contributions

### Dataset Contributions

If contributing datasets:

- **Ethical compliance**: Ensure datasets comply with privacy and copyright laws
- **Documentation**: Provide clear metadata and usage instructions
- **Format**: Use standard formats (CSV, JSON, Parquet)
- **Quality**: Clean, well-structured data with proper labels
- **License**: Ensure appropriate licensing for research use

### Algorithm Improvements

For game theory or ML algorithm contributions:

- **Mathematical rigor**: Provide theoretical justification
- **Empirical validation**: Include experimental results
- **Comparative analysis**: Benchmark against existing methods
- **Reproducibility**: Ensure results can be replicated
- **Documentation**: Explain algorithm in detail

## <¨ Design Guidelines

### UI/UX Principles

- **Accessibility**: Follow WCAG 2.1 guidelines
- **Responsive**: Mobile-first design approach
- **Performance**: Optimize for fast loading
- **Consistency**: Use design system components
- **Research-focused**: Prioritize data visualization clarity

### Visualization Standards

- **Color palette**: Use colorblind-friendly schemes
- **Interactivity**: Provide meaningful interactions
- **Performance**: Optimize for large datasets
- **Export**: Support multiple export formats
- **Documentation**: Explain visualization methodology

## = Security Guidelines

### Security Best Practices

- **Never commit secrets**: Use environment variables
- **Input validation**: Sanitize all user inputs
- **Authentication**: Follow JWT best practices
- **Dependencies**: Keep dependencies updated
- **Vulnerability reporting**: Follow responsible disclosure

### Reporting Security Issues

Please **DO NOT** create public issues for security vulnerabilities. Instead:

1. Email security concerns to: [security@fake-news-game-theory.org]
2. Include detailed description and reproduction steps
3. Allow 90 days for response before public disclosure

## =Þ Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Discord Server**: [Link to Discord] - Real-time chat
- **Email**: [maintainers@fake-news-game-theory.org]

### Documentation Resources

- **API Documentation**: `/docs/api/`
- **Architecture Guide**: `/docs/architecture.md`
- **Research Methodology**: `/docs/methodology/`
- **Deployment Guide**: `/docs/deployment.md`

## <Æ Recognition

### Contributors

All contributors are recognized in:

- **README.md**: Contributors section
- **CHANGELOG.md**: Version release notes
- **Academic papers**: Co-authorship for significant research contributions
- **Conference presentations**: Speaking opportunities for major contributions

### Levels of Contribution

- **Code Contributors**: Bug fixes, features, improvements
- **Research Contributors**: Algorithm development, theoretical contributions
- **Community Contributors**: Documentation, issue triage, user support
- **Mentor Contributors**: Code review, guidance for new contributors

## =Ü Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to [conduct@fake-news-game-theory.org].

## =Ä License

By contributing to this project, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

## =O Thank You

Your contributions help advance the fight against misinformation through rigorous research and innovative technology. Every contribution, no matter how small, makes a difference!

---

**Questions?** Don't hesitate to ask! We're here to help you contribute successfully to this important research project.