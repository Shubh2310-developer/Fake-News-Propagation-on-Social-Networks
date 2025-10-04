# Fake News Game Theory Platform

An innovative research platform integrating game theory, machine learning, and network analysis to predict fake news propagation and inform policy decisions.

## 🚀 Quick Start

```bash
# Start everything with one command
./start.sh
```

Then open http://localhost:3000/simulation

**First time?** Run setup first:
```bash
./scripts/setup.sh
```

## 📚 Documentation

All documentation is in the [`docs/`](docs/) directory.

**Start here:**
- 📖 [Documentation Index](docs/README.md) - Complete documentation catalog
- 📖 [Getting Started Guide](docs/START_HERE.md) - Quick 1-page guide
- 📖 [Quick Start Guide](docs/QUICKSTART.md) - Detailed setup
- 🚀 [**Deployment Guide**](docs/QUICK_DEPLOY.md) - **Deploy to production** ⭐
- 📖 [API Documentation](docs/api/endpoints.md) - API reference
- 📖 [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute

## 🚀 Production Deployment

**Deploy in 30 minutes with Vercel + Render** (~$21/month):

1. Create accounts: [Vercel](https://vercel.com) & [Render](https://render.com)
2. Deploy database + backend on Render
3. Deploy frontend on Vercel
4. Configure environment variables

**See:** [Quick Deploy Guide](docs/QUICK_DEPLOY.md) | [Full Deployment Guide](docs/DEPLOYMENT_GUIDE.md)

**Alternative Options:**
- Docker on DigitalOcean ($12/month)
- AWS/GCP (Enterprise scale)

[→ View all deployment options](docs/QUICK_DEPLOY.md)

## ✨ Features

- 🎮 **Game Theory Simulations** - Multi-agent strategic interactions
- 🕸️ **Network Analysis** - Social network topology & propagation
- 🤖 **Machine Learning** - 7 trained models (DistilBERT 99.98% F1, LSTM 99.96% F1)
- 📊 **Nash Equilibrium** - Strategy stability analysis
- 🔄 **Real-time Updates** - Live simulation monitoring
- 📈 **Visualizations** - Interactive charts & graphs

## 🛠️ Tech Stack

**Backend:**
- FastAPI (Python)
- PostgreSQL + Redis
- Game Theory Engine
- Network Analysis (NetworkX)
- ML Models: DistilBERT, LSTM, Random Forest, Gradient Boosting, Ensemble

**Frontend:**
- Next.js 14 + React
- TypeScript
- Tailwind CSS
- Real-time visualization

**ML Training:**
- PyTorch (LSTM, DistilBERT)
- Scikit-learn (Traditional ML)
- Transformers (Hugging Face)
- Dataset: 44,898 samples (80/10/10 split)

## 🤖 Machine Learning Models

The platform includes **7 trained models** for fake news classification:

| Model | Type | Performance | Training Samples |
|-------|------|-------------|-----------------|
| **DistilBERT** | Transformer | 99.98% F1 | 35,918 |
| **LSTM** | Neural Network | 99.96% F1 | 35,918 |
| **Random Forest** | Ensemble Trees | 87.3% Accuracy | 3,500 |
| **Gradient Boosting** | Boosting | 86.0% Accuracy | 3,500 |
| **Ensemble** | Voting | 82.9% Accuracy | 3,500 |
| **Logistic Regression** | Linear | 77.0% Accuracy | 3,500 |
| **Naive Bayes** | Probabilistic | 76.8% Accuracy | 3,500 |

**Training Details:**
- Total Dataset: 44,898 samples (Fake: 23,490, True: 21,418)
- Split: 80% train, 10% validation, 10% test
- Hardware: RTX 4050 (6GB VRAM), 16GB RAM
- Training Time: ~22 minutes (all models)

**To retrain models:**
```bash
python notebooks/complete_training_pipeline.py
```

## 📊 Usage

### Classify News

1. Navigate to http://localhost:3000/classifier
2. Enter text to analyze
3. Select a model (Ensemble recommended)
4. View prediction with confidence scores

### Run a Simulation

1. Navigate to http://localhost:3000/simulation
2. Configure parameters (or use defaults)
3. Click "Start Simulation"
4. Watch results in real-time

### Access Points

- 🌐 **Frontend**: http://localhost:3000
- 🤖 **Classifier**: http://localhost:3000/classifier
- 📊 **Simulation**: http://localhost:3000/simulation
- 🎯 **Equilibrium**: http://localhost:3000/equilibrium
- 🕸️ **Network Analysis**: http://localhost:3000/network
- 🔧 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

## 🔧 Development

### Start Backend & Frontend

```bash
./start.sh
```

### Backend Only

```bash
cd backend
conda activate fake_news
python -m uvicorn app.main:app --reload
```

### Frontend Only

```bash
cd frontend
npm run dev
```

## 📖 Documentation

Complete documentation is available in the `docs/` directory:

- **[START_HERE.md](docs/START_HERE.md)** - Quick start
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Detailed setup
- **[SCRIPTS_GUIDE.md](docs/SCRIPTS_GUIDE.md)** - Script documentation
- **[DATABASE_SETUP.md](docs/DATABASE_SETUP.md)** - Database configuration
- **[INTEGRATION_STATUS.md](docs/INTEGRATION_STATUS.md)** - Features status
- **[WORKFLOW.md](docs/WORKFLOW.md)** - Architecture & workflow
- **[INDEX.md](docs/INDEX.md)** - Complete documentation index

## 🐛 Troubleshooting

See [QUICKSTART.md](docs/QUICKSTART.md) for detailed troubleshooting.

**Quick fixes:**

```bash
# Database issues
./scripts/init-db.sh

# Port conflicts
pkill -f "uvicorn\|next dev"

# Frontend cache
cd frontend && rm -rf .next && npm run dev
```

## 🤝 Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for contribution guidelines.

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

This project integrates research in game theory, network science, and machine learning to combat misinformation.

---

**Ready to explore?** Run `./start.sh` and visit http://localhost:3000/simulation!

For detailed instructions, see [docs/START_HERE.md](docs/START_HERE.md)
