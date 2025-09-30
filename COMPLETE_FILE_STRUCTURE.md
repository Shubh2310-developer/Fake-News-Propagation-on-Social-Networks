# Complete File Structure - Fake News Game Theory Project

**Generated:** 2025-09-30
**Total Files:** 440 (excluding node_modules, __pycache__, .git, .next cache)

## 📊 Project Statistics

- **Backend Python Files:** ~80 files
- **Frontend TypeScript/JavaScript Files:** ~150 files
- **Jupyter Notebooks:** 8 notebooks
- **Configuration Files:** ~30 files
- **Documentation Files:** ~20 files
- **Test Files:** ~25 files
- **Infrastructure Files:** ~40 files
- **Data Files:** ~30 files

---

## 📂 Complete Directory Tree

```
fake-news-game-theory/
│
├── 📁 .claude/
│   └── settings.local.json
│
├── 📁 .github/
│   ├── 📁 ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── research_question.md
│   ├── 📁 PULL_REQUEST_TEMPLATE/
│   │   └── pull_request_template.md
│   ├── 📁 workflows/
│   │   ├── cd.yml
│   │   ├── ci.yml
│   │   ├── security.yml
│   │   └── test.yml
│   ├── FUNDING.yml
│   └── gitleaks.toml
│
├── 📁 assets/
│   ├── 📁 images/
│   │   └── 📁 results-preview/
│   ├── 📁 papers/
│   │   ├── 📁 literature-review/
│   │   ├── 📁 methodology/
│   │   └── 📁 results/
│   └── 📁 presentations/
│
├── 📁 backend/
│   ├── 📁 app/
│   │   ├── 📁 api/
│   │   │   ├── 📁 v1/
│   │   │   │   ├── analysis.py
│   │   │   │   ├── classifier.py
│   │   │   │   ├── data.py
│   │   │   │   ├── equilibrium.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── simulation.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📁 core/
│   │   │   ├── cache.py
│   │   │   ├── config.py
│   │   │   ├── database.py
│   │   │   ├── __init__.py
│   │   │   ├── logging.py
│   │   │   └── security.py
│   │   │
│   │   ├── 📁 models/
│   │   │   ├── analysis.py
│   │   │   ├── classifier.py
│   │   │   ├── common.py
│   │   │   ├── data.py
│   │   │   ├── game_theory.py
│   │   │   ├── __init__.py
│   │   │   ├── news.py
│   │   │   ├── simulation.py
│   │   │   ├── social.py
│   │   │   └── user.py
│   │   │
│   │   ├── 📁 routers/
│   │   │   ├── auth.py
│   │   │   ├── __init__.py
│   │   │   ├── news.py
│   │   │   ├── simulation.py
│   │   │   └── social.py
│   │   │
│   │   ├── 📁 services/
│   │   │   ├── classifier_service.py
│   │   │   ├── data_service.py
│   │   │   ├── equilibrium_service.py
│   │   │   ├── __init__.py
│   │   │   ├── network_service.py
│   │   │   └── simulation_service.py
│   │   │
│   │   ├── 📁 utils/
│   │   │   ├── data_preprocessing.py
│   │   │   ├── file_handlers.py
│   │   │   ├── __init__.py
│   │   │   ├── validators.py
│   │   │   └── visualization.py
│   │   │
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── 📁 game_theory/
│   │   ├── analysis.py
│   │   ├── equilibrium.py
│   │   ├── __init__.py
│   │   ├── payoffs.py
│   │   ├── players.py
│   │   ├── simulation.py
│   │   └── strategies.py
│   │
│   ├── 📁 ml_models/
│   │   ├── 📁 classifiers/
│   │   │   ├── base_classifier.py
│   │   │   ├── bert_classifier.py
│   │   │   ├── ensemble.py
│   │   │   ├── __init__.py
│   │   │   ├── logistic_regression.py
│   │   │   └── lstm_classifier.py
│   │   │
│   │   ├── 📁 evaluation/
│   │   │   ├── cross_validation.py
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   └── visualization.py
│   │   │
│   │   ├── 📁 preprocessing/
│   │   │   ├── data_augmentation.py
│   │   │   ├── feature_extraction.py
│   │   │   ├── __init__.py
│   │   │   └── text_processing.py
│   │   │
│   │   └── __init__.py
│   │
│   ├── 📁 network/
│   │   ├── graph_generator.py
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── propagation.py
│   │   └── visualization.py
│   │
│   ├── 📁 scripts/
│   │   ├── data_pipeline.py
│   │   ├── generate_results.py
│   │   ├── run_simulation.py
│   │   └── train_models.py
│   │
│   ├── 📁 tests/
│   │   ├── 📁 api/
│   │   │   ├── test_classifier.py
│   │   │   ├── test_data.py
│   │   │   └── test_simulation.py
│   │   │
│   │   ├── 📁 core/
│   │   │   ├── test_equilibrium.py
│   │   │   └── test_simulation.py
│   │   │
│   │   ├── 📁 game_theory/
│   │   │   ├── test_equilibrium.py
│   │   │   ├── test_players.py
│   │   │   └── test_strategies.py
│   │   │
│   │   ├── 📁 network/
│   │   │   └── test_graph_generation.py
│   │   │
│   │   ├── 📁 services/
│   │   │   ├── test_classifier_service.py
│   │   │   ├── test_data_service.py
│   │   │   ├── test_equilibrium_service.py
│   │   │   └── test_simulation_service.py
│   │   │
│   │   ├── conftest.py
│   │   └── test_placeholder.py
│   │
│   ├── .env
│   ├── .env.example
│   ├── .gitignore
│   ├── Dockerfile
│   ├── pytest.ini
│   ├── README.md
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── requirements-ci.txt
│
├── 📁 config/
│   ├── 📁 database/
│   │   └── init.sql
│   ├── .env.example
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   ├── docker-compose.test.yml
│   └── nginx.conf
│
├── 📁 data/
│   ├── 📁 models/
│   │   └── 📁 best_random_forest_20250929_102955/
│   │       ├── metadata.json
│   │       └── model.pkl
│   │
│   ├── 📁 networks/
│   │   ├── 📁 real_networks/
│   │   └── 📁 synthetic_networks/
│   │
│   ├── 📁 processed/
│   │   ├── 📁 features/
│   │   │   ├── all_features.csv
│   │   │   ├── correlation_heatmap.png
│   │   │   ├── domain_distribution.png
│   │   │   ├── feature_correlations.csv
│   │   │   ├── feature_correlations.png
│   │   │   ├── feature_distributions.png
│   │   │   ├── feature_names.pkl
│   │   │   ├── feature_summary.json
│   │   │   ├── label_distribution.png
│   │   │   ├── scaler.pkl
│   │   │   ├── selected_feature_names.pkl
│   │   │   ├── selected_features.csv
│   │   │   ├── tfidf_vectorizer.pkl
│   │   │   └── word_clouds.png
│   │   │
│   │   ├── 📁 test/
│   │   │   ├── X_test.csv
│   │   │   └── y_test.csv
│   │   │
│   │   ├── 📁 train/
│   │   │   ├── X_train.csv
│   │   │   └── y_train.csv
│   │   │
│   │   ├── 📁 validation/
│   │   │   ├── X_val.csv
│   │   │   └── y_val.csv
│   │   │
│   │   ├── pipeline_status.json
│   │   ├── simulation_results.csv
│   │   └── simulation_results_new.csv
│   │
│   ├── 📁 raw/
│   │   ├── 📁 fakenewsnet/
│   │   │   ├── gossipcop_fake.csv
│   │   │   ├── gossipcop_real.csv
│   │   │   ├── politifact_fake.csv
│   │   │   └── politifact_real.csv
│   │   │
│   │   ├── 📁 kaggle_fake_news/
│   │   │   ├── Fake.csv
│   │   │   └── True.csv
│   │   │
│   │   └── 📁 liar_dataset/
│   │       ├── README
│   │       ├── test.tsv
│   │       ├── train.tsv
│   │       └── valid.tsv
│   │
│   ├── 📁 results/
│   │   ├── 📁 figures/
│   │   │   ├── enhanced_game_theory_analysis.png
│   │   │   └── enhanced_propagation_analysis.png
│   │   │
│   │   ├── 📁 models/
│   │   ├── 📁 reports/
│   │   │   ├── cascade_analysis_statistical.json
│   │   │   ├── COMPREHENSIVE_ANALYSIS_SUMMARY.json
│   │   │   ├── equilibrium_analysis_fixed.json
│   │   │   └── intervention_analysis_realistic.json
│   │   │
│   │   ├── 📁 simulations/
│   │   ├── 📁 visualizations/
│   │   ├── enhanced_analysis_report.txt
│   │   ├── enhanced_game_theory_results.csv
│   │   ├── enhanced_propagation_results.csv
│   │   ├── enhanced_results_complete.json
│   │   ├── model_comparison.csv
│   │   └── statistical_analysis.json
│   │
│   └── DATASET_ANALYSIS.md
│
├── 📁 docs/
│   ├── 📁 api/
│   │   ├── endpoints.md
│   │   └── examples.md
│   │
│   ├── 📁 methodology/
│   │   ├── game_theory.md
│   │   ├── machine_learning.md
│   │   └── network_analysis.md
│   │
│   ├── 📁 tutorials/
│   │   ├── model_training.md
│   │   ├── running_simulations.md
│   │   └── setup.md
│   │
│   ├── architecture.md
│   ├── contributing.md
│   ├── deployment.md
│   └── model_analysis_report.md
│
├── 📁 frontend/
│   ├── 📁 docs/
│   │   ├── API.md
│   │   ├── COMPONENTS.md
│   │   ├── DEPLOYMENT.md
│   │   └── GAME_THEORY.md
│   │
│   ├── 📁 public/
│   │   ├── 📁 icons/
│   │   │   ├── favicon.ico
│   │   │   ├── logo.svg
│   │   │   └── manifest.json
│   │   │
│   │   ├── 📁 images/
│   │   │   ├── game-theory-diagram.svg
│   │   │   ├── hero-bg.webp
│   │   │   └── network-visualization.png
│   │   │
│   │   └── {robots.txt}
│   │
│   ├── 📁 src/
│   │   ├── 📁 app/
│   │   │   ├── 📁 (dashboard)/
│   │   │   │   ├── 📁 analytics/
│   │   │   │   │   └── page.tsx
│   │   │   │   │
│   │   │   │   ├── 📁 classifier/
│   │   │   │   │   └── page.tsx
│   │   │   │   │
│   │   │   │   ├── 📁 equilibrium/
│   │   │   │   │   └── page.tsx
│   │   │   │   │
│   │   │   │   ├── 📁 simulation/
│   │   │   │   │   ├── 📁 components/
│   │   │   │   │   │   ├── GameParameters.tsx
│   │   │   │   │   │   ├── NetworkGraph.tsx
│   │   │   │   │   │   └── PayoffMatrix.tsx
│   │   │   │   │   └── page.tsx
│   │   │   │   │
│   │   │   │   └── layout.tsx
│   │   │   │
│   │   │   ├── 📁 about/
│   │   │   │   └── page.tsx
│   │   │   │
│   │   │   ├── 📁 api/
│   │   │   │   ├── 📁 auth/
│   │   │   │   │   └── 📁 [...nextauth]/
│   │   │   │   │       └── route.ts
│   │   │   │   │
│   │   │   │   ├── 📁 classifier/
│   │   │   │   │   ├── 📁 metrics/
│   │   │   │   │   │   └── route.ts
│   │   │   │   │   ├── 📁 predict/
│   │   │   │   │   │   └── route.ts
│   │   │   │   │   └── 📁 train/
│   │   │   │   │       └── route.ts
│   │   │   │   │
│   │   │   │   ├── 📁 data/
│   │   │   │   │   ├── 📁 datasets/
│   │   │   │   │   │   └── route.ts
│   │   │   │   │   ├── 📁 export/
│   │   │   │   │   │   └── route.ts
│   │   │   │   │   └── 📁 upload/
│   │   │   │   │       └── route.ts
│   │   │   │   │
│   │   │   │   └── 📁 simulation/
│   │   │   │       ├── 📁 parameters/
│   │   │   │       │   └── route.ts
│   │   │   │       ├── 📁 results/
│   │   │   │       │   └── route.ts
│   │   │   │       └── 📁 run/
│   │   │   │           └── route.ts
│   │   │   │
│   │   │   ├── 📁 datasets/
│   │   │   │   └── page.tsx
│   │   │   │
│   │   │   ├── 📁 research/
│   │   │   │   ├── 📁 methodology/
│   │   │   │   │   └── page.tsx
│   │   │   │   └── page.tsx
│   │   │   │
│   │   │   ├── error.tsx
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   ├── loading.tsx
│   │   │   ├── not-found.tsx
│   │   │   └── page.tsx
│   │   │
│   │   ├── 📁 components/
│   │   │   ├── 📁 charts/
│   │   │   │   ├── BarChart.tsx
│   │   │   │   ├── ConfusionMatrix.tsx
│   │   │   │   ├── Heatmap.tsx
│   │   │   │   ├── LineChart.tsx
│   │   │   │   ├── ModelPerformance.tsx
│   │   │   │   ├── NetworkVisualization.tsx
│   │   │   │   ├── PayoffHeatmap.tsx
│   │   │   │   ├── PropagationChart.tsx
│   │   │   │   └── ScatterPlot.tsx
│   │   │   │
│   │   │   ├── 📁 common/
│   │   │   │   ├── ErrorBoundary.tsx
│   │   │   │   ├── LoadingSpinner.tsx
│   │   │   │   ├── PageHeader.tsx
│   │   │   │   ├── SearchBar.tsx
│   │   │   │   └── ThemeToggle.tsx
│   │   │   │
│   │   │   ├── 📁 data-display/
│   │   │   │   ├── DataTable.tsx
│   │   │   │   ├── MetricsCard.tsx
│   │   │   │   ├── ModelPerformance.tsx
│   │   │   │   ├── ResultsViewer.tsx
│   │   │   │   └── StatisticsPanel.tsx
│   │   │   │
│   │   │   ├── 📁 forms/
│   │   │   │   ├── ClassifierConfigForm.tsx
│   │   │   │   ├── DataUploadForm.tsx
│   │   │   │   ├── GameParametersForm.tsx
│   │   │   │   └── SimulationConfigForm.tsx
│   │   │   │
│   │   │   ├── 📁 game-theory/
│   │   │   │   ├── EquilibriumAnalysis.tsx
│   │   │   │   ├── EquilibriumVisualizer.tsx
│   │   │   │   ├── GameResults.tsx
│   │   │   │   ├── PayoffMatrix.tsx
│   │   │   │   ├── PlayerActions.tsx
│   │   │   │   ├── StrategyEvolution.tsx
│   │   │   │   └── StrategySelector.tsx
│   │   │   │
│   │   │   ├── 📁 layout/
│   │   │   │   ├── Breadcrumbs.tsx
│   │   │   │   ├── Footer.tsx
│   │   │   │   ├── Header.tsx
│   │   │   │   ├── Navigation.tsx
│   │   │   │   └── Sidebar.tsx
│   │   │   │
│   │   │   ├── 📁 simulation/
│   │   │   │   ├── PropagationMetrics.tsx
│   │   │   │   └── SimulationControls.tsx
│   │   │   │
│   │   │   └── 📁 ui/
│   │   │       ├── alert.tsx
│   │   │       ├── badge.tsx
│   │   │       ├── button.tsx
│   │   │       ├── card.tsx
│   │   │       ├── dialog.tsx
│   │   │       ├── input.tsx
│   │   │       ├── label.tsx
│   │   │       ├── progress.tsx
│   │   │       ├── select.tsx
│   │   │       ├── separator.tsx
│   │   │       ├── skeleton.tsx
│   │   │       ├── slider.tsx
│   │   │       ├── spinner.tsx
│   │   │       ├── switch.tsx
│   │   │       ├── table.tsx
│   │   │       ├── tabs.tsx
│   │   │       ├── textarea.tsx
│   │   │       └── toast.tsx
│   │   │
│   │   ├── 📁 config/
│   │   │   ├── api.ts
│   │   │   ├── database.ts
│   │   │   ├── ml-models.ts
│   │   │   └── visualization.ts
│   │   │
│   │   ├── 📁 hooks/
│   │   │   ├── index.ts
│   │   │   ├── useApi.ts
│   │   │   ├── useClassifier.ts
│   │   │   ├── useDataUpload.ts
│   │   │   ├── useGameState.ts
│   │   │   ├── useLocalStorage.ts
│   │   │   ├── useSimulation.ts
│   │   │   └── useWebSocket.ts
│   │   │
│   │   ├── 📁 lib/
│   │   │   ├── api.ts
│   │   │   ├── auth.ts
│   │   │   ├── constants.ts
│   │   │   ├── dataProcessing.ts
│   │   │   ├── gameTheory.ts
│   │   │   ├── networkAnalysis.ts
│   │   │   ├── utils.ts
│   │   │   └── validations.ts
│   │   │
│   │   ├── 📁 store/
│   │   │   ├── classifierStore.ts
│   │   │   ├── dataStore.ts
│   │   │   ├── gameStore.ts
│   │   │   ├── index.ts
│   │   │   ├── simulationStore.ts
│   │   │   └── uiStore.ts
│   │   │
│   │   ├── 📁 styles/
│   │   │   ├── charts.css
│   │   │   ├── components.css
│   │   │   └── globals.css
│   │   │
│   │   └── 📁 types/
│   │       ├── api.ts
│   │       ├── classifier.ts
│   │       ├── data.ts
│   │       ├── gameTheory.ts
│   │       ├── index.ts
│   │       ├── network.ts
│   │       └── simulation.ts
│   │
│   ├── 📁 tests/
│   │   ├── 📁 components/
│   │   │   ├── NetworkVisualization.test.tsx
│   │   │   └── PayoffMatrix.test.tsx
│   │   │
│   │   ├── 📁 utils/
│   │   │   └── gameTheory.test.ts
│   │   │
│   │   └── setup.ts
│   │
│   ├── .env.example
│   ├── .env.local
│   ├── .eslintrc.json
│   ├── .gitignore
│   ├── .prettierrc
│   ├── Dockerfile
│   ├── jest.config.js
│   ├── jest.setup.js
│   ├── next.config.js
│   ├── next-env.d.ts
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── README.md
│   ├── tailwind.config.js
│   └── tsconfig.json
│
├── 📁 infrastructure/
│   ├── 📁 kubernetes/
│   │   ├── 📁 base/
│   │   │   ├── backend-deployment.yaml
│   │   │   ├── configmap.yaml
│   │   │   ├── frontend-deployment.yaml
│   │   │   ├── hpa.yaml
│   │   │   ├── ingress.yaml
│   │   │   ├── namespace.yaml
│   │   │   ├── persistent-volumes.yaml
│   │   │   ├── postgres-deployment.yaml
│   │   │   ├── redis-deployment.yaml
│   │   │   ├── secrets.yaml
│   │   │   └── service.yaml
│   │   │
│   │   ├── 📁 monitoring/
│   │   │   ├── grafana.yaml
│   │   │   └── prometheus.yaml
│   │   │
│   │   ├── 📁 production/
│   │   │   ├── deployment-patch.yaml
│   │   │   └── kustomization.yaml
│   │   │
│   │   ├── 📁 staging/
│   │   │   ├── deployment-patch.yaml
│   │   │   └── kustomization.yaml
│   │   │
│   │   ├── configmap.yaml
│   │   ├── deployment.yaml
│   │   ├── ingress.yaml
│   │   ├── namespace.yaml
│   │   └── service.yaml
│   │
│   ├── 📁 monitoring/
│   │   ├── 📁 alertmanager/
│   │   │   └── alertmanager.yml
│   │   │
│   │   ├── 📁 grafana/
│   │   │   ├── 📁 dashboards/
│   │   │   │   ├── 📁 application/
│   │   │   │   │   └── application-health.json
│   │   │   │   ├── 📁 business/
│   │   │   │   │   └── business-metrics.json
│   │   │   │   └── 📁 kubernetes/
│   │   │   │       └── cluster-overview.json
│   │   │   │
│   │   │   ├── 📁 datasources/
│   │   │   │   └── prometheus.yml
│   │   │   │
│   │   │   └── 📁 provisioning/
│   │   │       └── dashboards.yml
│   │   │
│   │   ├── 📁 rules/
│   │   │   └── application-alerts.yml
│   │   │
│   │   ├── docker-compose.yml
│   │   ├── prometheus.yml
│   │   └── README.md
│   │
│   ├── 📁 scripts/
│   │   ├── deploy.sh
│   │   └── setup-cluster.sh
│   │
│   ├── 📁 terraform/
│   │   ├── 📁 environments/
│   │   │   ├── 📁 production/
│   │   │   │   └── terraform.tfvars
│   │   │   └── 📁 staging/
│   │   │       └── terraform.tfvars
│   │   │
│   │   ├── 📁 modules/
│   │   │   ├── 📁 compute/
│   │   │   │   ├── main.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   ├── userdata.sh
│   │   │   │   └── variables.tf
│   │   │   │
│   │   │   ├── 📁 database/
│   │   │   │   ├── main.tf
│   │   │   │   ├── outputs.tf
│   │   │   │   └── variables.tf
│   │   │   │
│   │   │   └── 📁 vpc/
│   │   │       ├── main.tf
│   │   │       ├── outputs.tf
│   │   │       └── variables.tf
│   │   │
│   │   ├── 📁 scripts/
│   │   │   ├── deploy.sh
│   │   │   └── setup-backend.sh
│   │   │
│   │   ├── main.tf
│   │   ├── outputs.tf
│   │   ├── README.md
│   │   └── variables.tf
│   │
│   └── README.md
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_network_analysis.ipynb
│   ├── 05_game_theory_analysis.ipynb
│   ├── 06_simulation_experiments.ipynb
│   ├── 07_results_visualization.ipynb
│   ├── NOTEBOOK_FILE_OUTPUTS.md
│   ├── run_analysis.py
│   └── run_training.py
│
├── 📁 reports/
│   ├── 📁 figures/
│   │   ├── intervention_strategies.png
│   │   ├── sensitivity_analysis.png
│   │   ├── simulation_results.png
│   │   ├── strategy_comparison.png
│   │   └── temporal_dynamics.png
│   │
│   ├── simulation_experiments_summary.json
│   └── simulation_summary.txt
│
├── 📁 results/
│   ├── 📁 enhanced_network_analysis/
│   │   ├── 📁 figures/
│   │   │   ├── enhanced_game_theory_analysis.png
│   │   │   └── enhanced_propagation_analysis.png
│   │   │
│   │   ├── enhanced_analysis_report.txt
│   │   ├── enhanced_game_theory_results.csv
│   │   ├── enhanced_propagation_results.csv
│   │   ├── enhanced_results_complete.json
│   │   └── statistical_analysis.json
│   │
│   └── 📁 network_analysis/
│       ├── 📁 figures/
│       │   ├── centrality_comparison.png
│       │   ├── degree_distribution.png
│       │   ├── game_theory_analysis.png
│       │   ├── network_2d.png
│       │   ├── payoff_matrix.png
│       │   ├── propagation_analysis.png
│       │   └── test_propagation.png
│       │
│       ├── comprehensive_analysis_report.txt
│       ├── comprehensive_results.pkl
│       ├── game_theory_results.csv
│       ├── network_properties.csv
│       └── propagation_results.csv
│
├── 📁 scripts/
│   ├── backup-data.sh
│   ├── deploy.sh
│   ├── install-dependencies.sh
│   ├── run-dev.sh
│   ├── run-prod.sh
│   ├── setup.sh
│   └── test-all.sh
│
├── .conda-env
├── .dockerignore
├── .editorconfig
├── .gitattributes
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── GTDS PROJECT DOCUMENTATION.odt
├── LICENSE
├── Makefile
├── README.md
├── SECURITY.md
└── WORKFLOW.md
```

---

## 📊 File Type Breakdown

### Backend (Python)
- **Core Application:** 38 files
- **Game Theory:** 7 files
- **ML Models:** 12 files
- **Network Analysis:** 5 files
- **Tests:** 16 files
- **Scripts:** 4 files

### Frontend (TypeScript/React)
- **Pages & Routes:** 25 files
- **Components:** 60 files
- **Hooks:** 8 files
- **Store:** 6 files
- **Types:** 7 files
- **Utilities:** 8 files
- **Tests:** 3 files

### Data
- **Raw Datasets:** 9 files
- **Processed Data:** 20+ files
- **Results:** 25+ files

### Infrastructure
- **Kubernetes:** 21 files
- **Terraform:** 16 files
- **Monitoring:** 8 files

### Notebooks
- **Jupyter Notebooks:** 7 notebooks
- **Helper Scripts:** 2 files

### Documentation
- **Methodology Docs:** 3 files
- **API Docs:** 2 files
- **Tutorials:** 3 files
- **Other Docs:** 7 files

### Configuration
- **Docker:** 4 files
- **CI/CD:** 4 workflows
- **Environment:** 6 files
- **Build Tools:** 15 files

---

## 🔑 Key File Locations

### Entry Points
- **Backend API:** `/backend/app/main.py`
- **Frontend App:** `/frontend/src/app/page.tsx`
- **Backend Tests:** `/backend/tests/conftest.py`
- **Frontend Tests:** `/frontend/tests/setup.ts`

### Core Logic
- **Game Theory Engine:** `/backend/game_theory/`
- **ML Classifiers:** `/backend/ml_models/classifiers/`
- **Network Analysis:** `/backend/network/`
- **Business Services:** `/backend/app/services/`

### State Management
- **Zustand Stores:** `/frontend/src/store/`
- **Custom Hooks:** `/frontend/src/hooks/`

### API Endpoints
- **Backend APIs:** `/backend/app/api/v1/`
- **Frontend APIs:** `/frontend/src/app/api/`

### Data Pipeline
- **Raw Data:** `/data/raw/`
- **Processed Data:** `/data/processed/`
- **Results:** `/data/results/` and `/results/`
- **Models:** `/data/models/`

---

## 📝 Notes

- **Node Modules:** Excluded from count (~2000+ packages)
- **Python Cache:** Excluded (__pycache__, .pyc files)
- **Build Artifacts:** Excluded (.next/, dist/, build/)
- **Git Directory:** Excluded (.git/)

**Last Updated:** 2025-09-30
