# Notebook File Output Locations

This document summarizes where each notebook saves its output files in the `/home/ghost/fake-news-game-theory/notebooks` directory.

## 01_data_exploration.ipynb

**Description:** Exploratory Data Analysis of fake news datasets (FakeNewsNet, Kaggle, LIAR)

**Output Files:** None - This notebook only performs analysis and displays visualizations inline. No files are saved to disk.

**Key Activities:**
- Loads and analyzes three datasets: FakeNewsNet, Kaggle Fake News, and LIAR
- Displays statistics and visualizations
- Performs cross-dataset comparison
- Analyzes game theory player behavior potential

---

## 02_feature_engineering.ipynb

**Description:** Feature extraction and data preprocessing for model training

**Output Files Saved:**
- `/home/ghost/fake-news-game-theory/data/processed/train/X_train.csv` - Training features
- `/home/ghost/fake-news-game-theory/data/processed/train/y_train.csv` - Training labels
- `/home/ghost/fake-news-game-theory/data/processed/validation/X_val.csv` - Validation features
- `/home/ghost/fake-news-game-theory/data/processed/validation/y_val.csv` - Validation labels
- `/home/ghost/fake-news-game-theory/data/processed/test/X_test.csv` - Test features
- `/home/ghost/fake-news-game-theory/data/processed/test/y_test.csv` - Test labels
- `/home/ghost/fake-news-game-theory/data/processed/features/all_features.csv` - Complete feature dataset
- `/home/ghost/fake-news-game-theory/data/processed/features/scaler.pkl` - StandardScaler object
- `/home/ghost/fake-news-game-theory/data/processed/features/feature_names.pkl` - List of feature names
- `/home/ghost/fake-news-game-theory/data/processed/features/tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `/home/ghost/fake-news-game-theory/data/processed/features/feature_correlations.csv` - Feature correlation data
- `/home/ghost/fake-news-game-theory/data/processed/features/feature_summary.json` - Summary statistics
- `/home/ghost/fake-news-game-theory/data/processed/features/selected_features.csv` - Top selected features
- `/home/ghost/fake-news-game-theory/data/processed/features/selected_feature_names.pkl` - Selected feature names
- `/home/ghost/fake-news-game-theory/data/processed/features/feature_correlations.png` - Correlation visualization
- `/home/ghost/fake-news-game-theory/data/processed/features/label_distribution.png` - Label distribution plot
- `/home/ghost/fake-news-game-theory/data/processed/features/domain_distribution.png` - Domain distribution plot
- `/home/ghost/fake-news-game-theory/data/processed/features/word_clouds.png` - Word clouds visualization
- `/home/ghost/fake-news-game-theory/data/processed/features/feature_distributions.png` - Feature distributions
- `/home/ghost/fake-news-game-theory/data/processed/features/correlation_heatmap.png` - Correlation heatmap
- `/home/ghost/fake-news-game-theory/data/processed/pipeline_status.json` - Pipeline status report

**Key Features:**
- Extracts linguistic features (word count, sentiment, readability)
- Creates TF-IDF features
- Generates train/validation/test splits
- Produces visualization plots

---

## 03_model_training.ipynb

**Description:** Machine learning model training and evaluation

**Output Files Saved:**
- `/home/ghost/fake-news-game-theory/data/models/[model_name]_[timestamp]/model.pkl` - Trained ML models
- `/home/ghost/fake-news-game-theory/data/models/[model_name]_[timestamp]/model.pth` - PyTorch models
- `/home/ghost/fake-news-game-theory/data/models/[model_name]_[timestamp]/metrics.json` - Model metrics
- `/home/ghost/fake-news-game-theory/data/models/[model_name]_[timestamp]/metadata.json` - Model metadata
- `/home/ghost/fake-news-game-theory/data/models/best_[model_name]_[timestamp]/` - Best performing model
- `/home/ghost/fake-news-game-theory/data/results/model_comparison.png` - Model comparison visualization
- `/home/ghost/fake-news-game-theory/data/results/confusion_matrices.png` - Confusion matrix plots
- `/home/ghost/fake-news-game-theory/data/results/model_comparison.csv` - Comparison results CSV

**Models Trained:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Naive Bayes
- Ensemble (Voting Classifier)
- Deep Neural Network (PyTorch)
- BERT (optional, resource-intensive)

---

## 04_network_analysis.ipynb

**Description:** Enhanced network analysis with game theory and propagation models

**Output Files Saved:**
- `/home/ghost/fake-news-game-theory/data/results/enhanced_propagation_results.csv` - Propagation simulation data
- `/home/ghost/fake-news-game-theory/data/results/enhanced_game_theory_results.csv` - Game theory equilibria data
- `/home/ghost/fake-news-game-theory/data/results/statistical_analysis.json` - Statistical analysis results
- `/home/ghost/fake-news-game-theory/data/results/enhanced_analysis_report.txt` - Comprehensive text report
- `/home/ghost/fake-news-game-theory/data/results/enhanced_results_complete.json` - Complete results JSON
- `/home/ghost/fake-news-game-theory/data/results/figures/enhanced_propagation_analysis.png` - Propagation visualizations
- `/home/ghost/fake-news-game-theory/data/results/figures/enhanced_game_theory_analysis.png` - Game theory plots

**Key Analyses:**
- Real-world network simulations (Facebook, Twitter)
- Heterogeneous agent modeling
- Temporal propagation dynamics
- Nash equilibrium calculation
- Intervention strategy testing
- Bootstrap confidence intervals

---

## 05_game_theory_analysis.ipynb

**Description:** Game theory modeling and Nash equilibrium analysis

**Output Files:** Not available in the scan (file too large to read completely)

**Expected Outputs:**
- Game theory simulation results
- Nash equilibrium calculations
- Strategic analysis data

---

## 06_simulation_experiments.ipynb

**Description:** Monte Carlo simulation experiments and sensitivity analysis

**Output Files Saved:**
- `../data/processed/simulation_results.csv` - Complete simulation results
- `../reports/simulation_summary.txt` - Text summary report
- `../reports/simulation_experiments_summary.json` - JSON summary with metadata
- `../reports/figures/simulation_results.png` - Main results visualization
- `../reports/figures/intervention_strategies.png` - Intervention comparison plot
- `../reports/figures/temporal_dynamics.png` - Temporal spread dynamics
- `../reports/figures/strategy_comparison.png` - Strategy effectiveness comparison
- `../reports/figures/sensitivity_analysis.png` - Parameter sensitivity plots
- `../models/simulation_checkpoint.pkl` - Simulation checkpoint (mentioned but path unclear)

**Key Experiments:**
- Network topology comparison (Barabási-Albert, Watts-Strogatz, Erdős-Rényi)
- Fake vs. real news propagation
- Intervention strategy effectiveness
- Mixed strategy Nash equilibria
- Temporal dynamics analysis
- Sensitivity analysis

---

## 07_results_visualization.ipynb

**Description:** Results visualization and reporting (EXCLUDED per request)

**Note:** This notebook is excluded from the summary as it focuses on visualization only.

---

## Summary Statistics

- **Total Notebooks Analyzed:** 6 (excluding results_visualization)
- **Primary Output Directory:** `/home/ghost/fake-news-game-theory/data/`
- **Key Subdirectories:**
  - `data/processed/` - Preprocessed datasets and features
  - `data/models/` - Trained machine learning models
  - `data/results/` - Analysis results and visualizations
  - `data/results/figures/` - Generated plots and figures
  - `reports/` - Summary reports (used by notebook 06)

## File Naming Conventions

- CSV files: Tabular data results
- PKL files: Python pickle objects (models, scalers, vectorizers)
- JSON files: Structured metadata and summaries
- PNG files: Visualization plots (300 DPI)
- TXT files: Human-readable reports

## Data Flow

1. **01_data_exploration.ipynb** → Analyzes raw data (no output files)
2. **02_feature_engineering.ipynb** → Creates processed datasets → `data/processed/`
3. **03_model_training.ipynb** → Trains models using processed data → `data/models/` and `data/results/`
4. **04_network_analysis.ipynb** → Network simulations → `data/results/` and `data/results/figures/`
5. **06_simulation_experiments.ipynb** → Monte Carlo experiments → `reports/` and `reports/figures/`

---

*Generated: 2025-09-30*
*Project: Fake News Detection with Game Theory*