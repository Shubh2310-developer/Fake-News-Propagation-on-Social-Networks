# Model Training Tutorial

This advanced tutorial is for data scientists who want to retrain the fake news classification models with their own data or experiment with different architectures and hyperparameters.

## Prerequisites

### Technical Requirements
- Python 3.10+ with conda environment
- GPU support recommended (CUDA 11.8+)
- Minimum 16GB RAM (32GB recommended for large models)
- At least 20GB free disk space

### Knowledge Requirements
- Basic understanding of machine learning concepts
- Familiarity with transformer models (BERT, RoBERTa)
- Experience with Python and command-line tools

## Training Data Requirements

### Data Format Specification

Your training data must be in CSV format with the following structure:

```csv
text,label,source,domain,date
"Scientists publish new study on climate change effects",0,reuters,science,2024-01-15
"Miracle cure discovered! Doctors hate this simple trick!",1,unknown,health,2024-01-14
"Local election results show increased voter turnout",0,ap_news,politics,2024-01-13
"Secret government program revealed by anonymous whistleblower",1,conspiracy_blog,politics,2024-01-12
```

#### Required Columns
- **text** (string): The article text or claim to classify
- **label** (integer): 0 = real/true, 1 = fake/false

#### Optional Columns (Recommended)
- **source** (string): Publication or source identifier
- **domain** (string): Topic category (politics, health, science, etc.)
- **date** (string): Publication date (YYYY-MM-DD format)
- **url** (string): Original article URL
- **author** (string): Article author

### Data Quality Guidelines

#### Text Quality
- **Minimum length**: 50 characters
- **Maximum length**: 10,000 characters (will be truncated)
- **Language**: Primarily English (multi-language support planned)
- **Encoding**: UTF-8

#### Label Quality
- **Binary classification**: Use 0/1 labels only
- **Ground truth**: Labels should be verified by fact-checkers
- **Consistency**: Avoid contradictory labels for similar content
- **Balance**: Aim for roughly 50-50 class distribution

#### Dataset Size Recommendations
- **Minimum**: 1,000 samples (for fine-tuning)
- **Recommended**: 10,000+ samples (for robust training)
- **Optimal**: 50,000+ samples (for best performance)

### Example Dataset Preparation

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and validate your data
df = pd.read_csv('your_dataset.csv')

# Data validation
print(f"Dataset shape: {df.shape}")
print(f"Label distribution: {df['label'].value_counts()}")
print(f"Missing values: {df.isnull().sum()}")

# Clean text data
df['text'] = df['text'].str.strip()
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)

# Filter by length
df = df[df['text'].str.len() >= 50]
df = df[df['text'].str.len() <= 10000]

# Remove duplicates
df = df.drop_duplicates(subset=['text'])

# Balance classes if needed
min_class_size = df['label'].value_counts().min()
df_balanced = df.groupby('label').sample(n=min_class_size, random_state=42)

# Save cleaned dataset
df_balanced.to_csv('cleaned_dataset.csv', index=False)
```

## Basic Model Training

### Using the Training Script

#### 1. BERT Model Training

```bash
# Navigate to backend directory
cd backend

# Activate environment
conda activate fake_news

# Train BERT model
python scripts/train_models.py \
  --model-type bert \
  --dataset-path data/your_dataset.csv \
  --experiment-name "bert_custom_v1" \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5
```

**Expected Output**:
```
2024-01-01 10:00:00,123 - __main__ - INFO - Starting model training...
2024-01-01 10:00:00,124 - __main__ - INFO - Loading dataset: data/your_dataset.csv
2024-01-01 10:00:01,234 - __main__ - INFO - Dataset loaded: 10000 samples
2024-01-01 10:00:01,235 - __main__ - INFO - Train/test split: 8000/2000
2024-01-01 10:00:02,345 - __main__ - INFO - Initializing BERT model...

Epoch 1/3:
Training: 100%|ˆˆˆˆˆˆˆˆˆˆˆˆ| 500/500 [15:23<00:00, 1.84s/it]
Validation: 100%|ˆˆˆˆˆˆˆˆˆˆˆˆ| 125/125 [02:14<00:00, 1.07s/it]
Train Loss: 0.234, Val Loss: 0.189, Val Accuracy: 0.923

Epoch 2/3:
Training: 100%|ˆˆˆˆˆˆˆˆˆˆˆˆ| 500/500 [15:18<00:00, 1.83s/it]
Validation: 100%|ˆˆˆˆˆˆˆˆˆˆˆˆ| 125/125 [02:12<00:00, 1.06s/it]
Train Loss: 0.156, Val Loss: 0.142, Val Accuracy: 0.941

Epoch 3/3:
Training: 100%|ˆˆˆˆˆˆˆˆˆˆˆˆ| 500/500 [15:21<00:00, 1.84s/it]
Validation: 100%|ˆˆˆˆˆˆˆˆˆˆˆˆ| 125/125 [02:13<00:00, 1.06s/it]
Train Loss: 0.098, Val Loss: 0.127, Val Accuracy: 0.952

Training completed!
Final test accuracy: 0.948
Model saved to: models/experiments/bert_custom_v1/
```

#### 2. Ensemble Model Training

```bash
# Train ensemble model (faster, good baseline)
python scripts/train_models.py \
  --model-type ensemble \
  --dataset-path data/your_dataset.csv \
  --experiment-name "ensemble_baseline" \
  --max-features 10000 \
  --n-estimators 100
```

### Understanding Training Parameters

#### BERT Parameters
- **epochs**: Number of training iterations (3-5 recommended)
- **batch-size**: Samples per gradient update (16 for 16GB GPU, 8 for 8GB GPU)
- **learning-rate**: Step size for optimization (2e-5 is standard)
- **bert-model**: Base model to fine-tune (bert-base-uncased, roberta-base)

#### Ensemble Parameters
- **max-features**: TF-IDF vocabulary size (10000 is good default)
- **n-estimators**: Number of trees in Random Forest (100-500)

### Monitoring Training Progress

#### Using MLflow Dashboard

```bash
# Start MLflow server (in a separate terminal)
mlflow ui --port 5000

# Open browser to http://localhost:5000
# View experiment tracking, metrics, and model comparisons
```

#### Training Logs

```bash
# Monitor training logs
tail -f logs/training.log

# Check GPU usage
nvidia-smi -l 1

# Monitor system resources
htop
```

## Advanced Training Scenarios

### Custom Hyperparameter Tuning

#### Grid Search for BERT

```python
# Create hyperparameter grid
hyperparams = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'batch_size': [8, 16, 32],
    'epochs': [3, 4, 5]
}

# Run grid search
for lr in hyperparams['learning_rate']:
    for bs in hyperparams['batch_size']:
        for ep in hyperparams['epochs']:
            experiment_name = f"bert_lr{lr}_bs{bs}_ep{ep}"

            # Run training with these parameters
            # (Use the training script with different arguments)
```

#### Automated Hyperparameter Search

```bash
# Install optimization library
pip install optuna

# Run automated hyperparameter optimization
python scripts/hyperparameter_optimization.py \
  --dataset-path data/your_dataset.csv \
  --n-trials 50 \
  --study-name "bert_optimization"
```

### Domain-Specific Fine-Tuning

#### Health Misinformation Model

```bash
# Train on health-specific dataset
python scripts/train_models.py \
  --model-type bert \
  --dataset-path data/health_misinformation.csv \
  --experiment-name "bert_health_v1" \
  --bert-model "allenai/scibert_scivocab_uncased"  # Science-specific BERT
```

#### Political Misinformation Model

```bash
# Train on political dataset
python scripts/train_models.py \
  --model-type bert \
  --dataset-path data/political_misinformation.csv \
  --experiment-name "bert_politics_v1" \
  --epochs 4 \
  --learning-rate 1e-5  # Lower learning rate for stability
```

### Multi-Language Training

```bash
# Train multilingual model
python scripts/train_models.py \
  --model-type bert \
  --dataset-path data/multilingual_dataset.csv \
  --experiment-name "bert_multilingual_v1" \
  --bert-model "bert-base-multilingual-cased"
```

## Model Evaluation and Comparison

### Comprehensive Evaluation Script

```python
#!/usr/bin/env python3
# evaluate_models.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_data_path):
    # Load model and test data
    # (Implementation depends on model type)

    # Generate predictions
    y_true, y_pred, y_prob = get_predictions(model_path, test_data_path)

    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_path}/confusion_matrix.png')

    return report

# Evaluate multiple models
models_to_evaluate = [
    'models/experiments/bert_custom_v1',
    'models/experiments/ensemble_baseline',
    'models/experiments/bert_health_v1'
]

results = {}
for model_path in models_to_evaluate:
    results[model_path] = evaluate_model(model_path, 'data/test_set.csv')

# Compare results
comparison_df = pd.DataFrame({
    model: {
        'accuracy': metrics['accuracy'],
        'precision': metrics['macro avg']['precision'],
        'recall': metrics['macro avg']['recall'],
        'f1_score': metrics['macro avg']['f1-score']
    }
    for model, metrics in results.items()
}).T

print(comparison_df)
comparison_df.to_csv('model_comparison.csv')
```

### A/B Testing Framework

```python
# ab_test_models.py

def ab_test_models(model_a_path, model_b_path, test_data_path, n_bootstrap=1000):
    """Statistical comparison of two models"""

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Get predictions from both models
    pred_a = predict_with_model(model_a_path, test_df['text'])
    pred_b = predict_with_model(model_b_path, test_df['text'])

    # Calculate accuracy for each model
    acc_a = accuracy_score(test_df['label'], pred_a)
    acc_b = accuracy_score(test_df['label'], pred_b)

    # Bootstrap confidence intervals
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample test set
        sample_indices = np.random.choice(len(test_df), len(test_df), replace=True)

        sample_acc_a = accuracy_score(test_df['label'].iloc[sample_indices],
                                     pred_a[sample_indices])
        sample_acc_b = accuracy_score(test_df['label'].iloc[sample_indices],
                                     pred_b[sample_indices])

        bootstrap_diffs.append(sample_acc_b - sample_acc_a)

    # Statistical significance test
    p_value = np.mean(np.array(bootstrap_diffs) <= 0)

    print(f"Model A Accuracy: {acc_a:.4f}")
    print(f"Model B Accuracy: {acc_b:.4f}")
    print(f"Difference: {acc_b - acc_a:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Model B is {'significantly' if p_value < 0.05 else 'not significantly'} better")

    return {
        'acc_a': acc_a,
        'acc_b': acc_b,
        'difference': acc_b - acc_a,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

## Production Deployment

### Model Validation Pipeline

```bash
# Validate model before deployment
python scripts/validate_model.py \
  --model-path models/experiments/bert_custom_v1 \
  --test-data data/holdout_test.csv \
  --min-accuracy 0.90 \
  --min-precision 0.88 \
  --min-recall 0.87
```

### Model Registration

```python
# register_model.py

import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(experiment_name, metric='accuracy', threshold=0.92):
    client = MlflowClient()

    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Find best run
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    best_run = runs.iloc[0]

    if best_run[f'metrics.{metric}'] >= threshold:
        # Register model
        model_uri = f"runs:/{best_run.run_id}/model"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name="fake-news-classifier"
        )

        # Promote to production
        client.transition_model_version_stage(
            name="fake-news-classifier",
            version=model_version.version,
            stage="Production"
        )

        print(f"Model registered and promoted to production: v{model_version.version}")
    else:
        print(f"Model performance ({best_run[f'metrics.{metric}']:.3f}) below threshold ({threshold})")

# Usage
register_best_model("bert_custom_experiments", metric='f1_score', threshold=0.92)
```

### API Integration

```python
# Update API to use new model
# In backend/api/classifier.py

def load_production_model():
    client = MlflowClient()

    # Get latest production model
    model_version = client.get_latest_versions(
        name="fake-news-classifier",
        stages=["Production"]
    )[0]

    model_uri = f"models:/{model_version.name}/{model_version.version}"
    model = mlflow.pytorch.load_model(model_uri)

    return model, model_version.version

# Load model at startup
PRODUCTION_MODEL, MODEL_VERSION = load_production_model()
```

## Troubleshooting Training Issues

### Common Problems and Solutions

#### 1. CUDA Out of Memory

```bash
# Error: RuntimeError: CUDA out of memory

# Solutions:
# Reduce batch size
python scripts/train_models.py --batch-size 8  # Instead of 16

# Use gradient accumulation
python scripts/train_models.py --gradient-accumulation-steps 2

# Use smaller model
python scripts/train_models.py --bert-model "distilbert-base-uncased"
```

#### 2. Training Divergence

```bash
# Symptoms: Loss increases instead of decreasing

# Solutions:
# Lower learning rate
python scripts/train_models.py --learning-rate 1e-5

# Add gradient clipping
python scripts/train_models.py --max-grad-norm 1.0

# Use learning rate scheduler
python scripts/train_models.py --lr-scheduler "linear"
```

#### 3. Overfitting

```bash
# Symptoms: Training accuracy much higher than validation accuracy

# Solutions:
# Add dropout
python scripts/train_models.py --dropout 0.3

# Use weight decay
python scripts/train_models.py --weight-decay 0.01

# Early stopping
python scripts/train_models.py --early-stopping-patience 3
```

#### 4. Slow Training

```bash
# Optimize for speed:
# Use mixed precision
python scripts/train_models.py --fp16

# Increase batch size (if memory allows)
python scripts/train_models.py --batch-size 32

# Use DataLoader optimization
python scripts/train_models.py --dataloader-num-workers 4
```

### Debugging Training Scripts

```bash
# Enable debug logging
export PYTHONPATH=./backend
python -m pdb scripts/train_models.py --model-type bert --dataset-path data/small_dataset.csv

# Profile memory usage
python -m memory_profiler scripts/train_models.py

# Check data loading
python -c "
from scripts.data_utils import load_dataset
df = load_dataset('data/your_dataset.csv')
print(df.head())
print(df.info())
"
```

## Best Practices Summary

### Data Preparation
1. **Clean and validate** your dataset thoroughly
2. **Balance classes** or use appropriate sampling strategies
3. **Split data** chronologically for temporal validity
4. **Hold out test set** that models never see during development

### Training Strategy
1. **Start with baselines** (ensemble models) before complex models
2. **Use cross-validation** for robust performance estimates
3. **Monitor both training and validation** metrics
4. **Save model checkpoints** regularly

### Experimentation
1. **Track all experiments** with MLflow
2. **Use systematic hyperparameter search**
3. **Compare models statistically** before choosing
4. **Document model assumptions** and limitations

### Production Readiness
1. **Validate on hold-out data** before deployment
2. **Set up monitoring** for model performance drift
3. **Plan for model updates** and versioning
4. **Test API integration** thoroughly

You now have the complete toolkit for training, evaluating, and deploying custom fake news detection models. These models will integrate seamlessly with the game theory simulation platform to provide realistic misinformation detection capabilities.