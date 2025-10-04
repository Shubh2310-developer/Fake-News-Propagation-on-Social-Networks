# Model Training Guide - Optimized for Performance

## Problem with Original Notebook
The original `03_model_training.ipynb` freezes due to:
1. **SVM training** - Very slow on large datasets (58+ seconds)
2. **Hyperparameter optimization** - 20 iterations × 3 CV folds = 60 model trainings
3. **BERT training** - Extremely resource-intensive, requires GPU
4. **Deep neural network** - 50 epochs by default
5. **No resource limits** - Can overwhelm system memory

## Solution: Use Optimized Script

### Option 1: Fast Training (Recommended) - ~2-3 minutes
```python
from notebooks.03_model_training_optimized import quick_train

# Train only baseline models (fastest)
results = quick_train(optimize=False, deep_nn=False)
```

### Option 2: With Deep Learning - ~5-7 minutes
```python
# Add neural network training
results = quick_train(optimize=False, deep_nn=True)
```

### Option 3: With Optimization - ~8-10 minutes
```python
# Add hyperparameter optimization (lightweight)
results = quick_train(optimize=True, deep_nn=False)
```

### Option 4: Full Training - ~12-15 minutes
```python
# Everything enabled
results = quick_train(optimize=True, deep_nn=True)
```

### Option 5: Single Model (Fastest) - ~30 seconds
```python
from notebooks.03_model_training_optimized import train_single_model

model, metrics, path = train_single_model('random_forest')
```

## What Was Optimized

### 1. Removed SVM
- **Before**: 58 seconds per training
- **After**: Excluded (use Random Forest or Gradient Boosting instead)

### 2. Reduced Hyperparameter Search
- **Before**: 20 iterations, 3-fold CV
- **After**: 5 iterations, 2-fold CV, only optimizes best model

### 3. Smaller Neural Network
- **Before**: [512, 256, 128] layers, 50 epochs
- **After**: [256, 128] layers, 15 epochs

### 4. Disabled BERT by Default
- **Before**: Included in pipeline
- **After**: Completely removed (too resource-intensive)

### 5. Added Progress Indicators
- Clear progress bars and time estimates
- Better error handling to prevent crashes

## Performance Comparison

| Method | Time | Models Trained | Recommended For |
|--------|------|----------------|-----------------|
| Single Model | 30s | 1 | Quick testing |
| Fast Training | 2-3 min | 5 (baseline + ensemble) | Development |
| With Deep Learning | 5-7 min | 6 | Better accuracy |
| With Optimization | 8-10 min | 7 | Production-ready |
| Full Training | 12-15 min | 8 | Best performance |

## Running from Command Line

```bash
cd /home/ghost/fake-news-game-theory
python -m notebooks.03_model_training_optimized
```

## Running in Jupyter

Create a new notebook cell:

```python
# Fast training
%run notebooks/03_model_training_optimized.py
```

Or import functions:

```python
import sys
sys.path.append('/home/ghost/fake-news-game-theory')
from notebooks.03_model_training_optimized import quick_train, train_single_model

# Your choice of training
results = quick_train(optimize=False, deep_nn=False)
```

## Results Access

```python
# Access trained models
models = results['models']
best_model = models['ensemble']  # or 'random_forest', 'gradient_boosting', etc.

# Access metrics
test_results = results['test_results']
comparison_df = results['comparison']

# Model is auto-saved
model_path = results['best_model_path']
```

## Troubleshooting

### Still Freezing?
1. Use single model training only: `train_single_model('random_forest')`
2. Check available RAM: `free -h`
3. Close other applications
4. Reduce dataset size in data preprocessing

### Out of Memory?
- Use `train_single_model()` instead of full pipeline
- Reduce batch size in deep learning (default: 128)
- Disable deep learning: `quick_train(deep_nn=False)`

### Too Slow?
- Disable optimization: `quick_train(optimize=False)`
- Use single model: `train_single_model('logistic_regression')`
- Check CPU cores: `n_jobs=-1` uses all cores

## Next Steps

After training:
1. Models saved to: `/home/ghost/fake-news-game-theory/data/models/`
2. Results saved to: `/home/ghost/fake-news-game-theory/data/results/`
3. Use best model for predictions in your application
