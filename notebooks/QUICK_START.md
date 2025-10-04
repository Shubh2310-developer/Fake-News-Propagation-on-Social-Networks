# Quick Start - Model Training

## ⚠️ IMPORTANT: Use Optimized Script

**DO NOT RUN** `03_model_training.ipynb` - it will freeze VSCode!

**USE** `03_model_training_optimized.py` instead.

## Fastest Way to Train Models

### Option 1: Command Line (Recommended)

```bash
# Fast training (15-30 seconds)
./scripts/train-models.sh fast

# Single model (30 seconds)
./scripts/train-models.sh single random_forest

# With deep learning (5-7 minutes)
./scripts/train-models.sh deep

# Full training (12-15 minutes)
./scripts/train-models.sh full
```

### Option 2: Python Direct

```bash
source /home/ghost/anaconda3/bin/activate fake_news
cd /home/ghost/fake-news-game-theory
python notebooks/03_model_training_optimized.py
```

### Option 3: In Python/Jupyter

```python
import sys
sys.path.append('/home/ghost/fake-news-game-theory')

# Fast training
from notebooks.03_model_training_optimized import quick_train
results = quick_train()

# Or single model
from notebooks.03_model_training_optimized import train_single_model
model, metrics, path = train_single_model('random_forest')
```

## What Gets Trained?

### Fast Mode (Default)
- Logistic Regression
- Random Forest
- Gradient Boosting
- Naive Bayes
- Voting Ensemble

**Time**: ~15-30 seconds

### With Deep Learning
All above + Neural Network (256→128 layers, 15 epochs)

**Time**: ~5-7 minutes

### With Optimization
All above + Hyperparameter tuning (lightweight, 5 iterations, 2-fold CV)

**Time**: ~8-10 minutes

### Full Mode
Everything enabled

**Time**: ~12-15 minutes

## Expected Performance

Based on test data (1000 samples):

| Model | F1 Score | Accuracy | AUC-ROC |
|-------|----------|----------|---------|
| Random Forest | 0.8693 | 0.8740 | 0.9303 |
| Gradient Boosting | 0.8591 | 0.8620 | 0.9259 |
| Ensemble | 0.8318 | 0.8320 | 0.9168 |
| Logistic Regression | 0.7708 | 0.7700 | 0.8425 |
| Naive Bayes | 0.7695 | 0.7680 | 0.8064 |

## Output Files

After training, check:

```
data/models/best_random_forest_TIMESTAMP.pkl  (or .pth for neural network)
data/results/comparison_TIMESTAMP.csv
```

## Troubleshooting

**VSCode still freezing?**
- Close the original `03_model_training.ipynb` notebook
- Use command line instead: `./scripts/train-models.sh fast`

**Out of memory?**
- Use single model: `./scripts/train-models.sh single random_forest`

**Want faster training?**
- Disable deep learning and optimization (default fast mode)

## Next Steps

1. Train models: `./scripts/train-models.sh fast`
2. Check results in `data/results/`
3. Use best model in your application
4. Integrate with backend API at `backend/app/services/classifier_service.py`
