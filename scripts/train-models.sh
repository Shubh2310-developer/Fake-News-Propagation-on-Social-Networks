#!/bin/bash

# Model Training Script - Optimized for Performance
# Usage: ./scripts/train-models.sh [fast|deep|optimize|full|single]

set -e

cd "$(dirname "$0")/.."

# Activate conda environment
source /home/ghost/anaconda3/bin/activate fake_news

MODE="${1:-fast}"

echo "=========================================="
echo "Fake News Detection Model Training"
echo "=========================================="
echo "Mode: $MODE"
echo ""

case $MODE in
  fast)
    echo "Running FAST training (baseline models only)..."
    python notebooks/model_training_optimized.py --mode fast
    ;;

  deep)
    echo "Running training with Deep Learning..."
    python notebooks/model_training_optimized.py --mode deep
    ;;

  optimize)
    echo "Running training with Hyperparameter Optimization..."
    python notebooks/model_training_optimized.py --mode optimize
    ;;

  full)
    echo "Running FULL training (all features enabled)..."
    python notebooks/model_training_optimized.py --mode full
    ;;

  single)
    MODEL_TYPE="${2:-random_forest}"
    echo "Training single model: $MODEL_TYPE..."
    python notebooks/model_training_optimized.py --mode single --model-type "$MODEL_TYPE"
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo ""
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Available modes:"
    echo "  fast     - Fast training, baseline models only (~15-30 seconds)"
    echo "  deep     - With deep neural network (~5-7 minutes)"
    echo "  optimize - With hyperparameter optimization (~8-10 minutes)"
    echo "  full     - All features enabled (~12-15 minutes)"
    echo "  single   - Train single model, specify type as 2nd arg (~30 seconds)"
    echo ""
    echo "Examples:"
    echo "  $0 fast"
    echo "  $0 single random_forest"
    echo "  $0 single gradient_boosting"
    exit 1
    ;;
esac

echo ""
echo "=========================================="
echo "Models saved to: data/models/"
echo "Results saved to: data/results/"
echo "=========================================="
