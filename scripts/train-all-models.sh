#!/bin/bash

# Complete Model Training Script
# Trains ALL models properly with real data

set -e

cd "$(dirname "$0")/.."

echo "=========================================================================="
echo "COMPLETE MODEL TRAINING - ALL MODELS WITH REAL DATA"
echo "=========================================================================="
echo "Hardware: RTX 4050 (6GB VRAM), 16GB RAM, Ryzen 7 7734HS"
echo "Models: Traditional ML + LSTM + DistilBERT + Ensemble"
echo ""
echo "This will train:"
echo "  1. Logistic Regression"
echo "  2. Random Forest"
echo "  3. Gradient Boosting"
echo "  4. Naive Bayes"
echo "  5. LSTM (PyTorch)"
echo "  6. DistilBERT (Lightweight BERT for 6GB VRAM)"
echo "  7. Ensemble (Voting)"
echo ""
echo "Estimated time: 30-45 minutes"
echo "=========================================================================="
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

# Activate conda environment
echo "→ Activating conda environment..."
source /home/ghost/anaconda3/bin/activate fake_news

# Run training
echo ""
echo "→ Starting complete training pipeline..."
echo ""

python notebooks/complete_training_pipeline.py

echo ""
echo "=========================================================================="
echo "✅ TRAINING COMPLETE!"
echo "=========================================================================="
echo ""
echo "Models saved to: backend/models/"
echo "Results saved to: data/results/final_evaluation.csv"
echo ""
echo "Next steps:"
echo "  1. Restart backend server to load new models"
echo "  2. Test classifier page at http://localhost:3000/classifier"
echo "  3. All models should now work with REAL predictions!"
echo ""
echo "=========================================================================="
