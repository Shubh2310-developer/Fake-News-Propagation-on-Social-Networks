#!/usr/bin/env python
"""Adjust ensemble threshold to reduce false positives."""
import sys
sys.path.insert(0, '/home/ghost/fake-news-game-theory/backend')
import joblib
from pathlib import Path

models_dir = Path('/home/ghost/fake-news-game-theory/backend/models')

print("Loading ensemble...")
ensemble = joblib.load(models_dir / 'ensemble_config.joblib')

print(f"Current threshold: {ensemble.threshold}")
print("Adjusting threshold to 0.60 (require 60% confidence for 'fake')")

ensemble.threshold = 0.60

joblib.dump(ensemble, models_dir / 'ensemble_config.joblib')

print("✓ Ensemble threshold updated!")
print("  This will make the model MORE conservative about predicting 'fake'")
print("  Reduces false positives (real news marked as fake)")
