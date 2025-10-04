"""
Model loader for the newly trained models.
Handles loading of all trained models including the SimpleEnsemble.
"""

import joblib
import pickle
import re
import numpy as np
import sys
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Define SimpleEnsemble class for loading
class SimpleEnsemble:
    """Lightweight ensemble with weighted voting."""

    def __init__(self, lr, nb, rf, gb):
        self.lr = lr
        self.nb = nb
        self.rf = rf
        self.gb = gb
        self.weights = [0.5, 1.0, 1.5, 2.0]

    def predict(self, X):
        pred_lr = self.lr.predict(X)
        pred_nb = self.nb.predict(X)
        pred_rf = self.rf.predict(X)
        pred_gb = self.gb.predict(X)

        weighted_sum = (
            pred_lr * self.weights[0] +
            pred_nb * self.weights[1] +
            pred_rf * self.weights[2] +
            pred_gb * self.weights[3]
        )

        threshold = sum(self.weights) / 2
        return (weighted_sum >= threshold).astype(int)

    def predict_proba(self, X):
        proba_lr = self.lr.predict_proba(X)
        proba_nb = self.nb.predict_proba(X)
        proba_rf = self.rf.predict_proba(X)
        proba_gb = self.gb.predict_proba(X)

        total_weight = sum(self.weights)
        proba = (
            proba_lr * self.weights[0] +
            proba_nb * self.weights[1] +
            proba_rf * self.weights[2] +
            proba_gb * self.weights[3]
        ) / total_weight

        return proba


class CalibratedEnsemble:
    """Ensemble with adjustable decision threshold to reduce false positives."""

    def __init__(self, lr, nb, rf, gb, threshold=0.523):
        self.lr = lr
        self.nb = nb
        self.rf = rf
        self.gb = gb
        self.weights = [1.0, 0.8, 1.2, 1.5]
        self.threshold = threshold

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        import numpy as np
        proba_lr = self.lr.predict_proba(X)
        proba_nb = self.nb.predict_proba(X)
        proba_rf = self.rf.predict_proba(X)
        proba_gb = self.gb.predict_proba(X)

        total_weight = sum(self.weights)
        proba = (
            proba_lr * self.weights[0] +
            proba_nb * self.weights[1] +
            proba_rf * self.weights[2] +
            proba_gb * self.weights[3]
        ) / total_weight

        return proba


# Register classes in __main__ module for pickle compatibility
sys.modules['__main__'].SimpleEnsemble = SimpleEnsemble
sys.modules['__main__'].CalibratedEnsemble = CalibratedEnsemble


class ModelLoader:
    """Load and manage trained models."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.preprocessing = None
        self.vectorizer = None
        self.scaler = None
        self.selector = None
        self.feature_names = []

    def load_preprocessing(self) -> bool:
        """Load preprocessing pipeline."""
        try:
            preprocessing_path = self.models_dir / 'preprocessing.pkl'
            if not preprocessing_path.exists():
                logger.error(f"Preprocessing file not found: {preprocessing_path}")
                return False

            with open(preprocessing_path, 'rb') as f:
                self.preprocessing = pickle.load(f)
                self.vectorizer = self.preprocessing.get('vectorizer')
                self.scaler = self.preprocessing.get('scaler')
                self.selector = self.preprocessing.get('selector')
                self.feature_names = self.preprocessing.get('feature_names', [])

            logger.info(f"Preprocessing loaded: {len(self.feature_names)} features")
            return True

        except Exception as e:
            logger.error(f"Error loading preprocessing: {e}")
            return False

    def load_model(self, model_name: str) -> bool:
        """Load a single model."""
        try:
            model_path = self.models_dir / f'{model_name}.joblib'
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return False

            model = joblib.load(model_path)
            self.models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            return False

    def load_all_models(self) -> Dict[str, bool]:
        """Load all available models."""
        results = {}

        # Load preprocessing first
        if not self.load_preprocessing():
            logger.error("Failed to load preprocessing, cannot load models")
            return results

        # Load individual models
        model_names = [
            'logistic_regression',
            'naive_bayes',
            'random_forest',
            'gradient_boosting'
        ]

        for model_name in model_names:
            results[model_name] = self.load_model(model_name)

        # Load ensemble (use ensemble_config.joblib)
        try:
            ensemble_path = self.models_dir / 'ensemble_config.joblib'
            if ensemble_path.exists():
                model = joblib.load(ensemble_path)
                self.models['ensemble'] = model
                logger.info("Loaded model: ensemble")
                results['ensemble'] = True
            else:
                logger.warning(f"Ensemble file not found: {ensemble_path}")
                results['ensemble'] = False
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            results['ensemble'] = False

        return results

    def clean_text(self, text: str) -> str:
        """Clean text for processing."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_basic_features(self, texts: List[str]) -> np.ndarray:
        """Extract enhanced numerical features (15 features to match training)."""
        features = []
        for text in texts:
            length = len(text)
            words = text.split()
            word_count = len(words)

            exclaim_count = text.count('!')
            question_count = text.count('?')
            quote_count = text.count('"') + text.count("'")

            avg_word_length = np.mean([len(w) for w in words]) if words else 0
            unique_word_ratio = len(set(words)) / word_count if word_count > 0 else 0
            upper_ratio = sum(1 for c in text if c.isupper()) / (length + 1)

            # Sentiment indicators
            sensational_words = ['shocking', 'breaking', 'unbelievable', 'amazing', 'incredible',
                                'urgent', 'alert', 'warning', 'danger', 'exclusive']
            formal_words = ['according', 'stated', 'reported', 'announced', 'confirmed',
                           'official', 'research', 'study', 'data', 'evidence']

            sensational_count = sum(1 for word in sensational_words if word in text.lower())
            formal_count = sum(1 for word in formal_words if word in text.lower())

            has_quotes = int(quote_count > 2)
            has_attribution = int(any(word in text.lower() for word in ['according to', 'said', 'stated']))

            features.append([
                length,
                word_count,
                exclaim_count,
                question_count,
                quote_count,
                avg_word_length,
                unique_word_ratio,
                upper_ratio,
                sensational_count,
                formal_count,
                has_quotes,
                has_attribution,
                exclaim_count / (word_count + 1),
                question_count / (word_count + 1),
                sensational_count / (word_count + 1),
            ])
        return np.array(features)

    def extract_features(self, texts: List[str]):
        """Extract all features for prediction."""
        if not self.vectorizer:
            raise ValueError("Vectorizer not loaded")

        # Clean texts
        cleaned = [self.clean_text(t) for t in texts]

        # Extract TF-IDF features
        tfidf_features = self.vectorizer.transform(cleaned)

        # Extract basic features
        basic_features = self.extract_basic_features(cleaned)

        # Combine features
        X = hstack([tfidf_features, csr_matrix(basic_features)])

        return X

    def predict(self, text: str, model_name: str = 'ensemble') -> Dict[str, Any]:
        """Make a prediction using the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        # Extract features
        X = self.extract_features([text])

        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Format result
        return {
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'real': float(probabilities[0]),
                'fake': float(probabilities[1])
            },
            'model_used': model_name
        }

    def batch_predict(self, texts: List[str], model_name: str = 'ensemble') -> List[Dict[str, Any]]:
        """Make predictions for multiple texts."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        model = self.models[model_name]

        # Extract features
        X = self.extract_features(texts)

        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Format results
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                'text': texts[i],
                'prediction': 'fake' if pred == 1 else 'real',
                'confidence': float(max(proba)),
                'probabilities': {
                    'real': float(proba[0]),
                    'fake': float(proba[1])
                },
                'model_used': model_name
            })

        return results

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about loaded models."""
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            return {
                'name': model_name,
                'loaded': True,
                'type': type(self.models[model_name]).__name__
            }

        # Return info for all models
        return {
            name: {
                'loaded': True,
                'type': type(model).__name__
            }
            for name, model in self.models.items()
        }

    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())
