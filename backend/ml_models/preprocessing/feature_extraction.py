# backend/ml_models/preprocessing/feature_extraction.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts various numerical features from cleaned text for traditional ML models.
    This class implements comprehensive feature extraction for fake news detection,
    including linguistic, stylistic, and semantic features.
    """

    def __init__(self,
                 max_tfidf_features: int = 5000,
                 tfidf_ngram_range: Tuple[int, int] = (1, 3),
                 max_topic_features: int = 20,
                 include_sentiment: bool = True,
                 include_readability: bool = True):
        """
        Initialize the FeatureExtractor with configurable parameters.

        Args:
            max_tfidf_features: Maximum number of TF-IDF features
            tfidf_ngram_range: N-gram range for TF-IDF (min_n, max_n)
            max_topic_features: Number of topics for LDA topic modeling
            include_sentiment: Whether to include sentiment analysis features
            include_readability: Whether to include readability metrics
        """
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.max_topic_features = max_topic_features
        self.include_sentiment = include_sentiment
        self.include_readability = include_readability

        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=tfidf_ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )

        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 1),
            stop_words='english'
        )

        # Topic modeling
        self.lda_model = LatentDirichletAllocation(
            n_components=max_topic_features,
            random_state=42,
            max_iter=10
        )

        # Fitted status
        self.is_fitted = False

    def extract_linguistic_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extracts linguistic features like readability, sentiment, and complexity.

        Args:
            texts: List of text samples

        Returns:
            DataFrame with linguistic features
        """
        features = []

        for text in texts:
            text_features = {}

            # Basic text statistics
            text_features.update(self._extract_basic_stats(text))

            # Readability metrics (if available)
            if self.include_readability:
                text_features.update(self._extract_readability_features(text))

            # Sentiment features (simplified implementation)
            if self.include_sentiment:
                text_features.update(self._extract_sentiment_features(text))

            # POS tag features (simplified)
            text_features.update(self._extract_pos_features(text))

            features.append(text_features)

        return pd.DataFrame(features)

    def extract_stylistic_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extracts stylistic features based on writing patterns and style.

        Args:
            texts: List of text samples

        Returns:
            DataFrame with stylistic features
        """
        features = []

        for text in texts:
            style_features = {}

            # Punctuation patterns
            style_features.update(self._extract_punctuation_features(text))

            # Capitalization patterns
            style_features.update(self._extract_capitalization_features(text))

            # Sentence and word patterns
            style_features.update(self._extract_structure_features(text))

            # Character-level features
            style_features.update(self._extract_character_features(text))

            features.append(style_features)

        return pd.DataFrame(features)

    def fit_transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Fits and transforms text data into TF-IDF features.

        Args:
            texts: List of text samples

        Returns:
            TF-IDF feature matrix
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_features.toarray()

    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        Transforms text data using pre-fitted TF-IDF vectorizer.

        Args:
            texts: List of text samples

        Returns:
            TF-IDF feature matrix
        """
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            raise ValueError("TF-IDF vectorizer must be fitted before transform")

        tfidf_features = self.tfidf_vectorizer.transform(texts)
        return tfidf_features.toarray()

    def extract_topic_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Extracts topic modeling features using Latent Dirichlet Allocation.

        Args:
            texts: List of text samples
            fit: Whether to fit the model (True for training, False for inference)

        Returns:
            Topic distribution matrix
        """
        # First get count features for LDA
        if fit:
            count_features = self.count_vectorizer.fit_transform(texts)
            topic_features = self.lda_model.fit_transform(count_features)
        else:
            count_features = self.count_vectorizer.transform(texts)
            topic_features = self.lda_model.transform(count_features)

        return topic_features

    def extract_all_features(self, texts: List[str], fit: bool = True) -> pd.DataFrame:
        """
        Master method that extracts all feature types and combines them.

        Args:
            texts: List of text samples
            fit: Whether to fit vectorizers (True for training, False for inference)

        Returns:
            Combined feature DataFrame ready for model training
        """
        logger.info(f"Extracting all features for {len(texts)} texts")

        all_features = []

        # Extract linguistic features
        linguistic_features = self.extract_linguistic_features(texts)
        logger.info(f"Extracted {linguistic_features.shape[1]} linguistic features")

        # Extract stylistic features
        stylistic_features = self.extract_stylistic_features(texts)
        logger.info(f"Extracted {stylistic_features.shape[1]} stylistic features")

        # Extract TF-IDF features
        if fit:
            tfidf_features = self.fit_transform_tfidf(texts)
        else:
            tfidf_features = self.transform_tfidf(texts)

        tfidf_df = pd.DataFrame(
            tfidf_features,
            columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        logger.info(f"Extracted {tfidf_features.shape[1]} TF-IDF features")

        # Extract topic features
        topic_features = self.extract_topic_features(texts, fit=fit)
        topic_df = pd.DataFrame(
            topic_features,
            columns=[f'topic_{i}' for i in range(topic_features.shape[1])]
        )
        logger.info(f"Extracted {topic_features.shape[1]} topic features")

        # Combine all features
        combined_features = pd.concat([
            linguistic_features.reset_index(drop=True),
            stylistic_features.reset_index(drop=True),
            tfidf_df.reset_index(drop=True),
            topic_df.reset_index(drop=True)
        ], axis=1)

        self.is_fitted = fit
        logger.info(f"Total combined features: {combined_features.shape[1]}")

        return combined_features

    def _extract_basic_stats(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics."""
        words = text.split()
        sentences = text.split('.')

        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }

    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability metrics (simplified implementation)."""
        words = text.split()
        sentences = text.split('.')
        syllables = sum(self._count_syllables(word) for word in words)

        # Simplified Flesch Reading Ease approximation
        if len(sentences) > 0 and len(words) > 0:
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        else:
            flesch_score = 0

        return {
            'flesch_reading_ease': max(0, min(100, flesch_score)),
            'syllable_count': syllables,
            'avg_syllables_per_word': avg_syllables_per_word if words else 0
        }

    def _extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features (simplified implementation)."""
        # This is a simplified implementation
        # In production, you'd use VADER, TextBlob, or similar
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate']

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'sentiment_polarity': (positive_count - negative_count) / len(words) if words else 0
        }

    def _extract_pos_features(self, text: str) -> Dict[str, float]:
        """Extract part-of-speech features (simplified)."""
        # Simplified POS feature extraction
        # In production, you'd use spaCy or NLTK
        words = text.split()
        return {
            'word_diversity': len(set(words)) / len(words) if words else 0,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0
        }

    def _extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """Extract punctuation-based features."""
        return {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'comma_count': text.count(','),
            'period_count': text.count('.'),
            'semicolon_count': text.count(';'),
            'colon_count': text.count(':'),
            'quote_count': text.count('"') + text.count("'"),
            'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0
        }

    def _extract_capitalization_features(self, text: str) -> Dict[str, float]:
        """Extract capitalization pattern features."""
        words = text.split()
        return {
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'title_case_ratio': sum(1 for word in words if word.istitle()) / len(words) if words else 0,
            'all_caps_words': sum(1 for word in words if word.isupper() and len(word) > 1),
            'sentence_start_caps': sum(1 for sentence in text.split('.')
                                     if sentence.strip() and sentence.strip()[0].isupper())
        }

    def _extract_structure_features(self, text: str) -> Dict[str, float]:
        """Extract structural features about text organization."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()

        return {
            'short_sentence_ratio': sum(1 for s in sentences if len(s.split()) < 10) / len(sentences) if sentences else 0,
            'long_sentence_ratio': sum(1 for s in sentences if len(s.split()) > 25) / len(sentences) if sentences else 0,
            'short_word_ratio': sum(1 for word in words if len(word) < 4) / len(words) if words else 0,
            'long_word_ratio': sum(1 for word in words if len(word) > 10) / len(words) if words else 0
        }

    def _extract_character_features(self, text: str) -> Dict[str, float]:
        """Extract character-level features."""
        return {
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'space_ratio': sum(1 for c in text if c.isspace()) / len(text) if text else 0,
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0
        }

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simplified implementation).

        Args:
            word: Word to count syllables for

        Returns:
            Estimated syllable count
        """
        word = word.lower().strip()
        if not word:
            return 0

        # Simple syllable counting heuristic
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False

        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    def get_feature_names(self) -> List[str]:
        """
        Get names of all extracted features.

        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting feature names")

        feature_names = []

        # Add linguistic feature names (you'd need to track these during extraction)
        # Add stylistic feature names
        # Add TF-IDF feature names
        if hasattr(self.tfidf_vectorizer, 'feature_names_out'):
            feature_names.extend(self.tfidf_vectorizer.get_feature_names_out())

        # Add topic feature names
        feature_names.extend([f'topic_{i}' for i in range(self.max_topic_features)])

        return feature_names

    def save_vectorizers(self, filepath_prefix: str) -> None:
        """
        Save fitted vectorizers to disk.

        Args:
            filepath_prefix: Prefix for saved files
        """
        import joblib

        if self.is_fitted:
            joblib.dump(self.tfidf_vectorizer, f"{filepath_prefix}_tfidf.pkl")
            joblib.dump(self.count_vectorizer, f"{filepath_prefix}_count.pkl")
            joblib.dump(self.lda_model, f"{filepath_prefix}_lda.pkl")
            logger.info(f"Vectorizers saved with prefix: {filepath_prefix}")
        else:
            logger.warning("Vectorizers not fitted, cannot save")

    def load_vectorizers(self, filepath_prefix: str) -> None:
        """
        Load fitted vectorizers from disk.

        Args:
            filepath_prefix: Prefix for saved files
        """
        import joblib

        try:
            self.tfidf_vectorizer = joblib.load(f"{filepath_prefix}_tfidf.pkl")
            self.count_vectorizer = joblib.load(f"{filepath_prefix}_count.pkl")
            self.lda_model = joblib.load(f"{filepath_prefix}_lda.pkl")
            self.is_fitted = True
            logger.info(f"Vectorizers loaded from prefix: {filepath_prefix}")
        except FileNotFoundError as e:
            logger.error(f"Could not load vectorizers: {e}")
            raise