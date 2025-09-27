# backend/app/services/data_service.py

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
from datetime import datetime
import json

from ml_models.preprocessing import TextProcessor, FeatureExtractor
from ml_models.evaluation.metrics import compute_classification_metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataService:
    """Manages the data loading and preprocessing pipeline."""

    def __init__(self):
        """Initialize the data service."""
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor()
        self.data_storage_path = Path(getattr(settings, 'DATA_STORAGE_PATH', 'data'))
        self.data_storage_path.mkdir(exist_ok=True)

        # Cache for processed datasets
        self.dataset_cache: Dict[str, pd.DataFrame] = {}
        self.cache_limit = 5

        logger.info("DataService initialized")

    async def load_dataset(self,
                          source: Union[str, Dict[str, Any]],
                          dataset_type: str = 'csv') -> str:
        """
        Load a dataset from various sources.

        Args:
            source: Path to file or dataset configuration
            dataset_type: Type of dataset ('csv', 'json', 'synthetic')

        Returns:
            Unique dataset identifier
        """
        try:
            if dataset_type == 'synthetic':
                # Generate synthetic dataset
                dataset = self._generate_synthetic_dataset(source)
                dataset_id = f"synthetic_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            elif dataset_type == 'csv':
                # Load from CSV file
                if isinstance(source, str):
                    dataset = pd.read_csv(source)
                    dataset_id = Path(source).stem
                else:
                    raise ValueError("CSV source must be a file path string")
            elif dataset_type == 'json':
                # Load from JSON file
                if isinstance(source, str):
                    dataset = pd.read_json(source)
                    dataset_id = Path(source).stem
                else:
                    raise ValueError("JSON source must be a file path string")
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            # Validate dataset structure
            self._validate_dataset(dataset)

            # Cache the dataset
            self._add_to_cache(dataset_id, dataset)

            # Save metadata
            await self._save_dataset_metadata(dataset_id, {
                'source': str(source),
                'dataset_type': dataset_type,
                'num_samples': len(dataset),
                'columns': list(dataset.columns),
                'loaded_at': datetime.utcnow().isoformat()
            })

            logger.info(f"Loaded dataset {dataset_id} with {len(dataset)} samples")
            return dataset_id

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _generate_synthetic_dataset(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate a synthetic dataset for testing purposes."""
        num_samples = config.get('num_samples', 1000)
        fake_ratio = config.get('fake_ratio', 0.3)
        random_seed = config.get('random_seed', 42)

        np.random.seed(random_seed)

        # Generate synthetic text data
        fake_templates = [
            "BREAKING: Shocking discovery about {topic} that {authority} doesn't want you to know!",
            "You won't believe what happened to {person} after they {action}!",
            "URGENT: {topic} is causing {negative_effect} - share before it's deleted!",
            "Scientists HATE this one weird trick about {topic}!",
            "EXPOSED: The truth about {topic} that mainstream media won't tell you!"
        ]

        real_templates = [
            "New research published in {journal} shows {finding} about {topic}.",
            "According to {authority}, {topic} has shown {improvement} in recent studies.",
            "{Expert} from {institution} discusses latest developments in {topic}.",
            "Study reveals {finding} about {topic} with implications for {application}.",
            "Research team at {institution} publishes findings on {topic} in peer-reviewed study."
        ]

        topics = ['climate change', 'vaccines', 'artificial intelligence', 'renewable energy',
                 'space exploration', 'medical research', 'technology', 'economics']
        authorities = ['WHO', 'NASA', 'university researchers', 'government officials']
        journals = ['Nature', 'Science', 'Cell', 'NEJM']
        institutions = ['MIT', 'Stanford', 'Harvard', 'Oxford']

        texts = []
        labels = []

        # Generate fake news samples
        num_fake = int(num_samples * fake_ratio)
        for _ in range(num_fake):
            template = np.random.choice(fake_templates)
            text = template.format(
                topic=np.random.choice(topics),
                authority=np.random.choice(authorities),
                person='a local resident',
                action='tried this simple trick',
                negative_effect='serious health issues',
                journal=np.random.choice(journals)
            )
            texts.append(text)
            labels.append(1)  # 1 = fake

        # Generate real news samples
        num_real = num_samples - num_fake
        for _ in range(num_real):
            template = np.random.choice(real_templates)
            text = template.format(
                journal=np.random.choice(journals),
                finding='significant improvements',
                topic=np.random.choice(topics),
                authority=np.random.choice(authorities),
                improvement='positive trends',
                expert='Dr. Smith',
                institution=np.random.choice(institutions),
                application='future research'
            )
            texts.append(text)
            labels.append(0)  # 0 = real

        # Create DataFrame
        dataset = pd.DataFrame({
            'text': texts,
            'label': labels
        })

        # Shuffle the dataset
        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        return dataset

    def _validate_dataset(self, dataset: pd.DataFrame) -> None:
        """Validate that the dataset has the required structure."""
        required_columns = ['text', 'label']

        for col in required_columns:
            if col not in dataset.columns:
                raise ValueError(f"Dataset missing required column: {col}")

        # Check for empty values
        if dataset['text'].isnull().any():
            logger.warning("Dataset contains null text values")

        # Validate label values
        unique_labels = dataset['label'].unique()
        if not all(label in [0, 1] for label in unique_labels):
            logger.warning("Dataset contains labels other than 0 and 1")

    async def get_processed_data_for_training(self,
                                            dataset_id: str,
                                            test_size: float = 0.2,
                                            validation_size: float = 0.1,
                                            random_state: int = 42,
                                            stratify: bool = True) -> Dict[str, Any]:
        """
        Process dataset for model training.

        Args:
            dataset_id: Identifier of the dataset to process
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits by label

        Returns:
            Dictionary containing processed training, validation, and test sets
        """
        try:
            # Load dataset
            dataset = await self._get_dataset(dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_id} not found")

            logger.info(f"Processing dataset {dataset_id} for training")

            # Clean text data
            logger.info("Cleaning text data")
            cleaned_texts = dataset['text'].apply(self.text_processor.clean).tolist()

            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(dataset['label'])

            # Extract basic features for traditional ML models
            logger.info("Extracting features")
            features_tfidf = self.feature_extractor.fit_transform_tfidf(cleaned_texts)
            features_linguistic = self.feature_extractor.extract_linguistic_features(cleaned_texts)
            features_stylistic = self.feature_extractor.extract_stylistic_features(cleaned_texts)

            # Combine features
            combined_features = np.hstack([
                features_tfidf.toarray() if hasattr(features_tfidf, 'toarray') else features_tfidf,
                features_linguistic,
                features_stylistic
            ])

            # Split data
            stratify_param = encoded_labels if stratify else None

            # First split: separate test set
            X_temp, X_test, y_temp, y_test, texts_temp, texts_test = train_test_split(
                combined_features, encoded_labels, cleaned_texts,
                test_size=test_size, random_state=random_state, stratify=stratify_param
            )

            # Second split: separate train and validation from remaining data
            if validation_size > 0:
                val_size_adjusted = validation_size / (1 - test_size)
                stratify_temp = y_temp if stratify else None

                X_train, X_val, y_train, y_val, texts_train, texts_val = train_test_split(
                    X_temp, y_temp, texts_temp,
                    test_size=val_size_adjusted, random_state=random_state, stratify=stratify_temp
                )
            else:
                X_train, y_train, texts_train = X_temp, y_temp, texts_temp
                X_val = y_val = texts_val = None

            # Prepare result
            result = {
                'dataset_id': dataset_id,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'data_splits': {
                    'train': {
                        'X': X_train,
                        'y': y_train,
                        'texts': texts_train,
                        'size': len(X_train)
                    },
                    'test': {
                        'X': X_test,
                        'y': y_test,
                        'texts': texts_test,
                        'size': len(X_test)
                    }
                },
                'feature_info': {
                    'tfidf_features': features_tfidf.shape[1] if hasattr(features_tfidf, 'shape') else len(features_tfidf[0]),
                    'linguistic_features': features_linguistic.shape[1],
                    'stylistic_features': features_stylistic.shape[1],
                    'total_features': combined_features.shape[1]
                },
                'label_encoding': {
                    'classes': label_encoder.classes_.tolist(),
                    'class_distribution': {
                        str(cls): int(np.sum(encoded_labels == i))
                        for i, cls in enumerate(label_encoder.classes_)
                    }
                }
            }

            # Add validation split if it exists
            if X_val is not None:
                result['data_splits']['validation'] = {
                    'X': X_val,
                    'y': y_val,
                    'texts': texts_val,
                    'size': len(X_val)
                }

            logger.info(f"Data processing completed for {dataset_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to process data for training: {e}")
            raise

    async def evaluate_predictions(self,
                                 dataset_id: str,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: Optional[np.ndarray] = None,
                                 model_name: str = "unknown") -> Dict[str, Any]:
        """
        Evaluate model predictions against true labels.

        Args:
            dataset_id: Identifier of the dataset
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            logger.info(f"Evaluating predictions for model {model_name} on dataset {dataset_id}")

            # Compute comprehensive metrics
            class_names = ['real', 'fake']  # Assuming binary classification
            metrics = compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                class_names=class_names
            )

            # Add metadata
            evaluation_result = {
                'dataset_id': dataset_id,
                'model_name': model_name,
                'evaluation_timestamp': datetime.utcnow().isoformat(),
                'sample_size': len(y_true),
                'metrics': metrics
            }

            # Save evaluation results
            await self._save_evaluation_results(dataset_id, model_name, evaluation_result)

            logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
            return evaluation_result

        except Exception as e:
            logger.error(f"Failed to evaluate predictions: {e}")
            raise

    async def get_dataset_statistics(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a dataset.

        Args:
            dataset_id: Identifier of the dataset

        Returns:
            Dictionary containing dataset statistics
        """
        try:
            dataset = await self._get_dataset(dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_id} not found")

            stats = {
                'dataset_id': dataset_id,
                'basic_info': {
                    'num_samples': len(dataset),
                    'num_columns': len(dataset.columns),
                    'columns': list(dataset.columns)
                },
                'label_distribution': {},
                'text_statistics': {},
                'data_quality': {}
            }

            # Label distribution
            if 'label' in dataset.columns:
                label_counts = dataset['label'].value_counts()
                stats['label_distribution'] = {
                    'counts': label_counts.to_dict(),
                    'proportions': (label_counts / len(dataset)).to_dict()
                }

            # Text statistics
            if 'text' in dataset.columns:
                text_lengths = dataset['text'].str.len()
                word_counts = dataset['text'].str.split().str.len()

                stats['text_statistics'] = {
                    'character_length': {
                        'mean': float(text_lengths.mean()),
                        'std': float(text_lengths.std()),
                        'min': int(text_lengths.min()),
                        'max': int(text_lengths.max()),
                        'median': float(text_lengths.median())
                    },
                    'word_count': {
                        'mean': float(word_counts.mean()),
                        'std': float(word_counts.std()),
                        'min': int(word_counts.min()),
                        'max': int(word_counts.max()),
                        'median': float(word_counts.median())
                    }
                }

            # Data quality checks
            stats['data_quality'] = {
                'missing_values': dataset.isnull().sum().to_dict(),
                'duplicate_rows': int(dataset.duplicated().sum()),
                'empty_text': int((dataset['text'].str.strip() == '').sum()) if 'text' in dataset.columns else 0
            }

            logger.info(f"Generated statistics for dataset {dataset_id}")
            return stats

        except Exception as e:
            logger.error(f"Failed to generate dataset statistics: {e}")
            raise

    async def create_cross_validation_splits(self,
                                           dataset_id: str,
                                           n_splits: int = 5,
                                           random_state: int = 42,
                                           stratify: bool = True) -> Dict[str, Any]:
        """
        Create cross-validation splits for a dataset.

        Args:
            dataset_id: Identifier of the dataset
            n_splits: Number of CV folds
            random_state: Random seed
            stratify: Whether to stratify splits

        Returns:
            Dictionary containing CV splits information
        """
        try:
            dataset = await self._get_dataset(dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_id} not found")

            # Prepare data
            X = dataset['text'].tolist()
            y = dataset['label'].values

            # Create cross-validation splitter
            if stratify:
                cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            else:
                from sklearn.model_selection import KFold
                cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            # Generate splits
            splits = []
            for fold_idx, (train_indices, val_indices) in enumerate(cv_splitter.split(X, y)):
                split_info = {
                    'fold': fold_idx,
                    'train_indices': train_indices.tolist(),
                    'val_indices': val_indices.tolist(),
                    'train_size': len(train_indices),
                    'val_size': len(val_indices),
                    'train_label_distribution': {
                        str(label): int(np.sum(y[train_indices] == label))
                        for label in np.unique(y)
                    },
                    'val_label_distribution': {
                        str(label): int(np.sum(y[val_indices] == label))
                        for label in np.unique(y)
                    }
                }
                splits.append(split_info)

            cv_result = {
                'dataset_id': dataset_id,
                'n_splits': n_splits,
                'stratified': stratify,
                'random_state': random_state,
                'splits': splits,
                'created_at': datetime.utcnow().isoformat()
            }

            logger.info(f"Created {n_splits}-fold CV splits for dataset {dataset_id}")
            return cv_result

        except Exception as e:
            logger.error(f"Failed to create CV splits: {e}")
            raise

    async def list_datasets(self) -> Dict[str, Any]:
        """List all available datasets."""
        try:
            datasets = []

            # Check cached datasets
            for dataset_id in self.dataset_cache.keys():
                dataset = self.dataset_cache[dataset_id]
                datasets.append({
                    'dataset_id': dataset_id,
                    'num_samples': len(dataset),
                    'cached': True
                })

            # Check saved datasets
            for metadata_file in self.data_storage_path.glob("*_metadata.json"):
                dataset_id = metadata_file.stem.replace('_metadata', '')
                if dataset_id not in self.dataset_cache:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            datasets.append({
                                'dataset_id': dataset_id,
                                'num_samples': metadata.get('num_samples', 'unknown'),
                                'cached': False,
                                'loaded_at': metadata.get('loaded_at', 'unknown')
                            })
                    except Exception as e:
                        logger.warning(f"Failed to read metadata for {dataset_id}: {e}")

            return {
                'datasets': datasets,
                'total_count': len(datasets)
            }

        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            raise

    async def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset from cache and storage."""
        try:
            # Remove from cache
            if dataset_id in self.dataset_cache:
                del self.dataset_cache[dataset_id]

            # Remove from storage
            metadata_file = self.data_storage_path / f"{dataset_id}_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()

            # Remove evaluation results
            eval_pattern = f"{dataset_id}_*_evaluation.json"
            for eval_file in self.data_storage_path.glob(eval_pattern):
                eval_file.unlink()

            logger.info(f"Dataset {dataset_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            return False

    async def _get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Retrieve dataset from cache or storage."""
        # Check cache first
        if dataset_id in self.dataset_cache:
            return self.dataset_cache[dataset_id]

        # Try to load from storage if metadata exists
        metadata_file = self.data_storage_path / f"{dataset_id}_metadata.json"
        if metadata_file.exists():
            logger.info(f"Dataset {dataset_id} not in cache, would need to reload from source")
            # In a full implementation, you would store the actual data and reload it here
            # For now, return None to indicate it needs to be reloaded

        return None

    async def _save_dataset_metadata(self, dataset_id: str, metadata: Dict[str, Any]) -> None:
        """Save dataset metadata to storage."""
        try:
            metadata_file = self.data_storage_path / f"{dataset_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata for {dataset_id}: {e}")

    async def _save_evaluation_results(self, dataset_id: str, model_name: str, results: Dict[str, Any]) -> None:
        """Save evaluation results to storage."""
        try:
            results_file = self.data_storage_path / f"{dataset_id}_{model_name}_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

    def _add_to_cache(self, dataset_id: str, dataset: pd.DataFrame) -> None:
        """Add dataset to cache with LRU eviction."""
        # Remove oldest item if cache is full
        if len(self.dataset_cache) >= self.cache_limit:
            oldest_id = next(iter(self.dataset_cache))
            del self.dataset_cache[oldest_id]

        self.dataset_cache[dataset_id] = dataset