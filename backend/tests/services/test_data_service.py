# backend/tests/services/test_data_service.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import json

from app.services.data_service import DataService
from app.utils.data_preprocessing import DataPreprocessor
from app.utils.validators import DataValidator


@pytest.mark.unit
class TestDataService:
    """Test suite for DataService business logic."""

    @pytest.fixture
    def data_service(self):
        """Create a DataService instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = DataService()
            service.data_storage_path = Path(temp_dir)
            yield service

    @pytest.fixture
    def sample_dataset(self):
        """Sample dataset for testing."""
        return pd.DataFrame({
            "text": [
                "Scientists discover new renewable energy source",
                "BREAKING: Government hiding alien contact!",
                "Local university announces new research program",
                "You won't believe this miracle cure doctors hate!",
                "Climate change report published in Nature"
            ],
            "label": [0, 1, 0, 1, 0],  # 0=real, 1=fake
            "source": ["reuters", "blog", "university", "blog", "nature"],
            "timestamp": pd.date_range("2023-01-01", periods=5)
        })

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data as string."""
        return """text,label,source
Scientists discover breakthrough,0,reuters
Fake news about aliens,1,blog
University research published,0,university"""

    def test_load_dataset_from_file(self, data_service, sample_csv_data):
        """Test loading dataset from file."""
        # Create temporary CSV file
        csv_file = data_service.data_storage_path / "test_data.csv"
        csv_file.write_text(sample_csv_data)

        dataset = data_service.load_dataset(str(csv_file))

        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset) == 3
        assert "text" in dataset.columns
        assert "label" in dataset.columns

    def test_load_dataset_from_url(self, data_service):
        """Test loading dataset from URL."""
        mock_response = MagicMock()
        mock_response.text = "text,label\nTest news,0\nFake news,1"

        with patch('requests.get', return_value=mock_response):
            dataset = data_service.load_dataset("https://example.com/data.csv")

            assert isinstance(dataset, pd.DataFrame)
            assert len(dataset) == 2

    def test_load_dataset_invalid_format(self, data_service):
        """Test loading dataset with invalid format."""
        # Create file with invalid format
        invalid_file = data_service.data_storage_path / "invalid.txt"
        invalid_file.write_text("This is not a valid dataset")

        with pytest.raises(ValueError, match="Unsupported file format"):
            data_service.load_dataset(str(invalid_file))

    def test_save_dataset(self, data_service, sample_dataset):
        """Test saving dataset to file."""
        output_path = data_service.data_storage_path / "output.csv"

        data_service.save_dataset(sample_dataset, str(output_path))

        assert output_path.exists()

        # Verify saved data
        loaded_data = pd.read_csv(output_path)
        assert len(loaded_data) == len(sample_dataset)

    def test_preprocess_dataset(self, data_service, sample_dataset):
        """Test dataset preprocessing."""
        with patch.object(DataPreprocessor, 'clean_text') as mock_clean, \
             patch.object(DataPreprocessor, 'extract_features') as mock_features:

            mock_clean.return_value = "cleaned text"
            mock_features.return_value = {"feature1": 1.0, "feature2": 0.5}

            preprocessed = data_service.preprocess_dataset(
                sample_dataset,
                clean_text=True,
                extract_features=True
            )

            assert "cleaned_text" in preprocessed.columns
            assert "features" in preprocessed.columns
            mock_clean.assert_called()

    def test_validate_dataset(self, data_service, sample_dataset):
        """Test dataset validation."""
        with patch.object(DataValidator, 'validate_schema') as mock_validate:
            mock_validate.return_value = {"valid": True, "errors": []}

            validation_result = data_service.validate_dataset(sample_dataset)

            assert validation_result["valid"] is True
            mock_validate.assert_called_once()

    def test_validate_dataset_invalid(self, data_service):
        """Test validation of invalid dataset."""
        invalid_dataset = pd.DataFrame({
            "text": ["sample text"],
            # Missing required 'label' column
            "source": ["reuters"]
        })

        with patch.object(DataValidator, 'validate_schema') as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "errors": ["Missing required column: label"]
            }

            validation_result = data_service.validate_dataset(invalid_dataset)

            assert validation_result["valid"] is False
            assert len(validation_result["errors"]) > 0

    def test_split_dataset(self, data_service, sample_dataset):
        """Test dataset splitting into train/validation/test sets."""
        splits = data_service.split_dataset(
            sample_dataset,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42
        )

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        total_samples = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total_samples == len(sample_dataset)

    def test_balance_dataset(self, data_service, sample_dataset):
        """Test dataset balancing."""
        # Create imbalanced dataset
        imbalanced = pd.concat([
            sample_dataset[sample_dataset["label"] == 0],  # Real news
            sample_dataset[sample_dataset["label"] == 1].iloc[:1]  # Only 1 fake news
        ])

        balanced = data_service.balance_dataset(imbalanced, strategy="undersample")

        # Should have equal number of each class
        class_counts = balanced["label"].value_counts()
        assert class_counts[0] == class_counts[1]

    def test_augment_dataset(self, data_service, sample_dataset):
        """Test dataset augmentation."""
        with patch.object(data_service, '_augment_text') as mock_augment:
            mock_augment.return_value = ["augmented text 1", "augmented text 2"]

            augmented = data_service.augment_dataset(
                sample_dataset,
                augmentation_factor=2
            )

            assert len(augmented) > len(sample_dataset)
            mock_augment.assert_called()

    def test_get_dataset_statistics(self, data_service, sample_dataset):
        """Test calculation of dataset statistics."""
        stats = data_service.get_dataset_statistics(sample_dataset)

        assert "num_samples" in stats
        assert "num_features" in stats
        assert "class_distribution" in stats
        assert "missing_values" in stats

        assert stats["num_samples"] == 5
        assert stats["class_distribution"][0] == 3  # 3 real news
        assert stats["class_distribution"][1] == 2  # 2 fake news

    def test_filter_dataset(self, data_service, sample_dataset):
        """Test dataset filtering."""
        # Filter by source
        filtered = data_service.filter_dataset(
            sample_dataset,
            filters={"source": ["reuters", "nature"]}
        )

        assert len(filtered) == 2
        assert all(filtered["source"].isin(["reuters", "nature"]))

    def test_merge_datasets(self, data_service, sample_dataset):
        """Test merging multiple datasets."""
        dataset2 = pd.DataFrame({
            "text": ["Additional news 1", "Additional news 2"],
            "label": [0, 1],
            "source": ["cnn", "blog"],
            "timestamp": pd.date_range("2023-02-01", periods=2)
        })

        merged = data_service.merge_datasets([sample_dataset, dataset2])

        assert len(merged) == len(sample_dataset) + len(dataset2)
        assert "text" in merged.columns

    def test_export_dataset(self, data_service, sample_dataset):
        """Test exporting dataset to different formats."""
        # Test CSV export
        csv_path = data_service.data_storage_path / "export.csv"
        data_service.export_dataset(sample_dataset, str(csv_path), format="csv")
        assert csv_path.exists()

        # Test JSON export
        json_path = data_service.data_storage_path / "export.json"
        data_service.export_dataset(sample_dataset, str(json_path), format="json")
        assert json_path.exists()

    def test_create_data_pipeline(self, data_service, sample_dataset):
        """Test creation of data processing pipeline."""
        pipeline_config = {
            "steps": [
                {"name": "clean_text", "params": {"remove_urls": True}},
                {"name": "extract_features", "params": {"feature_type": "tfidf"}},
                {"name": "balance_dataset", "params": {"strategy": "oversample"}}
            ]
        }

        with patch.object(data_service, 'preprocess_dataset') as mock_preprocess, \
             patch.object(data_service, 'balance_dataset') as mock_balance:

            mock_preprocess.return_value = sample_dataset
            mock_balance.return_value = sample_dataset

            processed = data_service.create_data_pipeline(sample_dataset, pipeline_config)

            assert processed is not None
            mock_preprocess.assert_called()
            mock_balance.assert_called()

    def test_dataset_versioning(self, data_service, sample_dataset):
        """Test dataset versioning functionality."""
        version_id = data_service.save_dataset_version(
            sample_dataset,
            version_name="v1.0",
            description="Initial dataset"
        )

        assert version_id is not None

        # Load version
        loaded_dataset = data_service.load_dataset_version(version_id)
        assert len(loaded_dataset) == len(sample_dataset)

    def test_data_quality_assessment(self, data_service, sample_dataset):
        """Test data quality assessment."""
        quality_report = data_service.assess_data_quality(sample_dataset)

        assert "completeness_score" in quality_report
        assert "consistency_score" in quality_report
        assert "quality_issues" in quality_report
        assert "recommendations" in quality_report

        assert 0 <= quality_report["completeness_score"] <= 1

    def test_dataset_sampling(self, data_service, sample_dataset):
        """Test dataset sampling strategies."""
        # Random sampling
        random_sample = data_service.sample_dataset(
            sample_dataset,
            strategy="random",
            sample_size=3,
            random_seed=42
        )

        assert len(random_sample) == 3

        # Stratified sampling
        stratified_sample = data_service.sample_dataset(
            sample_dataset,
            strategy="stratified",
            sample_size=4,
            stratify_column="label"
        )

        assert len(stratified_sample) == 4

    @pytest.mark.slow
    def test_large_dataset_processing(self, data_service, performance_timer):
        """Test processing of large datasets."""
        timer = performance_timer(threshold_seconds=2.0)

        # Create large synthetic dataset
        large_dataset = pd.DataFrame({
            "text": [f"Sample text {i}" for i in range(10000)],
            "label": np.random.randint(0, 2, 10000)
        })

        with patch.object(data_service, '_optimize_memory_usage'):
            stats = data_service.get_dataset_statistics(large_dataset)

        elapsed = timer()
        assert elapsed < 2.0, f"Large dataset processing took {elapsed:.3f}s"
        assert stats["num_samples"] == 10000

    def test_data_anonymization(self, data_service):
        """Test data anonymization functionality."""
        sensitive_dataset = pd.DataFrame({
            "text": ["John Smith reported the news", "Mary Johnson discovered this"],
            "label": [0, 1],
            "author": ["john.smith@email.com", "mary.j@email.com"]
        })

        anonymized = data_service.anonymize_dataset(
            sensitive_dataset,
            anonymize_columns=["author"],
            anonymization_strategy="hash"
        )

        assert "author" in anonymized.columns
        assert "john.smith@email.com" not in anonymized["author"].values

    def test_dataset_comparison(self, data_service, sample_dataset):
        """Test comparison between datasets."""
        # Create modified dataset
        modified_dataset = sample_dataset.copy()
        modified_dataset.loc[0, "label"] = 1  # Change one label

        comparison = data_service.compare_datasets(sample_dataset, modified_dataset)

        assert "schema_differences" in comparison
        assert "data_differences" in comparison
        assert "statistical_differences" in comparison

    def test_data_lineage_tracking(self, data_service, sample_dataset):
        """Test data lineage and provenance tracking."""
        lineage_id = data_service.track_data_lineage(
            sample_dataset,
            source_info={"origin": "test_data", "processed_at": "2023-01-01"},
            transformation_history=["loaded", "cleaned", "validated"]
        )

        assert lineage_id is not None

        lineage_info = data_service.get_data_lineage(lineage_id)
        assert "source_info" in lineage_info
        assert "transformation_history" in lineage_info

    def test_error_handling_corrupt_data(self, data_service):
        """Test handling of corrupt or malformed data."""
        corrupt_file = data_service.data_storage_path / "corrupt.csv"
        corrupt_file.write_text("text,label\nValid data,0\nCorrupt line missing comma\n")

        with pytest.raises(ValueError, match="Unable to parse data"):
            data_service.load_dataset(str(corrupt_file), strict=True)

        # With strict=False, should handle gracefully
        dataset = data_service.load_dataset(str(corrupt_file), strict=False)
        assert len(dataset) == 1  # Only valid row loaded

    def test_memory_efficient_processing(self, data_service):
        """Test memory-efficient processing for large datasets."""
        # Simulate large dataset processing in chunks
        with patch.object(data_service, '_process_in_chunks') as mock_chunks:
            mock_chunks.return_value = pd.DataFrame({"text": ["processed"], "label": [0]})

            result = data_service.process_large_dataset(
                file_path="large_file.csv",
                chunk_size=1000,
                processing_function=lambda x: x
            )

            assert result is not None
            mock_chunks.assert_called()

    def test_data_validation_rules(self, data_service):
        """Test custom data validation rules."""
        validation_rules = {
            "text_min_length": 10,
            "text_max_length": 1000,
            "required_columns": ["text", "label"],
            "label_values": [0, 1]
        }

        valid_data = pd.DataFrame({
            "text": ["This is a valid news article with sufficient length"],
            "label": [0]
        })

        invalid_data = pd.DataFrame({
            "text": ["Short"],  # Too short
            "label": [2]  # Invalid label
        })

        valid_result = data_service.validate_with_rules(valid_data, validation_rules)
        invalid_result = data_service.validate_with_rules(invalid_data, validation_rules)

        assert valid_result["valid"] is True
        assert invalid_result["valid"] is False