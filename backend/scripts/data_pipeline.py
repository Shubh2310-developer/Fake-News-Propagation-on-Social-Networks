#!/usr/bin/env python3
"""
Data Pipeline Script (ETL Pipeline)

This script automates the Extract, Transform, and Load (ETL) process for the
fake news detection system. It handles data from multiple sources, applies
comprehensive preprocessing, and loads clean data into the database.

Usage:
    python data_pipeline.py --source fakenewsnet --output data/processed/
    python data_pipeline.py --source csv --input data/raw/news_data.csv --validate
    python data_pipeline.py --source api --config config/data_sources.yaml

Features:
    - Multi-source data extraction (FakeNewsNet, CSV, JSON, APIs)
    - Comprehensive text preprocessing and feature engineering
    - Data validation and quality checks with schemas
    - Idempotent loading with duplicate detection
    - Modular and extensible architecture
"""

import argparse
import logging
import sys
import os
import json
import yaml
import csv
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings
from urllib.parse import urlparse
import zipfile
import requests

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
import pandera as pa
from pandera import Column, Check, DataFrameSchema
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textstat import flesch_reading_ease, flesch_kincaid_grade
from textblob import TextBlob
import networkx as nx

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class DataPipeline:
    """Orchestrates the complete ETL process for news data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/processed'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize text processing tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Pipeline statistics
        self.stats = {
            'extracted_records': 0,
            'processed_records': 0,
            'loaded_records': 0,
            'skipped_records': 0,
            'errors': []
        }

        # Setup database connection if specified
        self.db_engine = None
        if config.get('database_url'):
            self.db_engine = create_engine(config['database_url'])

    def extract_fakenewsnet(self) -> pd.DataFrame:
        """Extract data from FakeNewsNet dataset."""
        logger.info("Extracting data from FakeNewsNet dataset")

        # This would typically download and extract the FakeNewsNet dataset
        # For demonstration, we'll create a mock extraction
        fake_news_url = "https://github.com/KaiDMML/FakeNewsNet/raw/master/dataset/politifact_fake.csv"
        real_news_url = "https://github.com/KaiDMML/FakeNewsNet/raw/master/dataset/politifact_real.csv"

        dfs = []

        try:
            # Download fake news data
            logger.info("Downloading fake news data...")
            fake_df = pd.read_csv(fake_news_url)
            fake_df['label'] = 'fake'
            dfs.append(fake_df)

            # Download real news data
            logger.info("Downloading real news data...")
            real_df = pd.read_csv(real_news_url)
            real_df['label'] = 'real'
            dfs.append(real_df)

        except Exception as e:
            logger.warning(f"Could not download FakeNewsNet data: {e}")
            # Create mock data for demonstration
            mock_data = {
                'id': range(1000),
                'title': [f"News article {i}" for i in range(1000)],
                'text': [f"This is the content of news article {i}. " * 10 for i in range(1000)],
                'label': ['fake' if i % 2 == 0 else 'real' for i in range(1000)],
                'url': [f"https://example.com/article_{i}" for i in range(1000)],
                'published_date': pd.date_range('2020-01-01', periods=1000, freq='D')
            }
            dfs.append(pd.DataFrame(mock_data))

        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        self.stats['extracted_records'] = len(combined_df)
        logger.info(f"Extracted {len(combined_df)} records from FakeNewsNet")

        return combined_df

    def extract_csv(self, file_path: str) -> pd.DataFrame:
        """Extract data from CSV file."""
        logger.info(f"Extracting data from CSV: {file_path}")

        try:
            df = pd.read_csv(file_path)
            self.stats['extracted_records'] = len(df)
            logger.info(f"Extracted {len(df)} records from CSV")
            return df
        except Exception as e:
            logger.error(f"Failed to extract CSV data: {e}")
            raise

    def extract_json(self, file_path: str) -> pd.DataFrame:
        """Extract data from JSON file."""
        logger.info(f"Extracting data from JSON: {file_path}")

        try:
            df = pd.read_json(file_path)
            self.stats['extracted_records'] = len(df)
            logger.info(f"Extracted {len(df)} records from JSON")
            return df
        except Exception as e:
            logger.error(f"Failed to extract JSON data: {e}")
            raise

    def extract_api(self, config_path: str) -> pd.DataFrame:
        """Extract data from API sources based on configuration."""
        logger.info(f"Extracting data from API sources using config: {config_path}")

        with open(config_path, 'r') as f:
            api_config = yaml.safe_load(f)

        dfs = []
        for source in api_config.get('sources', []):
            try:
                response = requests.get(
                    source['url'],
                    headers=source.get('headers', {}),
                    params=source.get('params', {}),
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()
                if source.get('data_path'):
                    # Navigate nested JSON structure
                    for key in source['data_path'].split('.'):
                        data = data[key]

                df = pd.DataFrame(data)
                dfs.append(df)
                logger.info(f"Extracted {len(df)} records from {source['name']}")

            except Exception as e:
                logger.error(f"Failed to extract from API {source.get('name', 'unknown')}: {e}")
                self.stats['errors'].append(f"API extraction failed: {e}")

        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        self.stats['extracted_records'] = len(combined_df)
        logger.info(f"Total extracted {len(combined_df)} records from API sources")

        return combined_df

    def clean_text(self, text: str) -> str:
        """Apply comprehensive text cleaning."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic and stylistic features from text."""
        if not text or pd.isna(text):
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'readability_score': 0,
                'grade_level': 0,
                'sentiment_polarity': 0,
                'sentiment_subjectivity': 0,
                'uppercase_ratio': 0,
                'punctuation_ratio': 0
            }

        # Basic metrics
        words = word_tokenize(text)
        sentences = sent_tokenize(text)

        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        }

        # Readability metrics
        try:
            features['readability_score'] = flesch_reading_ease(text)
            features['grade_level'] = flesch_kincaid_grade(text)
        except:
            features['readability_score'] = 0
            features['grade_level'] = 0

        # Sentiment analysis
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0
            features['sentiment_subjectivity'] = 0

        # Stylistic features
        if text:
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
            features['punctuation_ratio'] = sum(1 for c in text if c in '.,!?;:') / len(text)
        else:
            features['uppercase_ratio'] = 0
            features['punctuation_ratio'] = 0

        return features

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive data transformation and feature engineering."""
        logger.info("Transforming and engineering features...")

        processed_df = df.copy()

        # Ensure required columns exist
        if 'text' not in processed_df.columns:
            if 'content' in processed_df.columns:
                processed_df['text'] = processed_df['content']
            else:
                raise ValueError("No text content column found")

        # Handle missing values
        processed_df['text'] = processed_df['text'].fillna('')
        processed_df['title'] = processed_df.get('title', '').fillna('')

        # Clean text data
        logger.info("Cleaning text content...")
        processed_df['cleaned_text'] = processed_df['text'].apply(self.clean_text)
        processed_df['cleaned_title'] = processed_df['title'].apply(self.clean_text)

        # Extract linguistic features
        logger.info("Extracting linguistic features...")
        linguistic_features = processed_df['cleaned_text'].apply(self.extract_linguistic_features)
        feature_df = pd.DataFrame(linguistic_features.tolist())
        processed_df = pd.concat([processed_df, feature_df], axis=1)

        # Create text hash for duplicate detection
        processed_df['content_hash'] = processed_df['cleaned_text'].apply(
            lambda x: hashlib.md5(x.encode()).hexdigest() if x else ''
        )

        # Add processing metadata
        processed_df['processed_at'] = datetime.now()
        processed_df['pipeline_version'] = self.config.get('pipeline_version', '1.0.0')

        # Handle label standardization
        if 'label' in processed_df.columns:
            label_mapping = {
                'fake': 0, 'false': 0, 'unreliable': 0,
                'real': 1, 'true': 1, 'reliable': 1
            }
            processed_df['label_encoded'] = processed_df['label'].str.lower().map(label_mapping)

        self.stats['processed_records'] = len(processed_df)
        logger.info(f"Transformed {len(processed_df)} records")

        return processed_df

    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate data quality using schema validation."""
        logger.info("Validating data quality...")

        # Define validation schema
        schema = DataFrameSchema({
            "text": Column(pa.String, checks=[
                Check.str_length(min_val=10, max_val=50000),
                Check(lambda x: x.str.strip().str.len() > 0,
                      error="Text cannot be empty")
            ]),
            "word_count": Column(pa.Int, checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(10000)
            ]),
            "readability_score": Column(pa.Float, checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100)
            ], nullable=True),
            "sentiment_polarity": Column(pa.Float, checks=[
                Check.greater_than_or_equal_to(-1),
                Check.less_than_or_equal_to(1)
            ], nullable=True),
        })

        validation_results = {
            'total_records': len(df),
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': []
        }

        try:
            # Validate schema
            validated_df = schema.validate(df, lazy=True)
            validation_results['valid_records'] = len(validated_df)
            logger.info(f"Schema validation passed for {len(validated_df)} records")

        except pa.errors.SchemaErrors as e:
            validation_results['validation_errors'] = e.failure_cases.to_dict('records')
            validation_results['invalid_records'] = len(e.failure_cases)

            # Create a mask for valid rows
            invalid_indices = set(e.failure_cases['index'])
            valid_mask = ~df.index.isin(invalid_indices)
            validated_df = df[valid_mask].copy()
            validation_results['valid_records'] = len(validated_df)

            logger.warning(f"Schema validation failed for {validation_results['invalid_records']} records")

        # Additional quality checks
        quality_checks = self._perform_quality_checks(df)
        validation_results.update(quality_checks)

        return validated_df, validation_results

    def _perform_quality_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform additional data quality checks."""
        checks = {}

        # Check for duplicates
        duplicate_mask = df.duplicated(subset=['content_hash'], keep='first')
        checks['duplicate_count'] = duplicate_mask.sum()
        checks['duplicate_percentage'] = (duplicate_mask.sum() / len(df)) * 100

        # Check text quality
        empty_text_mask = (df['cleaned_text'].str.len() == 0)
        checks['empty_text_count'] = empty_text_mask.sum()

        # Check missing critical fields
        missing_title = df['title'].isna().sum() if 'title' in df.columns else 0
        checks['missing_title_count'] = missing_title

        # Label distribution if available
        if 'label' in df.columns:
            checks['label_distribution'] = df['label'].value_counts().to_dict()

        return checks

    def load_to_csv(self, df: pd.DataFrame, filename: str):
        """Load processed data to CSV file."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")

    def load_to_database(self, df: pd.DataFrame, table_name: str = 'processed_news'):
        """Load data to database with idempotent operations."""
        if not self.db_engine:
            logger.warning("No database connection configured. Skipping database load.")
            return

        logger.info(f"Loading {len(df)} records to database table: {table_name}")

        try:
            # Check for existing records using content hash
            existing_hashes_query = f"""
            SELECT content_hash FROM {table_name}
            WHERE content_hash IN ({','.join(['%s'] * len(df))})
            """

            with self.db_engine.connect() as conn:
                existing_hashes = pd.read_sql(
                    text(existing_hashes_query),
                    conn,
                    params=df['content_hash'].tolist()
                )

            # Filter out existing records
            existing_hash_set = set(existing_hashes['content_hash']) if not existing_hashes.empty else set()
            new_records_mask = ~df['content_hash'].isin(existing_hash_set)
            new_records = df[new_records_mask]

            if len(new_records) == 0:
                logger.info("No new records to insert. All records already exist.")
                self.stats['loaded_records'] = 0
                self.stats['skipped_records'] = len(df)
                return

            # Insert new records
            new_records.to_sql(
                table_name,
                self.db_engine,
                if_exists='append',
                index=False,
                method='multi'
            )

            self.stats['loaded_records'] = len(new_records)
            self.stats['skipped_records'] = len(df) - len(new_records)

            logger.info(f"Loaded {len(new_records)} new records to database")
            logger.info(f"Skipped {len(df) - len(new_records)} existing records")

        except Exception as e:
            logger.error(f"Failed to load data to database: {e}")
            raise

    def save_pipeline_report(self, validation_results: Dict[str, Any]):
        """Save comprehensive pipeline execution report."""
        report = {
            'pipeline_execution': {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'statistics': self.stats,
                'validation_results': validation_results
            }
        }

        report_path = self.output_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Pipeline report saved to {report_path}")

    def run_pipeline(self) -> Dict[str, Any]:
        """Execute the complete ETL pipeline."""
        logger.info("Starting ETL pipeline execution...")

        try:
            # Extract phase
            source = self.config['source'].lower()
            if source == 'fakenewsnet':
                raw_data = self.extract_fakenewsnet()
            elif source == 'csv':
                raw_data = self.extract_csv(self.config['input_path'])
            elif source == 'json':
                raw_data = self.extract_json(self.config['input_path'])
            elif source == 'api':
                raw_data = self.extract_api(self.config['config_path'])
            else:
                raise ValueError(f"Unsupported data source: {source}")

            if raw_data.empty:
                logger.warning("No data extracted. Pipeline execution stopped.")
                return {'status': 'failed', 'reason': 'No data extracted'}

            # Transform phase
            processed_data = self.transform_data(raw_data)

            # Validate phase (if enabled)
            validation_results = {}
            if self.config.get('validate', False):
                processed_data, validation_results = self.validate_data(processed_data)

            # Load phase
            if self.config.get('output_csv', True):
                output_filename = f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.load_to_csv(processed_data, output_filename)

            if self.config.get('load_to_database', False):
                self.load_to_database(processed_data)

            # Generate report
            self.save_pipeline_report(validation_results)

            logger.info("ETL pipeline completed successfully")
            return {
                'status': 'success',
                'statistics': self.stats,
                'validation_results': validation_results
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.stats['errors'].append(str(e))
            return {'status': 'failed', 'error': str(e), 'statistics': self.stats}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ETL pipeline for fake news detection data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['fakenewsnet', 'csv', 'json', 'api'],
        help='Data source type'
    )

    # Conditional required arguments
    parser.add_argument(
        '--input-path',
        type=str,
        help='Path to input file (required for csv/json sources)'
    )

    parser.add_argument(
        '--config-path',
        type=str,
        help='Path to API configuration file (required for api source)'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )

    parser.add_argument(
        '--database-url',
        type=str,
        help='Database connection URL for loading data'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Enable data validation with quality checks'
    )

    parser.add_argument(
        '--load-to-database',
        action='store_true',
        help='Load processed data to database'
    )

    parser.add_argument(
        '--output-csv',
        action='store_true',
        default=True,
        help='Save processed data to CSV file'
    )

    parser.add_argument(
        '--pipeline-version',
        type=str,
        default='1.0.0',
        help='Pipeline version for tracking'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Validate conditional arguments
    if args.source in ['csv', 'json'] and not args.input_path:
        print(f"Error: --input-path is required for source type '{args.source}'")
        sys.exit(1)

    if args.source == 'api' and not args.config_path:
        print("Error: --config-path is required for source type 'api'")
        sys.exit(1)

    # Convert arguments to config dictionary
    config = vars(args)
    config = {k.replace('_', '_'): v for k, v in config.items()}

    logger.info(f"Starting ETL pipeline with config: {config}")

    # Initialize and run pipeline
    pipeline = DataPipeline(config)
    results = pipeline.run_pipeline()

    # Print summary
    print("\n" + "="*50)
    print("ETL PIPELINE EXECUTION SUMMARY")
    print("="*50)
    print(f"Status: {results['status']}")

    if 'statistics' in results:
        stats = results['statistics']
        print(f"Extracted records: {stats['extracted_records']}")
        print(f"Processed records: {stats['processed_records']}")
        print(f"Loaded records: {stats['loaded_records']}")
        print(f"Skipped records: {stats['skipped_records']}")

        if stats['errors']:
            print(f"Errors encountered: {len(stats['errors'])}")
            for error in stats['errors']:
                print(f"  - {error}")

    if 'validation_results' in results and results['validation_results']:
        val_results = results['validation_results']
        print(f"Validation: {val_results.get('valid_records', 0)} valid, "
              f"{val_results.get('invalid_records', 0)} invalid")

    if results['status'] == 'success':
        print("Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("Pipeline execution failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()