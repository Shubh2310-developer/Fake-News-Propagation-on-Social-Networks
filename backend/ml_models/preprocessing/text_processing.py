# backend/ml_models/preprocessing/text_processing.py

import re
import string
import unicodedata
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Handles all text cleaning and normalization tasks.
    Provides a configurable pipeline for processing raw text into clean,
    standardized format suitable for feature extraction or model input.
    """

    def __init__(self,
                 lowercase: bool = True,
                 remove_urls: bool = True,
                 remove_html: bool = True,
                 remove_punctuation: bool = False,
                 remove_extra_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_numbers: bool = False,
                 preserve_important_chars: bool = True):
        """
        Initialize the TextProcessor with configurable options.

        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove HTTP(S) URLs from text
            remove_html: Remove HTML tags from text
            remove_punctuation: Remove punctuation marks
            remove_extra_whitespace: Remove extra spaces and normalize whitespace
            normalize_unicode: Normalize unicode characters
            remove_numbers: Remove numeric characters
            preserve_important_chars: Keep important punctuation (.,!?)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_numbers = remove_numbers
        self.preserve_important_chars = preserve_important_chars

        # Compile regex patterns for efficiency
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._html_pattern = re.compile(r'<[^<]+?>')
        self._extra_whitespace_pattern = re.compile(r'\s+')
        self._number_pattern = re.compile(r'\d+')

    def clean(self, text: str) -> str:
        """
        Applies the full text cleaning pipeline to a single string.

        Args:
            text: Raw input text to clean

        Returns:
            Cleaned and normalized text
        """
        if not isinstance(text, str):
            logger.warning(f"Expected string input, got {type(text)}. Converting to string.")
            text = str(text)

        if not text or not text.strip():
            return ""

        # Apply cleaning steps in order
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        if self.remove_html:
            text = self._remove_html_tags(text)

        if self.remove_urls:
            text = self._remove_urls(text)

        if self.lowercase:
            text = self._lowercase_text(text)

        if self.remove_numbers:
            text = self._remove_numbers(text)

        if self.remove_punctuation:
            text = self._remove_punctuation(text)

        if self.remove_extra_whitespace:
            text = self._remove_extra_whitespace(text)

        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts efficiently.

        Args:
            texts: List of raw text strings

        Returns:
            List of cleaned text strings
        """
        return [self.clean(text) for text in texts]

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters using NFKC normalization."""
        return unicodedata.normalize('NFKC', text)

    def _remove_urls(self, text: str) -> str:
        """Remove HTTP(S) URLs and www links from text."""
        return self._url_pattern.sub(' ', text)

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        return self._html_pattern.sub(' ', text)

    def _lowercase_text(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation marks from text.
        Optionally preserves important punctuation for sentence structure.
        """
        if self.preserve_important_chars:
            # Keep important punctuation that affects meaning
            important_chars = '.!?'
            translator = str.maketrans('', '', ''.join(
                c for c in string.punctuation if c not in important_chars
            ))
        else:
            translator = str.maketrans('', '', string.punctuation)

        return text.translate(translator)

    def _remove_numbers(self, text: str) -> str:
        """Remove numeric characters from text."""
        return self._number_pattern.sub(' ', text)

    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing."""
        return self._extra_whitespace_pattern.sub(' ', text)

    def get_config(self) -> dict:
        """
        Get the current configuration of the text processor.

        Returns:
            Dictionary containing current configuration settings
        """
        return {
            'lowercase': self.lowercase,
            'remove_urls': self.remove_urls,
            'remove_html': self.remove_html,
            'remove_punctuation': self.remove_punctuation,
            'remove_extra_whitespace': self.remove_extra_whitespace,
            'normalize_unicode': self.normalize_unicode,
            'remove_numbers': self.remove_numbers,
            'preserve_important_chars': self.preserve_important_chars
        }


def create_news_processor() -> TextProcessor:
    """
    Create a TextProcessor configured specifically for news article processing.

    Returns:
        Configured TextProcessor for news articles
    """
    return TextProcessor(
        lowercase=True,
        remove_urls=True,
        remove_html=True,
        remove_punctuation=False,  # Keep punctuation for news articles
        remove_extra_whitespace=True,
        normalize_unicode=True,
        remove_numbers=False,  # Keep numbers in news articles
        preserve_important_chars=True
    )


def create_social_media_processor() -> TextProcessor:
    """
    Create a TextProcessor configured for social media text processing.

    Returns:
        Configured TextProcessor for social media content
    """
    return TextProcessor(
        lowercase=True,
        remove_urls=True,
        remove_html=True,
        remove_punctuation=True,  # Remove most punctuation from social media
        remove_extra_whitespace=True,
        normalize_unicode=True,
        remove_numbers=True,  # Remove numbers from social media
        preserve_important_chars=False
    )