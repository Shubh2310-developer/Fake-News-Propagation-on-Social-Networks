# backend/app/utils/data_preprocessing.py

import re
import unicodedata
from typing import List


class TextProcessor:
    """Utility for cleaning and normalizing text."""

    def clean(self, text: str) -> str:
        text = self.remove_html_tags(text)
        text = self.normalize_unicode(text)
        text = self.remove_urls(text)
        text = self.remove_special_chars(text)
        return text.strip().lower()

    def remove_html_tags(self, text: str) -> str:
        return re.sub(r"<.*?>", " ", text)

    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http\S+|www\.\S+", "", text)

    def remove_special_chars(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)


def normalize_text(texts: List[str]) -> List[str]:
    processor = TextProcessor()
    return [processor.clean(t) for t in texts]