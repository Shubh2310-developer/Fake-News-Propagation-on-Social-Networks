# backend/app/utils/validators.py

from typing import List, Dict
from fastapi import HTTPException


def validate_text_input(text: str, min_length: int = 5) -> None:
    if not text or len(text.strip()) < min_length:
        raise HTTPException(status_code=400, detail="Input text too short")


def validate_dataset(dataset: List[Dict], required_keys=("text", "label")) -> None:
    if not dataset:
        raise HTTPException(status_code=400, detail="Dataset is empty")
    for idx, item in enumerate(dataset):
        for key in required_keys:
            if key not in item:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing key '{key}' in dataset item at index {idx}"
                )