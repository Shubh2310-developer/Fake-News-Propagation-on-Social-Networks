# backend/app/utils/file_handlers.py

import os
import shutil
from typing import Optional
from fastapi import UploadFile, HTTPException


UPLOAD_DIR = "data/uploads"


def save_file(upload: UploadFile, subdir: Optional[str] = None) -> str:
    """Save uploaded file to disk and return path."""
    directory = os.path.join(UPLOAD_DIR, subdir) if subdir else UPLOAD_DIR
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, upload.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

    return file_path


def load_file(file_path: str) -> bytes:
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(file_path, "rb") as f:
        return f.read()


def delete_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)