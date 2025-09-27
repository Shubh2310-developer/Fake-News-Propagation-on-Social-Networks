"""
Utility package for backend.
Contains helper modules for preprocessing, validation,
file handling, and visualization.
"""

from .data_preprocessing import TextProcessor, normalize_text
from .validators import validate_text_input, validate_dataset
from .file_handlers import save_file, load_file, delete_file
from .visualization import plot_confusion_matrix, plot_training_curves