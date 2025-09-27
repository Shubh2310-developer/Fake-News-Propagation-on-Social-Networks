# backend/ml_models/evaluation/cross_validation.py

import numpy as np
from typing import Generator, Tuple


class StratifiedTimeSeriesSplit:
    """
    Custom cross-validator that respects temporal order.
    The training set expands progressively in each split.
    """

    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, X, y, timestamps: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test sets."""
        sorted_indices = np.argsort(timestamps)
        n_samples = len(X)
        test_size_abs = int(n_samples * self.test_size)

        if self.n_splits * test_size_abs > n_samples:
            raise ValueError("n_splits * test_size exceeds number of samples")

        for i in range(self.n_splits):
            train_end = n_samples - (self.n_splits - i) * test_size_abs
            test_start = train_end
            test_end = test_start + test_size_abs

            yield sorted_indices[:train_end], sorted_indices[test_start:test_end]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits