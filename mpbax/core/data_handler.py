"""Data Handler for managing (X, Y) pairs with shape validation."""

import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple


class DataHandler:
    """Manages (X, Y) data pairs with strict shape validation.

    Conventions:
    - X has shape (n, d) where n = number of samples, d = dimensionality
    - Y has shape (n, k) where n = number of samples, k >= 1 output dimensions
    """

    def __init__(self, input_dim: int):
        """Initialize DataHandler.

        Args:
            input_dim: Expected dimensionality d of input X
        """
        self.input_dim = input_dim
        self.X = None
        self.Y = None
        self.loop_indices = []  # Track which loop each sample came from

    def add_data(self, X: np.ndarray, Y: np.ndarray, loop: Optional[int] = None) -> None:
        """Add new (X, Y) pairs to the dataset.

        Args:
            X: Input data with shape (n, d)
            Y: Output data with shape (n, k) where k >= 1
            loop: Optional loop number to track which loop this data came from

        Raises:
            ValueError: If shapes are invalid
        """
        # Validate shapes
        self._validate_shapes(X, Y)

        # Track loop indices if provided
        if loop is not None:
            loop_array = np.full(X.shape[0], loop, dtype=int)
            self.loop_indices.append(loop_array)

        # Add to existing data or initialize
        if self.X is None:
            self.X = X.copy()
            self.Y = Y.copy()
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])

    def get_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get all data.

        Returns:
            Tuple of (X, Y), both None if no data exists
        """
        if self.X is None:
            return None, None
        return self.X.copy(), self.Y.copy()

    def get_data_with_metadata(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
        """Get all data with metadata about loop indices.

        Returns:
            Tuple of (X, Y, metadata) where metadata is a dict containing:
                - 'loop_indices': array of loop numbers for each sample (if tracked)
        """
        X, Y = self.get_data()
        metadata = {}

        if self.loop_indices:
            metadata['loop_indices'] = np.concatenate(self.loop_indices)

        return X, Y, metadata

    def get_size(self) -> int:
        """Get number of data points.

        Returns:
            Number of samples, or 0 if no data
        """
        if self.X is None:
            return 0
        return self.X.shape[0]

    def save(self, filepath: str) -> None:
        """Save data to disk.

        Args:
            filepath: Path to save the data
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'X': self.X,
            'Y': self.Y,
            'input_dim': self.input_dim,
            'loop_indices': self.loop_indices
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'DataHandler':
        """Load data from disk.

        Args:
            filepath: Path to load the data from

        Returns:
            DataHandler instance with loaded data
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        handler = cls(input_dim=data['input_dim'])
        handler.X = data['X']
        handler.Y = data['Y']
        handler.loop_indices = data.get('loop_indices', [])  # Backward compatibility

        return handler

    def _validate_shapes(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Validate shapes of X and Y.

        Args:
            X: Input data
            Y: Output data

        Raises:
            ValueError: If shapes are invalid
        """
        # Check X shape
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array with shape (n, d), got shape {X.shape}")

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X dimension mismatch: expected d={self.input_dim}, got d={X.shape[1]}"
            )

        # Check Y shape
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D array with shape (n, k), got shape {Y.shape}")

        if Y.shape[1] < 1:
            raise ValueError(f"Y must have at least 1 output dimension, got shape {Y.shape}")

        # Check matching sample sizes
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have same number of samples: X has {X.shape[0]}, Y has {Y.shape[0]}"
            )
