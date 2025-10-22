"""Model interface for surrogate models."""

import numpy as np
import pickle
from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for surrogate models.

    All models must implement train() and predict() methods.
    The predict() method signature must match the oracle function: X (n, d) -> Y (n, 1)
    """

    def __init__(self, input_dim: int):
        """Initialize model.

        Args:
            input_dim: Input dimensionality d
        """
        self.input_dim = input_dim
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train the model on data.

        Args:
            X: Input data with shape (n, d)
            Y: Output data with shape (n, 1)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for inputs X.

        Must have same signature as oracle function.

        Args:
            X: Input data with shape (n, d)

        Returns:
            Y: Predicted outputs with shape (n, 1)
        """
        pass

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'BaseModel':
        """Load model from disk.

        Args:
            filepath: Path to load the model from

        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        return model


class DummyModel(BaseModel):
    """Simple model that predicts the mean of training Y values.

    Useful for testing and as a baseline.
    """

    def __init__(self, input_dim: int):
        """Initialize DummyModel.

        Args:
            input_dim: Input dimensionality d
        """
        super().__init__(input_dim)
        self.mean_y = None

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train by computing mean of Y.

        Args:
            X: Input data with shape (n, d)
            Y: Output data with shape (n, 1)
        """
        self._validate_data(X, Y)
        self.mean_y = np.mean(Y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean value for all inputs.

        Args:
            X: Input data with shape (n, d)

        Returns:
            Y: Predicted outputs with shape (n, 1), all equal to mean_y

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self._validate_input(X)

        n = X.shape[0]
        return np.full((n, 1), self.mean_y)

    def _validate_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Validate training data shapes.

        Args:
            X: Input data
            Y: Output data

        Raises:
            ValueError: If shapes are invalid
        """
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n, {self.input_dim}), got {X.shape}")

        if Y.ndim != 2 or Y.shape[1] != 1:
            raise ValueError(f"Y must have shape (n, 1), got {Y.shape}")

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same number of samples")

    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input shape for prediction.

        Args:
            X: Input data

        Raises:
            ValueError: If shape is invalid
        """
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must have shape (n, {self.input_dim}), got {X.shape}")
