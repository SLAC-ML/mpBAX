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
            Y: Output data with shape (n, k) where k >= 1
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for inputs X.

        Must have same signature as oracle function.

        Args:
            X: Input data with shape (n, d)

        Returns:
            Y: Predicted outputs with shape (n, k) where k >= 1
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

    def get_state(self) -> dict:
        """Get model state for persistence.

        This method allows models to expose internal state (e.g., normalization
        parameters, hyperparameters) for inspection, debugging, or manual transfer.

        For finetuning workflows, state persistence happens automatically via:
        1. Python object reuse (same model instance across loops)
        2. Pickle serialization (checkpoint save/load)

        This method is optional and useful for:
        - Inspecting normalization params (e.g., X_mu, X_sigma)
        - Debugging model state
        - Manual state transfer between models
        - Non-pickle serialization formats

        Returns:
            Dictionary containing model state. Default: empty dict.

        Example:
            >>> state = model.get_state()
            >>> print(state)
            {'X_mu': array([0.5, 0.3]), 'X_sigma': array([0.2, 0.1])}
        """
        return {}

    def set_state(self, state_dict: dict) -> None:
        """Restore model state from dictionary.

        Args:
            state_dict: Dictionary containing model state

        Example:
            >>> state = {'X_mu': np.array([0.5, 0.3])}
            >>> model.set_state(state)
        """
        pass

    def get_best_model_snapshot(self) -> 'BaseModel':
        """Get best model snapshot from training.

        Some models track the "best" model during training (e.g., based on
        validation loss, early stopping). This method allows retrieving that
        snapshot for checkpointing.

        Used by checkpoint_mode='best' or 'both' in config.

        Returns:
            Best model instance if tracked, otherwise None.

        Example:
            >>> model.train(X, Y)  # Training tracks best model internally
            >>> best = model.get_best_model_snapshot()
            >>> if best:
            >>>     best.save('model_best.pkl')
        """
        return None


class DummyModel(BaseModel):
    """Simple model that predicts the mean of training Y values.

    For multi-output Y, computes mean per output dimension independently.
    Useful for testing and as a baseline.
    """

    def __init__(self, input_dim: int):
        """Initialize DummyModel.

        Args:
            input_dim: Input dimensionality d
        """
        super().__init__(input_dim)
        self.mean_y = None  # Will have shape (k,) for k output dimensions

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Train by computing mean of Y per output dimension.

        Args:
            X: Input data with shape (n, d)
            Y: Output data with shape (n, k) where k >= 1
        """
        self._validate_data(X, Y)
        # Compute mean per output dimension, result has shape (k,)
        self.mean_y = np.mean(Y, axis=0)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean value for all inputs.

        Args:
            X: Input data with shape (n, d)

        Returns:
            Y: Predicted outputs with shape (n, k), all rows equal to mean_y

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self._validate_input(X)

        n = X.shape[0]
        # Broadcast mean_y to (n, k)
        return np.tile(self.mean_y, (n, 1))

    def get_state(self) -> dict:
        """Get DummyModel state.

        Returns:
            Dictionary with mean_y if trained, empty dict otherwise
        """
        if self.is_trained:
            return {'mean_y': self.mean_y}
        return {}

    def set_state(self, state_dict: dict) -> None:
        """Restore DummyModel state.

        Args:
            state_dict: Dictionary containing 'mean_y'
        """
        if 'mean_y' in state_dict:
            self.mean_y = state_dict['mean_y']
            self.is_trained = True

    def get_best_model_snapshot(self) -> 'BaseModel':
        """DummyModel doesn't track best model.

        Returns:
            None (not applicable for DummyModel)
        """
        return None

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

        if Y.ndim != 2 or Y.shape[1] < 1:
            raise ValueError(f"Y must have shape (n, k) where k >= 1, got {Y.shape}")

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
