"""Evaluator wrapper for oracle functions."""

import numpy as np
from typing import Callable


class Evaluator:
    """Wrapper for user-provided oracle functions (expensive simulations).

    Validates input/output shapes and tracks evaluation counts.

    Conventions:
    - fn_oracle has signature: X (n, d) -> Y (n, 1)
    """

    def __init__(self, fn_oracle: Callable[[np.ndarray], np.ndarray], input_dim: int, name: str = "objective"):
        """Initialize Evaluator.

        Args:
            fn_oracle: Black box simulation function with signature X -> Y
            input_dim: Expected input dimensionality d
            name: Name of this objective (for logging/debugging)
        """
        self.fn_oracle = fn_oracle
        self.input_dim = input_dim
        self.name = name
        self.eval_count = 0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate oracle function on inputs X.

        Args:
            X: Input array with shape (n, d)

        Returns:
            Y: Output array with shape (n, 1)

        Raises:
            ValueError: If input or output shapes are invalid
        """
        # Validate input shape
        self._validate_input(X)

        # Call oracle function
        Y = self.fn_oracle(X)

        # Validate output shape
        self._validate_output(Y, X.shape[0])

        # Update evaluation count
        self.eval_count += X.shape[0]

        return Y

    def get_eval_count(self) -> int:
        """Get total number of evaluations performed.

        Returns:
            Number of samples evaluated
        """
        return self.eval_count

    def reset_count(self) -> None:
        """Reset evaluation counter to zero."""
        self.eval_count = 0

    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input array shape.

        Args:
            X: Input array

        Raises:
            ValueError: If shape is invalid
        """
        if X.ndim != 2:
            raise ValueError(
                f"[{self.name}] Input X must be 2D array with shape (n, d), got shape {X.shape}"
            )

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"[{self.name}] Input dimension mismatch: expected d={self.input_dim}, got d={X.shape[1]}"
            )

    def _validate_output(self, Y: np.ndarray, n_samples: int) -> None:
        """Validate output array shape.

        Args:
            Y: Output array
            n_samples: Expected number of samples

        Raises:
            ValueError: If shape is invalid
        """
        if Y.ndim != 2:
            raise ValueError(
                f"[{self.name}] Output Y must be 2D array with shape (n, 1), got shape {Y.shape}"
            )

        if Y.shape[1] != 1:
            raise ValueError(
                f"[{self.name}] Output Y must have shape (n, 1), got shape {Y.shape}"
            )

        if Y.shape[0] != n_samples:
            raise ValueError(
                f"[{self.name}] Output sample count mismatch: expected {n_samples}, got {Y.shape[0]}"
            )
