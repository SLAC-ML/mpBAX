"""Oracle functions for YAML example.

These functions must be importable via module path for YAML-based configs.
"""

import numpy as np


def quadratic(X: np.ndarray) -> np.ndarray:
    """Quadratic objective function.

    Minimize sum of squared distances from center point [0.5, 0.5].

    Args:
        X: Input array with shape (n, 2)

    Returns:
        Y: Output array with shape (n, 1)
    """
    return np.sum((X - 0.5)**2, axis=1, keepdims=True)


def sphere(X: np.ndarray) -> np.ndarray:
    """Sphere function (sum of squares).

    Args:
        X: Input array with shape (n, d)

    Returns:
        Y: Output array with shape (n, 1)
    """
    return np.sum(X**2, axis=1, keepdims=True)
