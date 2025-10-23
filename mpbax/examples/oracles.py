"""Oracle functions for mpBAX examples."""

import numpy as np


def oracle_quadratic(X: np.ndarray) -> np.ndarray:
    """Quadratic oracle function with minimum at (0.3, 0.7).

    Args:
        X: Input array with shape (n, 2)

    Returns:
        Y: Output array with shape (n, 1)
    """
    # f(x1, x2) = (x1 - 0.3)^2 + (x2 - 0.7)^2
    Y = (X[:, 0] - 0.3)**2 + (X[:, 1] - 0.7)**2
    return Y.reshape(-1, 1)


def oracle_quadratic_4d(X: np.ndarray) -> np.ndarray:
    """4D quadratic oracle function.

    Args:
        X: Input array with shape (n, 4)

    Returns:
        Y: Output array with shape (n, 1)
    """
    Y = np.sum(X ** 2, axis=1, keepdims=True)
    return Y


def oracle_rosenbrock(X: np.ndarray) -> np.ndarray:
    """Rosenbrock function (banana function).

    Args:
        X: Input array with shape (n, 2)

    Returns:
        Y: Output array with shape (n, 1)
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    Y = (1 - x1)**2 + 100 * (x2 - x1**2)**2
    return Y.reshape(-1, 1)
