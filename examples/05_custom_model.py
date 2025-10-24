"""Example 05: Custom Model

Demonstrates how to create a custom surrogate model:
- Define a class extending BaseModel
- Implement train() and predict() methods
- Use it directly in config (no separate module needed)
- This example implements a simple Gaussian Process-like model
"""
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.model import BaseModel
from mpbax.core.algorithm import GreedySampling


class SimpleGPModel(BaseModel):
    """Simple Gaussian Process-like surrogate model.

    Uses radial basis function (RBF) kernel for predictions.
    This is a simplified version for demonstration - not a full GP!
    """

    def __init__(self, input_dim: int, length_scale: float = 0.1):
        """Initialize model.

        Args:
            input_dim: Dimensionality of input space
            length_scale: RBF kernel length scale
        """
        super().__init__(input_dim)
        self.length_scale = length_scale
        self.X_train = None
        self.Y_train = None

    def train(self, X: np.ndarray, Y: np.ndarray, metadata: dict = None):
        """Store training data (lazy learning).

        Args:
            X: Training inputs of shape (n, d)
            Y: Training outputs of shape (n, k)
            metadata: Optional metadata (not used here)
        """
        self.X_train = X.copy()
        self.Y_train = Y.copy()
        print(f"  SimpleGPModel: Stored {X.shape[0]} training samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using RBF kernel weighted average.

        Args:
            X: Test inputs of shape (n, d)

        Returns:
            Y: Predicted outputs of shape (n, k)
        """
        if self.X_train is None:
            # Not trained yet - return zeros
            return np.zeros((X.shape[0], self.Y_train.shape[1] if self.Y_train is not None else 1))

        # Compute RBF kernel weights
        predictions = []
        for x in X:
            # Distance to all training points
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

            # RBF kernel
            weights = np.exp(-(distances**2) / (2 * self.length_scale**2))

            # Normalize weights
            weights = weights / (np.sum(weights) + 1e-10)

            # Weighted average of training outputs
            y_pred = np.sum(weights[:, np.newaxis] * self.Y_train, axis=0)
            predictions.append(y_pred)

        return np.array(predictions)


def noisy_quadratic(X: np.ndarray) -> np.ndarray:
    """Quadratic function with noise.

    Args:
        X: Input array of shape (n, d)

    Returns:
        Y: Output array of shape (n, 1) with noise
    """
    noise = np.random.randn(X.shape[0], 1) * 0.1
    return np.sum(X**2, axis=1, keepdims=True) + noise


# Configuration using custom model
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {
        'dir': 'checkpoints_05',
        'freq': 1
    },
    'oracles': [{
        'name': 'noisy_quadratic',
        'input_dim': 2,
        'n_initial': 20,
        'function': {'class': noisy_quadratic},
        'model': {
            'class': SimpleGPModel,  # Custom model class!
            'params': {'length_scale': 0.2}  # Custom parameter
        }
    }],
    'algorithm': {
        'class': GreedySampling,
        'params': {
            'input_dims': [2],
            'n_propose': 10,
            'n_candidates': 500
        }
    }
}


if __name__ == '__main__':
    print("="*60)
    print("Example 05: Custom Model")
    print("="*60)
    print("\nThis example demonstrates:")
    print("  • Defining custom BaseModel subclass")
    print("  • Implementing train() and predict() methods")
    print("  • Passing custom model directly in config")
    print("\nModel: SimpleGPModel (RBF kernel-based predictions)")
    print("Oracle: Noisy quadratic function")
    print("\n" + "="*60 + "\n")

    engine = Engine(config)
    engine.run()

    print("\n" + "="*60)
    print("Custom model example complete!")
    print("="*60)
    print("\nKey takeaways:")
    print("  • Custom models must extend BaseModel")
    print("  • Implement train(X, Y, metadata) and predict(X)")
    print("  • Can define model in same file as config")
    print("  • No need for separate module!")
    print("="*60)
