"""Example 01: Basic Single-Oracle Optimization

Demonstrates the simplest mpBAX setup:
- Single oracle function (quadratic minimization)
- DummyModel for simplicity
- GreedySampling algorithm
- Everything defined in one file
"""
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


def quadratic(X: np.ndarray) -> np.ndarray:
    """Simple quadratic function to minimize.

    Args:
        X: Input array of shape (n, d)

    Returns:
        Y: Output array of shape (n, 1) with squared distances from origin
    """
    return np.sum(X**2, axis=1, keepdims=True)


# Configuration
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {
        'dir': 'checkpoints_01',
        'freq': 1  # Save every loop
    },
    'oracles': [{
        'name': 'quadratic',
        'input_dim': 3,
        'n_initial': 20,
        'function': {'class': quadratic},
        'model': {'class': DummyModel}
    }],
    'algorithm': {
        'class': GreedySampling,
        'params': {
            'input_dims': [3],
            'n_propose': 10,
            'n_candidates': 500
        }
    }
}


if __name__ == '__main__':
    print("="*60)
    print("Example 01: Basic Single-Oracle Optimization")
    print("="*60)
    print("\nMinimizing quadratic function: f(x) = sum(x^2)")
    print("Search space: [0, 1]^3")
    print("\n" + "="*60 + "\n")

    engine = Engine(config)
    engine.run()

    print("\n" + "="*60)
    print("Optimization complete!")
    print("Check 'checkpoints_01/' for saved data and models")
    print("="*60)
