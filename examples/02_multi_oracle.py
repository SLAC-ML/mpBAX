"""Example 02: Multi-Oracle Optimization

Demonstrates optimization with multiple independent oracles:
- Two oracles with different input dimensions
- Each oracle has its own model and initial data
- Single algorithm proposes candidates for both oracles
- Useful for multi-objective or multi-fidelity optimization
"""
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


def sphere(X: np.ndarray) -> np.ndarray:
    """Sphere function: sum of squares.

    Args:
        X: Input array of shape (n, 2)

    Returns:
        Y: Output array of shape (n, 1)
    """
    return np.sum(X**2, axis=1, keepdims=True)


def rosenbrock(X: np.ndarray) -> np.ndarray:
    """Rosenbrock function: (1-x)^2 + 100(y-x^2)^2

    Args:
        X: Input array of shape (n, 2)

    Returns:
        Y: Output array of shape (n, 1)
    """
    x = X[:, 0]
    y = X[:, 1]
    result = (1 - x)**2 + 100 * (y - x**2)**2
    return result.reshape(-1, 1)


# Configuration with two oracles
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {
        'dir': 'checkpoints_02',
        'freq': 1
    },
    'oracles': [
        {
            'name': 'sphere',
            'input_dim': 2,
            'n_initial': 15,
            'function': {'class': sphere},
            'model': {'class': DummyModel}
        },
        {
            'name': 'rosenbrock',
            'input_dim': 2,
            'n_initial': 20,  # Can differ per oracle
            'function': {'class': rosenbrock},
            'model': {'class': DummyModel}
        }
    ],
    'algorithm': {
        'class': GreedySampling,
        'params': {
            'input_dims': [2, 2],  # List matching oracle dimensions
            'n_propose': 8,
            'n_candidates': 400
        }
    }
}


if __name__ == '__main__':
    print("="*60)
    print("Example 02: Multi-Oracle Optimization")
    print("="*60)
    print("\nOptimizing two functions simultaneously:")
    print("  1. Sphere:      f(x) = sum(x^2)")
    print("  2. Rosenbrock:  f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print("\nEach oracle:")
    print("  • Has its own model")
    print("  • Has independent initial samples")
    print("  • Receives candidates from same algorithm")
    print("\n" + "="*60 + "\n")

    engine = Engine(config)
    engine.run()

    print("\n" + "="*60)
    print("Optimization complete!")
    print("="*60)
