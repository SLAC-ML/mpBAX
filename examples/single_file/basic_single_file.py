"""Basic single-file example using instance-based config.

This demonstrates mpBAX's ability to define everything in one file
without needing separate modules. Perfect for:
- Quick prototyping
- Simple examples
- Jupyter notebooks
- Educational purposes
"""
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


# Define oracle function directly in this file
def quadratic_oracle(X: np.ndarray) -> np.ndarray:
    """Simple quadratic oracle: sum of squares.

    Args:
        X: Array of shape (n, d) with inputs in [0, 1]^d

    Returns:
        Y: Array of shape (n, 1) with objectives (minimization)
    """
    return np.sum(X**2, axis=1, keepdims=True)


# Create config using function and class instances
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {
        'dir': 'checkpoints_single_file',
        'freq': 1
    },
    'oracles': [
        {
            'name': 'quadratic',
            'input_dim': 3,
            'n_initial': 20,
            'function': {
                'class': quadratic_oracle,  # Pass function directly!
            },
            'model': {
                'class': DummyModel,  # Pass class directly!
                'params': {}
            }
        }
    ],
    'algorithm': {
        'class': GreedySampling,  # Pass class directly!
        'params': {
            'input_dims': [3],
            'n_propose': 10,
            'n_candidates': 500
        }
    }
}


if __name__ == '__main__':
    print("=" * 60)
    print("Single-File mpBAX Example")
    print("=" * 60)
    print("\nThis example demonstrates instance-based config where:")
    print("  • Oracle function defined in same file")
    print("  • Model and algorithm classes imported and passed directly")
    print("  • No separate modules needed!")
    print("\n" + "=" * 60)

    # Run optimization
    engine = Engine(config)
    engine.run()

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
