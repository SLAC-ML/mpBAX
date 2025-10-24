"""Example 03: Multi-Output Oracle

Demonstrates an oracle that returns multiple outputs:
- Single oracle function returning Y with shape (n, k) where k > 1
- Model predicts all outputs simultaneously
- Useful for:
  - Vector-valued objectives
  - Simulations with multiple measurements
  - Multi-task learning scenarios
"""
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


def multi_output_oracle(X: np.ndarray) -> np.ndarray:
    """Oracle returning 3 outputs for each input.

    Computes three different metrics from the same input:
    1. L2 norm (distance from origin)
    2. L1 norm (Manhattan distance)
    3. Max absolute value

    Args:
        X: Input array of shape (n, d)

    Returns:
        Y: Output array of shape (n, 3) with three outputs per sample
    """
    l2_norm = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    l1_norm = np.sum(np.abs(X), axis=1, keepdims=True)
    max_abs = np.max(np.abs(X), axis=1, keepdims=True)

    return np.hstack([l2_norm, l1_norm, max_abs])


# Configuration
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {
        'dir': 'checkpoints_03',
        'freq': 1
    },
    'oracles': [{
        'name': 'multi_output',
        'input_dim': 4,
        'n_initial': 25,
        'function': {'class': multi_output_oracle},  # Returns shape (n, 3)
        'model': {'class': DummyModel}  # Model learns to predict all 3 outputs
    }],
    'algorithm': {
        'class': GreedySampling,
        'params': {
            'input_dims': [4],
            'n_propose': 10,
            'n_candidates': 500
        }
    }
}


if __name__ == '__main__':
    print("="*60)
    print("Example 03: Multi-Output Oracle")
    print("="*60)
    print("\nOracle computes 3 metrics from each input:")
    print("  1. L2 norm (Euclidean distance)")
    print("  2. L1 norm (Manhattan distance)")
    print("  3. Max absolute value")
    print("\nModel learns to predict all 3 outputs simultaneously")
    print("\n" + "="*60 + "\n")

    engine = Engine(config)
    engine.run()

    print("\n" + "="*60)
    print("Optimization complete!")
    print("\nNote: DummyModel returns mean of Y, so all 3 outputs are averaged")
    print("For real applications, use a model that preserves output structure")
    print("="*60)
