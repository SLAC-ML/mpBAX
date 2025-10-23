"""
Multi-output oracle example.

This example demonstrates a single oracle that outputs multiple objectives simultaneously.
The oracle returns Y with shape (n, 3), where each column represents a different metric.

Oracle (2D input, 3 outputs):
  - Output 1: Sum of squares
  - Output 2: Sum
  - Output 3: Maximum value
"""

import numpy as np
import yaml

from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel


def oracle_multi_output(X: np.ndarray) -> np.ndarray:
    """Oracle with multiple outputs.

    Args:
        X: Input array with shape (n, 2)

    Returns:
        Y: Output array with shape (n, 3)
            - Column 0: Sum of squares
            - Column 1: Sum
            - Column 2: Maximum value
    """
    sum_squares = np.sum(X**2, axis=1, keepdims=True)
    sum_vals = np.sum(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)

    Y = np.hstack([sum_squares, sum_vals, max_vals])
    return Y


def main():
    """Run optimization with multi-output oracle."""

    # Create config file
    config = {
        'seed': 42,
        'max_loops': 5,
        'n_initial': 10,
        'checkpoint': {
            'dir': 'checkpoints_multi_output',
            'freq': 1,
            'resume_from': None
        },
        'oracles': [
            {
                'name': 'multi_output_oracle',
                'input_dim': 2
            }
        ],
        'algorithm': {
            'class': 'GreedySampling',
            'params': {
                'input_dims': [2],
                'n_propose': 5,
                'n_candidates': 500
            }
        }
    }

    # Save config
    config_path = 'config_multi_output.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("=" * 60)
    print("Multi-Output Oracle Demonstration")
    print("=" * 60)
    print("Single oracle with 3 outputs:")
    print("  Output 1: Sum of squares (minimize)")
    print("  Output 2: Sum (varies)")
    print("  Output 3: Maximum value (varies)")
    print("=" * 60)

    # Create and run engine
    engine = Engine(
        config_path=config_path,
        fn_oracles=[oracle_multi_output],
        model_class=DummyModel,
        algorithm=None  # Auto-instantiate from config
    )

    engine.run()

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)

    # Load and analyze results
    from mpbax.core.checkpoint import CheckpointManager
    checkpoint_manager = CheckpointManager(config['checkpoint']['dir'])
    loop, data_handlers, _, _, _ = checkpoint_manager.load_checkpoint()

    X_all, Y_all = data_handlers[0].get_data()

    print(f"\nData collected:")
    print(f"  X shape: {X_all.shape}")
    print(f"  Y shape: {Y_all.shape}")  # Should be (n, 3)
    print(f"  Total evaluations: {engine.evaluators[0].get_eval_count()}")

    # Analyze each output dimension
    print(f"\nOutput statistics:")
    for i in range(Y_all.shape[1]):
        min_val = np.min(Y_all[:, i])
        max_val = np.max(Y_all[:, i])
        mean_val = np.mean(Y_all[:, i])
        print(f"  Output {i+1}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

    # Find best solution for first output (sum of squares)
    best_idx = np.argmin(Y_all[:, 0])
    best_x = X_all[best_idx]
    best_y = Y_all[best_idx]

    print(f"\nBest solution (minimizing output 1):")
    print(f"  x = [{best_x[0]:.4f}, {best_x[1]:.4f}]")
    print(f"  y = [{best_y[0]:.4f}, {best_y[1]:.4f}, {best_y[2]:.4f}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
