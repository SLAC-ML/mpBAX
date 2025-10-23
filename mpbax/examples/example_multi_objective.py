"""
Multi-objective optimization example.

This example demonstrates how to use mpBAX for multi-objective optimization with
two independent oracles that have different input dimensions:

Oracle 1 (2D): Minimize (x1 - 0.3)^2 + (x2 - 0.7)^2
Oracle 2 (3D): Minimize x1^2 + x2^2 + x3^2

Each oracle operates on its own input space independently.
"""

import numpy as np
import yaml
from pathlib import Path

from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel


# Define oracle functions
def oracle_obj1(X: np.ndarray) -> np.ndarray:
    """Oracle for objective 1: quadratic with minimum at (0.3, 0.7).

    Args:
        X: Input array with shape (n, 2)

    Returns:
        Y: Output array with shape (n, 1)
    """
    Y = (X[:, 0] - 0.3)**2 + (X[:, 1] - 0.7)**2
    return Y.reshape(-1, 1)


def oracle_obj2(X: np.ndarray) -> np.ndarray:
    """Oracle for objective 2: sum of squares (3D).

    Args:
        X: Input array with shape (n, 3)

    Returns:
        Y: Output array with shape (n, 1)
    """
    Y = np.sum(X**2, axis=1)
    return Y.reshape(-1, 1)


def main():
    """Run multi-objective optimization."""

    # Create config file
    config = {
        'seed': 42,
        'max_loops': 8,
        'n_initial': 15,
        'checkpoint': {
            'dir': 'checkpoints_multi_obj',
            'freq': 1,
            'resume_from': None  # Set to 'latest' to resume
        },
        'oracles': [
            {
                'name': 'obj1_2d',
                'input_dim': 2
            },
            {
                'name': 'obj2_3d',
                'input_dim': 3
            }
        ],
        'algorithm': {
            'class': 'GreedySampling',
            'params': {
                'input_dims': [2, 3],  # Must match oracle dimensions
                'n_propose': 8,
                'n_candidates': 500
            }
        }
    }

    # Save config
    config_path = 'config_multi_obj.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("=" * 60)
    print("Multi-Objective Optimization with mpBAX")
    print("=" * 60)
    print("Oracle 1 (2D): Minimize (x1-0.3)^2 + (x2-0.7)^2")
    print("  Global minimum: (0.3, 0.7) with f = 0")
    print("\nOracle 2 (3D): Minimize x1^2 + x2^2 + x3^2")
    print("  Global minimum: (0, 0, 0) with f = 0")
    print("=" * 60)

    # Create and run engine (algorithm auto-instantiated from config)
    engine = Engine(
        config=config,
        fn_oracles=[oracle_obj1, oracle_obj2],
        model_class=DummyModel,
        algorithm=None  # Auto-instantiate from config
    )

    engine.run()

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)

    # Load final data to find best solutions for each oracle
    from mpbax.core.checkpoint import CheckpointManager
    checkpoint_manager = CheckpointManager(config['checkpoint']['dir'])
    loop, data_handlers, _, _, oracle_names = checkpoint_manager.load_checkpoint()

    # Analyze each oracle
    for i, (dh, oracle_name) in enumerate(zip(data_handlers, oracle_names)):
        X_all, Y_all = dh.get_data()

        # Find best solution
        best_idx = np.argmin(Y_all)
        best_x = X_all[best_idx]
        best_y = Y_all[best_idx]

        print(f"\n{oracle_name}:")
        print(f"  Total evaluations: {engine.evaluators[i].get_eval_count()}")
        print(f"  Best solution: x = {best_x}")
        print(f"  Best value: f(x) = {best_y[0]:.6f}")

        # Compute distance to true optimum
        if i == 0:
            true_opt = np.array([0.3, 0.7])
            dist = np.linalg.norm(best_x - true_opt)
            print(f"  True optimum: x = [0.3000, 0.7000], f(x) = 0.000000")
        else:
            true_opt = np.array([0.0, 0.0, 0.0])
            dist = np.linalg.norm(best_x - true_opt)
            print(f"  True optimum: x = [0.0000, 0.0000, 0.0000], f(x) = 0.000000")

        print(f"  Distance to optimum: {dist:.6f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
