"""
Simple single-objective optimization example.

This example demonstrates how to use mpBAX to optimize a simple 2D function:
f(x1, x2) = (x1 - 0.3)^2 + (x2 - 0.7)^2

The global minimum is at (0.3, 0.7) with f = 0.
"""

import numpy as np
import yaml
from pathlib import Path

from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


# Define the oracle function (simulation)
def oracle_quadratic(X: np.ndarray) -> np.ndarray:
    """Oracle function: quadratic with minimum at (0.3, 0.7).

    Args:
        X: Input array with shape (n, 2)

    Returns:
        Y: Output array with shape (n, 1)
    """
    # f(x1, x2) = (x1 - 0.3)^2 + (x2 - 0.7)^2
    Y = (X[:, 0] - 0.3)**2 + (X[:, 1] - 0.7)**2
    return Y.reshape(-1, 1)


def main():
    """Run single-objective optimization."""

    # Create config file
    config = {
        'seed': 42,
        'max_loops': 10,
        'n_initial': 20,
        'checkpoint': {
            'dir': 'checkpoints_single_obj',
            'freq': 1,
            'resume_from': None  # Set to 'latest' to resume
        },
        'n_propose': 10,
        'objectives': [
            {
                'name': 'quadratic',
                'input_dim': 2
            }
        ]
    }

    # Save config
    config_path = 'config_single_obj.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print("=" * 60)
    print("Single-Objective Optimization with mpBAX")
    print("=" * 60)
    print(f"Objective: Minimize (x1-0.3)^2 + (x2-0.7)^2")
    print(f"Global minimum: (0.3, 0.7) with f = 0")
    print(f"Search space: [0, 1]^2")
    print("=" * 60)

    # Create algorithm
    algorithm = GreedySampling(
        n_propose=config['n_propose'],
        input_dim=2,
        seed=config['seed'],
        n_candidates=1000
    )

    # Create and run engine
    engine = Engine(
        config_path=config_path,
        fn_oracles=[oracle_quadratic],
        model_class=DummyModel,
        algorithm=algorithm
    )

    engine.run()

    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Total evaluations: {engine.evaluators[0].get_eval_count()}")

    # Load final data to find best solution
    from mpbax.core.checkpoint import CheckpointManager
    checkpoint_manager = CheckpointManager(config['checkpoint']['dir'])
    loop, data_handlers, _, _, _ = checkpoint_manager.load_checkpoint()

    X_all, Y_all = data_handlers[0].get_data()

    # Find best solution
    best_idx = np.argmin(Y_all)
    best_x = X_all[best_idx]
    best_y = Y_all[best_idx]

    print(f"\nBest solution found:")
    print(f"  x = [{best_x[0]:.4f}, {best_x[1]:.4f}]")
    print(f"  f(x) = {best_y[0]:.6f}")
    print(f"  True optimum: x = [0.3000, 0.7000], f(x) = 0.000000")
    print(f"  Distance to optimum: {np.linalg.norm(best_x - np.array([0.3, 0.7])):.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
