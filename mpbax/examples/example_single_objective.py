"""
Simple single-objective optimization example.

This example demonstrates how to use mpBAX to optimize a simple 2D function:
f(x1, x2) = (x1 - 0.3)^2 + (x2 - 0.7)^2

The global minimum is at (0.3, 0.7) with f = 0.

This example uses the pure config-first API - everything is specified in config.yaml.
"""

import numpy as np
import yaml
from pathlib import Path

from mpbax.core.engine import Engine


def main():
    """Run single-objective optimization."""

    # Create config with NEW structure - everything specified in config!
    config = {
        'seed': 42,
        'max_loops': 10,
        'checkpoint': {
            'dir': 'checkpoints_single_obj',
            'freq': 1,
            'resume_from': None  # Set to 'latest' to resume
        },
        'oracles': [
            {
                'name': 'quadratic',
                'input_dim': 2,
                'n_initial': 20,  # Per-oracle n_initial
                'function': {
                    'class': 'mpbax.examples.oracles.oracle_quadratic',
                    'params': {}
                },
                'generate': None,  # Use default uniform [0, 1]^d generator
                'model': {
                    'class': 'DummyModel',
                    'params': {}
                }
            }
        ],
        'algorithm': {
            'class': 'GreedySampling',
            'params': {
                'input_dims': [2],
                'n_propose': 10,
                'n_candidates': 1000
            }
        }
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

    # NEW API: Engine takes only config!
    engine = Engine(config=config)

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
