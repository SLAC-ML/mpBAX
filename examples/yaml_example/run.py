"""YAML-based mpBAX example.

This example demonstrates loading configuration from a YAML file and running
the optimization engine. This approach is ideal for:
- Production workflows
- Configuration management
- Reproducible experiments
- Team collaboration

The config.yaml file contains all settings including import paths to oracle
functions defined in oracles.py.
"""

import sys
from pathlib import Path

# Add mpBAX root to path so 'examples' package is importable
mpbax_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(mpbax_root))

from mpbax.core.engine import Engine


def main():
    """Run optimization using YAML configuration."""
    # Get path to config.yaml in this directory
    config_path = Path(__file__).parent / 'config.yaml'

    print("="*60)
    print("mpBAX YAML Configuration Example")
    print("="*60)
    print(f"\nLoading config from: {config_path}")

    # Create engine from YAML file
    engine = Engine(str(config_path))

    print("\nStarting optimization...")
    print("-"*60)

    # Run optimization
    engine.run()

    print("\n" + "="*60)
    print("Optimization complete!")
    print("="*60)

    # Print summary
    print(f"\nCompleted {engine.current_loop} loops")
    print(f"Total evaluations: {engine.evaluators[0].get_eval_count()}")
    print(f"Checkpoints saved to: {engine.checkpoint_manager.checkpoint_dir}")

    # Get best result
    X, Y = engine.data_handlers[0].get_data()
    best_idx = Y.argmin()
    print(f"\nBest objective value: {Y[best_idx, 0]:.6f}")
    print(f"Best parameters: {X[best_idx]}")


if __name__ == '__main__':
    main()
