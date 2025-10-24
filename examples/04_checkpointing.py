"""Example 04: Checkpointing and Resuming

Demonstrates mpBAX's checkpointing capabilities:
- Automatic checkpointing every loop
- Resume from latest checkpoint
- Resume from specific loop
- Useful for long-running optimizations or recovering from crashes
"""
import numpy as np
import os
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


def expensive_oracle(X: np.ndarray) -> np.ndarray:
    """Simulates an expensive oracle function.

    Args:
        X: Input array of shape (n, d)

    Returns:
        Y: Output array of shape (n, 1)
    """
    return np.sum(X**2, axis=1, keepdims=True)


# Base configuration
base_config = {
    'seed': 42,
    'max_loops': 10,
    'checkpoint': {
        'dir': 'checkpoints_04',
        'freq': 1,  # Save every loop
        'resume_from': None  # Will be modified below
    },
    'oracles': [{
        'name': 'expensive',
        'input_dim': 3,
        'n_initial': 15,
        'function': {'class': expensive_oracle},
        'model': {'class': DummyModel}
    }],
    'algorithm': {
        'class': GreedySampling,
        'params': {
            'input_dims': [3],
            'n_propose': 8,
            'n_candidates': 300
        }
    }
}


def run_fresh():
    """Run optimization from scratch for 3 loops."""
    print("\n" + "="*60)
    print("PART 1: Running fresh optimization (3 loops)")
    print("="*60 + "\n")

    config = base_config.copy()
    config['max_loops'] = 3

    engine = Engine(config)
    engine.run()

    print("\n✓ Completed 3 loops and saved checkpoints\n")


def resume_from_latest():
    """Resume from latest checkpoint and continue."""
    print("\n" + "="*60)
    print("PART 2: Resuming from latest checkpoint")
    print("="*60 + "\n")

    if not os.path.exists('checkpoints_04'):
        print("ERROR: No checkpoints found. Run Part 1 first!")
        return

    config = base_config.copy()
    config['max_loops'] = 6  # Continue to loop 6
    config['checkpoint']['resume_from'] = 'latest'

    engine = Engine(config)
    engine.run()

    print("\n✓ Resumed from loop 3 and continued to loop 6\n")


def resume_from_specific():
    """Resume from a specific loop number."""
    print("\n" + "="*60)
    print("PART 3: Resuming from loop 2 specifically")
    print("="*60 + "\n")

    if not os.path.exists('checkpoints_04'):
        print("ERROR: No checkpoints found. Run Part 1 first!")
        return

    config = base_config.copy()
    config['max_loops'] = 5  # Continue to loop 5
    config['checkpoint']['resume_from'] = 2  # Resume from loop 2

    engine = Engine(config)
    engine.run()

    print("\n✓ Resumed from loop 2 and continued to loop 5\n")


if __name__ == '__main__':
    print("="*60)
    print("Example 04: Checkpointing and Resuming")
    print("="*60)
    print("\nThis example demonstrates:")
    print("  1. Fresh optimization with automatic checkpointing")
    print("  2. Resume from latest checkpoint")
    print("  3. Resume from specific loop")

    # Run all three parts
    run_fresh()
    resume_from_latest()
    # resume_from_specific()  # Uncomment to try specific resume

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)
    print("\nCheckpoint directory structure:")
    print("  checkpoints_04/")
    print("    ├── expensive/")
    print("    │   ├── data_0.pkl")
    print("    │   ├── data_1.pkl")
    print("    │   ├── model_0.pkl")
    print("    │   ├── model_1.pkl")
    print("    │   └── ...")
    print("    └── config.pkl")
    print("="*60)
