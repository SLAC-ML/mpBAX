"""
Example demonstrating checkpoint, resume, and rollback functionality.

This example shows how to:
1. Run optimization with checkpointing
2. Resume from latest checkpoint
3. Rollback to a specific checkpoint and continue from there
"""

import numpy as np
import yaml
import os
import shutil
from pathlib import Path

from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import RandomSampling
from mpbax.core.checkpoint import CheckpointManager


# Simple oracle function
def oracle_sphere(X: np.ndarray) -> np.ndarray:
    """Oracle function: sphere function (minimize sum of squares).

    Args:
        X: Input array with shape (n, 2)

    Returns:
        Y: Output array with shape (n, 1)
    """
    return np.sum(X**2, axis=1, keepdims=True)


def run_initial():
    """Step 1: Run initial optimization with checkpointing."""
    print("\n" + "=" * 70)
    print("STEP 1: Initial Run (5 loops)")
    print("=" * 70)

    config = {
        'seed': 42,
        'max_loops': 5,
        'n_initial': 10,
        'checkpoint': {
            'dir': 'checkpoint_demo',
            'freq': 1,
            'resume_from': None
        },
        'n_propose': 5,
        'oracles': [
            {'name': 'sphere', 'input_dim': 2}
        ]
    }

    config_path = 'config_checkpoint_demo.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    algorithm = RandomSampling(input_dims=[2], n_propose=5, seed=42)

    engine = Engine(
        config_path=config_path,
        fn_oracles=[oracle_sphere],
        model_class=DummyModel,
        algorithm=algorithm
    )

    engine.run()

    # Check available checkpoints
    checkpoint_manager = CheckpointManager('checkpoint_demo')
    loops = checkpoint_manager.list_checkpoints()
    print(f"\nCheckpoints created: {loops}")
    print(f"Total evaluations: {engine.evaluators[0].get_eval_count()}")


def run_resume():
    """Step 2: Resume from latest checkpoint and continue."""
    print("\n" + "=" * 70)
    print("STEP 2: Resume from Latest Checkpoint (continue to loop 8)")
    print("=" * 70)

    config = {
        'seed': 42,
        'max_loops': 8,  # Continue for 3 more loops (5, 6, 7)
        'n_initial': 10,
        'checkpoint': {
            'dir': 'checkpoint_demo',
            'freq': 1,
            'resume_from': 'latest'  # Resume from latest checkpoint
        },
        'n_propose': 5,
        'oracles': [
            {'name': 'sphere', 'input_dim': 2}
        ]
    }

    config_path = 'config_checkpoint_demo.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    algorithm = RandomSampling(input_dims=[2], n_propose=5, seed=42)

    engine = Engine(
        config_path=config_path,
        fn_oracles=[oracle_sphere],
        model_class=DummyModel,
        algorithm=algorithm
    )

    engine.run()

    # Check checkpoints
    checkpoint_manager = CheckpointManager('checkpoint_demo')
    loops = checkpoint_manager.list_checkpoints()
    print(f"\nCheckpoints available: {loops}")
    print(f"New evaluations in this run: {engine.evaluators[0].get_eval_count()}")


def run_rollback():
    """Step 3: Rollback to loop 3 and continue from there."""
    print("\n" + "=" * 70)
    print("STEP 3: Rollback to Loop 3 (delete loops 4-7, continue to loop 6)")
    print("=" * 70)

    # Rollback: delete checkpoints after loop 3
    checkpoint_manager = CheckpointManager('checkpoint_demo')
    print(f"Before rollback: {checkpoint_manager.list_checkpoints()}")

    checkpoint_manager.delete_checkpoints_after(loop=3)
    print(f"After rollback: {checkpoint_manager.list_checkpoints()}")

    # Resume from loop 3 and continue
    config = {
        'seed': 999,  # Different seed for different trajectory
        'max_loops': 6,  # Continue to loop 6 (will run loops 4, 5)
        'n_initial': 10,
        'checkpoint': {
            'dir': 'checkpoint_demo',
            'freq': 1,
            'resume_from': 'latest'  # Will load loop 3
        },
        'n_propose': 5,
        'oracles': [
            {'name': 'sphere', 'input_dim': 2}
        ]
    }

    config_path = 'config_checkpoint_demo.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    algorithm = RandomSampling(input_dims=[2], n_propose=5, seed=999)

    engine = Engine(
        config_path=config_path,
        fn_oracles=[oracle_sphere],
        model_class=DummyModel,
        algorithm=algorithm
    )

    engine.run()

    # Check final checkpoints
    loops = checkpoint_manager.list_checkpoints()
    print(f"\nFinal checkpoints: {loops}")
    print(f"New evaluations after rollback: {engine.evaluators[0].get_eval_count()}")


def analyze_results():
    """Analyze final results."""
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    checkpoint_manager = CheckpointManager('checkpoint_demo')
    loop, data_handlers, _, _, _ = checkpoint_manager.load_checkpoint()

    X_all, Y_all = data_handlers[0].get_data()

    best_idx = np.argmin(Y_all)
    best_x = X_all[best_idx]
    best_y = Y_all[best_idx]

    print(f"Final loop: {loop}")
    print(f"Total data points: {X_all.shape[0]}")
    print(f"Best solution: x = [{best_x[0]:.4f}, {best_x[1]:.4f}]")
    print(f"Best value: f(x) = {best_y[0]:.6f}")
    print(f"Distance to optimum (0, 0): {np.linalg.norm(best_x):.6f}")


def main():
    """Run the complete checkpoint workflow demonstration."""
    print("\n" + "=" * 70)
    print("Checkpoint/Resume/Rollback Workflow Demonstration")
    print("=" * 70)

    # Clean up any existing checkpoint directory
    if os.path.exists('checkpoint_demo'):
        shutil.rmtree('checkpoint_demo')

    # Run the workflow
    run_initial()
    run_resume()
    run_rollback()
    analyze_results()

    print("\n" + "=" * 70)
    print("Workflow Complete!")
    print("=" * 70)
    print("\nThis demonstrates:")
    print("  1. Creating checkpoints during optimization")
    print("  2. Resuming from the latest checkpoint")
    print("  3. Rolling back to a specific checkpoint")
    print("  4. Continuing optimization from the rollback point")
    print("=" * 70)


if __name__ == "__main__":
    main()
