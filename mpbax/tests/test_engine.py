"""Tests for Engine with full workflow."""

import numpy as np
import tempfile
import yaml
import shutil
from pathlib import Path

from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import RandomSampling


# Simple oracle functions for testing
def oracle_sum_squares(X: np.ndarray) -> np.ndarray:
    """Oracle: sum of squares."""
    return np.sum(X**2, axis=1, keepdims=True)


def oracle_sum(X: np.ndarray) -> np.ndarray:
    """Oracle: simple sum."""
    return np.sum(X, axis=1, keepdims=True)


def test_single_objective_run():
    """Test full run with single objective."""
    print("Testing single-objective run...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config file
        config = {
            'seed': 42,
            'max_loops': 3,
            'n_initial': 5,
            'checkpoint': {
                'dir': str(Path(tmpdir) / 'checkpoints'),
                'freq': 1,
                'resume_from': None
            },
            'oracles': [
                {'name': 'obj_1', 'input_dim': 2}
            ]
        }

        config_path = Path(tmpdir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Create algorithm
        algorithm = RandomSampling(input_dims=[2], n_propose=3, seed=42)

        # Create and run engine
        engine = Engine(
            config_path=str(config_path),
            fn_oracles=[oracle_sum_squares],
            model_class=DummyModel,
            algorithm=algorithm
        )

        engine.run()

        # Verify checkpoints were created
        checkpoint_dir = Path(config['checkpoint']['dir'])
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / 'config.yaml').exists()
        assert (checkpoint_dir / 'obj_0' / 'data_0.pkl').exists()
        assert (checkpoint_dir / 'obj_0' / 'model_0.pkl').exists()

        # Verify all loops completed
        loops = engine.checkpoint_manager.list_checkpoints()
        assert loops == [0, 1, 2]

        # Verify evaluation count
        # Loop 0: 5 initial
        # Loop 1: 3 proposed
        # Loop 2: 3 proposed
        # Total: 11
        assert engine.evaluators[0].get_eval_count() == 11

    print("Single-objective run test passed!\n")


def test_multi_objective_run():
    """Test full run with multiple objectives."""
    print("Testing multi-objective run...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config file for 2 objectives
        config = {
            'seed': 42,
            'max_loops': 2,
            'n_initial': 4,
            'checkpoint': {
                'dir': str(Path(tmpdir) / 'checkpoints'),
                'freq': 1,
                'resume_from': None
            },
            'oracles': [
                {'name': 'objective_1', 'input_dim': 2},
                {'name': 'objective_2', 'input_dim': 3}
            ]
        }

        config_path = Path(tmpdir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Create algorithm with dimensions for both oracles
        algorithm = RandomSampling(input_dims=[2, 3], n_propose=2, seed=42)

        # Create and run engine with 2 oracles
        engine = Engine(
            config_path=str(config_path),
            fn_oracles=[oracle_sum_squares, oracle_sum],
            model_class=DummyModel,
            algorithm=algorithm
        )

        engine.run()

        # Verify checkpoints for both objectives
        checkpoint_dir = Path(config['checkpoint']['dir'])
        for obj_idx in [0, 1]:
            assert (checkpoint_dir / f'obj_{obj_idx}' / 'data_0.pkl').exists()
            assert (checkpoint_dir / f'obj_{obj_idx}' / 'model_0.pkl').exists()

        # Verify evaluation counts
        # Each objective: Loop 0: 4 initial, Loop 1: 2 proposed = 6 total
        assert engine.evaluators[0].get_eval_count() == 6
        assert engine.evaluators[1].get_eval_count() == 6

    print("Multi-objective run test passed!\n")


def test_resume_from_checkpoint():
    """Test resuming from checkpoint."""
    print("Testing resume from checkpoint...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / 'checkpoints'

        # First run: loops 0 and 1 (max_loops=2 means run while current_loop < 2)
        config = {
            'seed': 42,
            'max_loops': 2,
            'n_initial': 3,
            'checkpoint': {
                'dir': str(checkpoint_dir),
                'freq': 1,
                'resume_from': None
            },
            'oracles': [
                {'name': 'obj_1', 'input_dim': 2}
            ]
        }

        config_path = Path(tmpdir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        algorithm = RandomSampling(input_dims=[2], n_propose=2, seed=42)

        # Run first engine
        engine1 = Engine(
            config_path=str(config_path),
            fn_oracles=[oracle_sum_squares],
            model_class=DummyModel,
            algorithm=algorithm
        )
        engine1.run()

        first_eval_count = engine1.evaluators[0].get_eval_count()
        # Loop 0: 3 initial, Loop 1: 2 proposed = 5 total
        assert first_eval_count == 5

        # Second run: resume and continue for 2 more loops (loops 2 and 3)
        config['checkpoint']['resume_from'] = 'latest'
        config['max_loops'] = 4  # Continue until loop 4 (runs loops 2 and 3)

        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        algorithm2 = RandomSampling(input_dims=[2], n_propose=2, seed=42)

        engine2 = Engine(
            config_path=str(config_path),
            fn_oracles=[oracle_sum_squares],
            model_class=DummyModel,
            algorithm=algorithm2
        )
        engine2.run()

        # Should have added 2 more loops (2 evaluations each)
        second_eval_count = engine2.evaluators[0].get_eval_count()
        assert second_eval_count == 4  # 2 + 2 for loops 2 and 3

        # Verify total checkpoints
        loops = engine2.checkpoint_manager.list_checkpoints()
        assert loops == [0, 1, 2, 3]

    print("Resume from checkpoint test passed!\n")


def test_checkpoint_frequency():
    """Test checkpoint frequency setting."""
    print("Testing checkpoint frequency...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with checkpoint every 2 loops
        config = {
            'seed': 42,
            'max_loops': 5,
            'n_initial': 3,
            'checkpoint': {
                'dir': str(Path(tmpdir) / 'checkpoints'),
                'freq': 2,  # Checkpoint every 2 loops
                'resume_from': None
            },
            'oracles': [
                {'name': 'obj_1', 'input_dim': 2}
            ]
        }

        config_path = Path(tmpdir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        algorithm = RandomSampling(input_dims=[2], n_propose=2, seed=42)

        engine = Engine(
            config_path=str(config_path),
            fn_oracles=[oracle_sum_squares],
            model_class=DummyModel,
            algorithm=algorithm
        )
        engine.run()

        # Should have checkpoints at loops 0, 2, 4 (every 2nd loop)
        loops = engine.checkpoint_manager.list_checkpoints()
        assert loops == [0, 2, 4]

    print("Checkpoint frequency test passed!\n")


if __name__ == "__main__":
    test_single_objective_run()
    test_multi_objective_run()
    test_resume_from_checkpoint()
    test_checkpoint_frequency()
    print("All Engine tests passed!")
