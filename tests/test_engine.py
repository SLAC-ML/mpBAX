"""Tests for Engine end-to-end integration."""

import numpy as np
import tempfile

from mpbax.core.engine import Engine
from mpbax.core.model import BaseModel, DummyModel
from mpbax.core.algorithm import RandomSampling, GreedySampling


# Define StatefulModel at module level so it can be pickled for checkpointing
class StatefulModel(BaseModel):
    """Model that tracks training calls for testing finetune mode."""

    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.train_count = 0
        self.mean = None

    def train(self, X, Y, metadata=None):
        self.train_count += 1
        self.mean = Y.mean()
        self.is_trained = True

    def predict(self, X):
        if self.mean is None:
            return np.zeros((X.shape[0], 1))
        return np.full((X.shape[0], 1), self.mean)


def test_engine_with_instances():
    """Test Engine with instance-based config (Python API)."""
    print("Testing Engine with instances...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 3,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'oracles': [{
                'name': 'test',
                'input_dim': 2,
                'n_initial': 10,
                'function': {'class': oracle},  # Function instance
                'model': {'class': DummyModel}  # Class instance
            }],
            'algorithm': {
                'class': GreedySampling,  # Class instance
                'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 50}
            }
        }

        engine = Engine(config)
        engine.run()

        # Verify completion
        assert engine.current_loop == 3
        assert engine.evaluators[0].get_eval_count() == 10 + 5 + 5  # initial + 2 loops

    print("  ✓ Engine with instances passed")


def test_engine_with_strings():
    """Test Engine with string-based config (YAML compatibility)."""
    print("Testing Engine with strings...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 2,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'oracles': [{
                'name': 'test',
                'input_dim': 2,
                'n_initial': 10,
                'function': {'class': oracle},  # Use instance for simplicity
                'model': {'class': 'DummyModel'}  # String
            }],
            'algorithm': {
                'class': 'GreedySampling',  # String
                'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 50}
            }
        }

        engine = Engine(config)
        engine.run()

        assert engine.current_loop == 2

    print("  ✓ Engine with strings passed")


def test_engine_multi_oracle():
    """Test Engine with multiple oracles (multi-objective)."""
    print("Testing Engine with multiple oracles...")

    def oracle1(X):
        return np.sum(X**2, axis=1, keepdims=True)

    def oracle2(X):
        return np.sum(X, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 3,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'oracles': [
                {
                    'name': 'oracle1',
                    'input_dim': 2,
                    'n_initial': 10,
                    'function': {'class': oracle1},
                    'model': {'class': DummyModel}
                },
                {
                    'name': 'oracle2',
                    'input_dim': 3,
                    'n_initial': 8,
                    'function': {'class': oracle2},
                    'model': {'class': DummyModel}
                }
            ],
            'algorithm': {
                'class': RandomSampling,
                'params': {'input_dims': [2, 3], 'n_propose': 5}
            }
        }

        engine = Engine(config)
        engine.run()

        # Verify both oracles ran
        assert len(engine.evaluators) == 2
        assert engine.evaluators[0].get_eval_count() == 10 + 5 + 5
        assert engine.evaluators[1].get_eval_count() == 8 + 5 + 5

    print("  ✓ Engine with multiple oracles passed")


def test_engine_finetune_mode():
    """Test Engine with finetune mode (model state preservation)."""
    print("Testing Engine finetune mode...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 3,
            'checkpoint': {'dir': tmpdir, 'freq': 999},  # Disable for this test
            'model': {'mode': 'finetune'},  # Enable finetune mode
            'oracles': [{
                'name': 'test',
                'input_dim': 2,
                'n_initial': 10,
                'function': {'class': oracle},
                'model': {'class': StatefulModel}
            }],
            'algorithm': {
                'class': RandomSampling,
                'params': {'input_dims': [2], 'n_propose': 5}
            }
        }

        engine = Engine(config)
        engine.run()

        # In finetune mode, model instance is reused across loops
        # We can verify by checking train_count - should be 3 (once per loop)
        assert engine.models[0].train_count == 3

        # Verify model was actually trained
        assert engine.models[0].is_trained
        assert engine.models[0].mean is not None

    print("  ✓ Engine finetune mode passed")


def test_engine_retrain_mode():
    """Test Engine with retrain mode (fresh model each loop)."""
    print("Testing Engine retrain mode...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 2,
            'checkpoint': {'dir': tmpdir, 'freq': 999},  # Disable for this test
            'model': {'mode': 'retrain'},  # Retrain mode (default)
            'oracles': [{
                'name': 'test',
                'input_dim': 2,
                'n_initial': 10,
                'function': {'class': oracle},
                'model': {'class': DummyModel}
            }],
            'algorithm': {
                'class': RandomSampling,
                'params': {'input_dims': [2], 'n_propose': 5}
            }
        }

        engine = Engine(config)
        engine.run()

        assert engine.current_loop == 2

    print("  ✓ Engine retrain mode passed")


def test_engine_checkpoint_resume():
    """Test Engine checkpoint and resume functionality."""
    print("Testing Engine checkpoint/resume...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # First run: complete 2 loops
        config = {
            'seed': 42,
            'max_loops': 2,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'oracles': [{
                'name': 'test',
                'input_dim': 2,
                'n_initial': 10,
                'function': {'class': oracle},
                'model': {'class': DummyModel}
            }],
            'algorithm': {
                'class': RandomSampling,
                'params': {'input_dims': [2], 'n_propose': 5}
            }
        }

        engine1 = Engine(config)
        engine1.run()

        first_eval_count = engine1.evaluators[0].get_eval_count()

        # Second run: resume and continue for 2 more loops
        config['max_loops'] = 4
        config['checkpoint']['resume_from'] = 'latest'

        engine2 = Engine(config)
        engine2.run()

        # Should have added 2 more loops worth of evaluations
        second_eval_count = engine2.evaluators[0].get_eval_count()
        assert second_eval_count == 5 + 5  # 2 more loops

    print("  ✓ Engine checkpoint/resume passed")


def run_all_tests():
    """Run all Engine tests."""
    print("\n" + "="*60)
    print("Engine Integration Tests")
    print("="*60 + "\n")

    test_engine_with_instances()
    test_engine_with_strings()
    test_engine_multi_oracle()
    test_engine_finetune_mode()
    test_engine_retrain_mode()
    test_engine_checkpoint_resume()

    print("\n" + "="*60)
    print("All Engine tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
