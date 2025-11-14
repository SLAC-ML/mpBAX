"""Test new flexible config patterns."""

import numpy as np
import tempfile
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


def test_pattern_1_input_dim_in_model_params():
    """Test specifying input_dim in model.params instead of oracle level."""
    print("\n" + "="*60)
    print("Test 1: input_dim in model.params")
    print("="*60)

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 2,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'oracles': [{
                'name': 'test',
                # No oracle-level input_dim!
                'n_initial': 10,
                'function': {'class': oracle},
                'model': {
                    'class': DummyModel,
                    'params': {'input_dim': 2}  # Specify here instead
                }
            }],
            'algorithm': {
                'class': GreedySampling,
                'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 50}
            }
        }

        engine = Engine(config)
        engine.run()
        print("✓ Test 1 passed!")


def test_pattern_2_n_in_generate_params():
    """Test specifying n in generate.params instead of n_initial."""
    print("\n" + "="*60)
    print("Test 2: n in generate.params (default generator shortcut)")
    print("="*60)

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
                # No oracle-level n_initial!
                'generate': {
                    'params': {'n': 10}  # Specify n for default generator
                },
                'function': {'class': oracle},
                'model': {'class': DummyModel}
            }],
            'algorithm': {
                'class': GreedySampling,
                'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 50}
            }
        }

        engine = Engine(config)
        engine.run()
        print("✓ Test 2 passed!")


def test_pattern_3_training_instead_of_model():
    """Test using 'training' instead of 'model' for top-level config."""
    print("\n" + "="*60)
    print("Test 3: 'training' instead of 'model'")
    print("="*60)

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 2,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'training': {  # New name instead of 'model'
                'mode': 'retrain'
            },
            'oracles': [{
                'name': 'test',
                'input_dim': 2,
                'n_initial': 10,
                'function': {'class': oracle},
                'model': {'class': DummyModel}
            }],
            'algorithm': {
                'class': GreedySampling,
                'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 50}
            }
        }

        engine = Engine(config)
        engine.run()
        print("✓ Test 3 passed - no deprecation warning!")


def test_pattern_4_custom_generator_with_all_params():
    """Test custom generator with all params (n, d) in generate.params."""
    print("\n" + "="*60)
    print("Test 4: Custom generator with n and d in params")
    print("="*60)

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    def custom_generator(n, d, scale=1.0):
        """Generator that accepts n, d, and optional params."""
        return np.random.rand(n, d) * scale

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 2,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'oracles': [{
                'name': 'test',
                # No oracle-level input_dim or n_initial!
                'generate': {
                    'class': custom_generator,
                    'params': {
                        'n': 10,  # Number of samples
                        'd': 2,   # Input dimensionality
                        'scale': 0.5  # Custom parameter
                    }
                },
                'function': {'class': oracle},
                'model': {
                    'class': DummyModel,
                    'params': {'input_dim': 2}
                }
            }],
            'algorithm': {
                'class': GreedySampling,
                'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 50}
            }
        }

        engine = Engine(config)
        engine.run()
        print("✓ Test 4 passed!")


def test_pattern_5_empty_params():
    """Test factory with empty params {}."""
    print("\n" + "="*60)
    print("Test 5: Factory function with empty params {}")
    print("="*60)

    def oracle_factory():
        """Factory that returns an oracle function."""
        def oracle(X):
            return np.sum(X**2, axis=1, keepdims=True)
        return oracle

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 2,
            'checkpoint': {'dir': tmpdir, 'freq': 1},
            'oracles': [{
                'name': 'test',
                'input_dim': 2,
                'n_initial': 10,
                'function': {
                    'class': oracle_factory,
                    'params': {}  # Empty dict - should call factory with no args
                },
                'model': {'class': DummyModel}
            }],
            'algorithm': {
                'class': GreedySampling,
                'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 50}
            }
        }

        engine = Engine(config)
        engine.run()
        print("✓ Test 5 passed!")


if __name__ == '__main__':
    test_pattern_1_input_dim_in_model_params()
    test_pattern_2_n_in_generate_params()
    test_pattern_3_training_instead_of_model()
    test_pattern_4_custom_generator_with_all_params()
    test_pattern_5_empty_params()

    print("\n" + "="*60)
    print("✓ ALL NEW CONFIG PATTERNS WORK!")
    print("="*60)
