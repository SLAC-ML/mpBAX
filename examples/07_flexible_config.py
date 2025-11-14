"""Example 07: Flexible Config Patterns

This example demonstrates the new flexible parameter placement patterns
introduced in mpBAX v2:

1. input_dim can be specified in oracle config, model.params, or generate.params.d
2. n_initial can be specified in oracle config or generate.params.n
3. 'training' instead of 'model' for top-level config (clearer naming)
4. Custom generators receive all params during call (not as factory)
5. Default generator shortcut: generate.params.n without specifying class

All old config patterns remain supported for backward compatibility.
"""

import numpy as np
import tempfile
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling


def quadratic(X):
    """Simple quadratic oracle function."""
    return np.sum((X - 0.5)**2, axis=1, keepdims=True)


def custom_generator(n, d, scale=1.0):
    """Custom generator with optional scale parameter.

    Unlike oracle factories, generators are NOT called during instantiation.
    They receive all params during the actual call.
    """
    return np.random.rand(n, d) * scale


def main():
    print("\n" + "="*60)
    print("Example 07: Flexible Config Patterns")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'seed': 42,
            'max_loops': 3,
            'checkpoint': {'dir': tmpdir, 'freq': 1},

            # Pattern 1: Use 'training' instead of 'model' (clearer naming)
            'training': {
                'mode': 'retrain',
                'checkpoint_mode': 'final'
            },

            'oracles': [
                # Oracle 1: Traditional pattern (backward compatible)
                {
                    'name': 'traditional',
                    'input_dim': 2,
                    'n_initial': 10,
                    'function': {'class': quadratic},
                    'model': {'class': DummyModel}
                },

                # Oracle 2: input_dim in model.params
                {
                    'name': 'input_dim_in_model',
                    'n_initial': 10,
                    'function': {'class': quadratic},
                    'model': {
                        'class': DummyModel,
                        'params': {'input_dim': 2}  # Specified here instead
                    }
                },

                # Oracle 3: Default generator shortcut with n in generate.params
                {
                    'name': 'default_gen_shortcut',
                    'input_dim': 2,
                    # No n_initial at oracle level!
                    'generate': {
                        # No 'class' field - uses default generator
                        'params': {'n': 10}  # Shortcut to specify n
                    },
                    'function': {'class': quadratic},
                    'model': {'class': DummyModel}
                },

                # Oracle 4: Custom generator with all params
                {
                    'name': 'custom_gen_all_params',
                    # No oracle-level input_dim or n_initial!
                    'generate': {
                        'class': custom_generator,
                        'params': {
                            'n': 10,      # Number of samples
                            'd': 2,       # Dimensionality
                            'scale': 0.5  # Custom parameter
                        }
                    },
                    'function': {'class': quadratic},
                    'model': {
                        'class': DummyModel,
                        'params': {'input_dim': 2}
                    }
                },
            ],

            'algorithm': {
                'class': GreedySampling,
                'params': {
                    'input_dims': [2, 2, 2, 2],
                    'n_propose': 5,
                    'n_candidates': 50
                }
            }
        }

        print("\nConfig patterns demonstrated:")
        print("1. Top-level 'training' instead of 'model'")
        print("2. Oracle 1: Traditional pattern (backward compatible)")
        print("3. Oracle 2: input_dim in model.params")
        print("4. Oracle 3: Default generator shortcut (generate.params.n)")
        print("5. Oracle 4: Custom generator with all params (n, d, scale)")

        print("\n" + "-"*60)
        print("Running optimization...")
        print("-"*60)

        engine = Engine(config)
        engine.run()

        print("\n" + "="*60)
        print("Success! All flexible config patterns work correctly.")
        print("="*60)

        # Show final results
        print("\nFinal evaluation counts:")
        for i, oracle_name in enumerate(['traditional', 'input_dim_in_model',
                                         'default_gen_shortcut', 'custom_gen_all_params']):
            eval_count = engine.evaluators[i].eval_count
            print(f"  {oracle_name}: {eval_count} evaluations")


if __name__ == '__main__':
    main()
