"""Example 06: DANetModel Plugin

Demonstrates the DANetModel plugin for deep neural network surrogate modeling:
- PyTorch-based deep neural network
- Finetune mode (preserves normalization and continues training)
- Sample weighting (emphasizes recent data)
- Adaptive epochs (more initial training, less incremental training)

Requirements: torch must be installed
"""
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.algorithm import GreedySampling

try:
    from mpbax.plugins.models.da_net_model import DANetModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not installed. DANetModel requires torch.")
    print("Install with: pip install torch")


def complex_function(X: np.ndarray) -> np.ndarray:
    """Complex non-linear function for surrogate modeling.

    Combines polynomial, sinusoidal, and interaction terms.

    Args:
        X: Input array of shape (n, d)

    Returns:
        Y: Output array of shape (n, 1)
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]

    result = (
        x1**2 + 0.5 * x2**2 + 0.1 * x3**2 +  # Quadratic
        0.3 * np.sin(5 * x1) * np.cos(5 * x2) +  # Interaction
        0.2 * x1 * x2 * x3 +  # 3-way interaction
        np.random.randn(X.shape[0]) * 0.05  # Small noise
    )
    return result.reshape(-1, 1)


# Configuration with DANetModel
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {
        'dir': 'checkpoints_06',
        'freq': 1
    },
    'model': {
        'mode': 'finetune',  # Reuse model across loops
        'checkpoint_mode': 'both'  # Save both best and final
    },
    'oracles': [{
        'name': 'complex_fn',
        'input_dim': 3,
        'n_initial': 50,
        'function': {'class': complex_function},
        'model': {
            'class': 'DANetModel',  # Can use string for built-in
            'params': {
                'epochs': 100,  # Initial training epochs
                'epochs_iter': 20,  # Incremental training epochs
                'n_neur': 400,  # Network width
                'dropout': 0.1,
                'weight_new_data': 10.0  # Weight recent data 10x
            }
        }
    }],
    'algorithm': {
        'class': GreedySampling,
        'params': {
            'input_dims': [3],
            'n_propose': 15,
            'n_candidates': 600
        }
    }
}


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("\nSkipping example - PyTorch not installed")
        print("Install with: pip install torch")
        exit(1)

    print("="*60)
    print("Example 06: DANetModel Plugin")
    print("="*60)
    print("\nFeatures demonstrated:")
    print("  • Deep neural network surrogate model")
    print("  • Finetune mode (model state preserved)")
    print("  • Sample weighting (recent data weighted 10x)")
    print("  • Adaptive epochs (100 initial, 20 incremental)")
    print("\nOracle: Complex non-linear function with interactions")
    print("\n" + "="*60 + "\n")

    engine = Engine(config)
    engine.run()

    print("\n" + "="*60)
    print("DANetModel example complete!")
    print("="*60)
    print("\nKey features:")
    print("  • Normalization fixed from loop 0 (X_mu, X_sigma)")
    print("  • Network weights continue from previous loop")
    print("  • Recent samples emphasized in loss function")
    print("  • Both best and final models checkpointed")
    print("\nSee mpbax/plugins/models/README.md for detailed documentation")
    print("="*60)
