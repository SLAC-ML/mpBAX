"""
Example demonstrating DA_Net model plugin for mpBAX.

This example shows:
1. Using DANetModel plugin with PyTorch neural network
2. Automatic input normalization (preserved across loops)
3. Finetuning mode with best model tracking
4. GPU acceleration when available
5. Early stopping for efficient training
"""

import numpy as np

from mpbax.core.engine import Engine
from mpbax.plugins.models import DANetModel


# Oracle function - simple quadratic
def oracle_quadratic(X: np.ndarray) -> np.ndarray:
    """Simple quadratic oracle: f(x1, x2) = (x1-0.3)^2 + (x2-0.7)^2

    Optimum at x = [0.3, 0.7] with f(x) = 0
    """
    Y = (X[:, 0] - 0.3)**2 + (X[:, 1] - 0.7)**2
    return Y.reshape(-1, 1)


def main():
    """Run optimization with DA_Net model."""

    print("=" * 70)
    print("DA_Net Model Plugin Example")
    print("=" * 70)

    # Check if PyTorch is available
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nERROR: PyTorch not found. Install with: pip install torch")
        print("This example requires PyTorch to run.")
        return

    # Config with DA_Net model
    config = {
        'seed': 42,
        'max_loops': 5,
        'n_initial': 20,
        'checkpoint': {
            'dir': 'checkpoints_danet',
            'freq': 1,
            'resume_from': None
        },
        'oracles': [
            {'name': 'quadratic', 'input_dim': 2}
        ],
        'model': {
            'mode': 'finetune',  # Enable finetuning - preserve normalization
            'checkpoint_mode': 'both'  # Save both best and final models
        },
        'algorithm': {
            'class': 'GreedySampling',
            'params': {
                'input_dims': [2],
                'n_propose': 10,
                'n_candidates': 500
            }
        }
    }

    print("\nConfiguration:")
    print(f"  Model: DANetModel (PyTorch neural network)")
    print(f"  Model mode: {config['model']['mode']} (preserves normalization)")
    print(f"  Checkpoint mode: {config['model']['checkpoint_mode']}")
    print(f"  Max loops: {config['max_loops']}")
    print(f"  Initial samples: {config['n_initial']}")
    print("=" * 70)

    # Run optimization
    engine = Engine(
        config=config,
        fn_oracles=[oracle_quadratic],
        model_class=DANetModel,
        algorithm=None  # Auto-instantiate from config
    )

    engine.run()

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    # Inspect final model state
    final_model = engine.models[0]
    state = final_model.get_state()

    print(f"\nFinal model state:")
    print(f"  Device: {final_model.device}")
    print(f"  Normalization (from loop 0):")
    print(f"    X_mu = {state['X_mu'].flatten()}")
    print(f"    X_sigma = {state['X_sigma'].flatten()}")
    print(f"  Best test loss: {state['best_test_loss']:.6f}")

    # Load and check results
    from mpbax.core.checkpoint import CheckpointManager
    checkpoint_manager = CheckpointManager(config['checkpoint']['dir'])
    loop, data_handlers, _, _, _ = checkpoint_manager.load_checkpoint()

    X_all, Y_all = data_handlers[0].get_data()
    best_idx = np.argmin(Y_all)
    best_x = X_all[best_idx]
    best_y = Y_all[best_idx]

    print(f"\nBest solution found:")
    print(f"  x = [{best_x[0]:.4f}, {best_x[1]:.4f}]")
    print(f"  f(x) = {best_y[0]:.6f}")
    print(f"  True optimum: x = [0.3000, 0.7000], f(x) = 0.000000")
    print(f"  Distance to optimum: {np.linalg.norm(best_x - np.array([0.3, 0.7])):.6f}")

    # Verify checkpoints include both best and final models
    import os
    checkpoint_dir = config['checkpoint']['dir']
    oracle_0_dir = os.path.join(checkpoint_dir, 'oracle_0')

    print(f"\nCheckpoint files in {oracle_0_dir}:")
    for loop_i in range(config['max_loops']):
        final_model_path = os.path.join(oracle_0_dir, f'model_{loop_i}_final.pkl')
        best_model_path = os.path.join(oracle_0_dir, f'model_{loop_i}_best.pkl')

        has_final = os.path.exists(final_model_path)
        has_best = os.path.exists(best_model_path)

        print(f"  Loop {loop_i}: final={has_final}, best={has_best}")

    print("\n" + "=" * 70)
    print("Key Features Demonstrated:")
    print("  ✓ PyTorch neural network as surrogate model")
    print("  ✓ Initial normalization preserved across all loops")
    print("  ✓ Model finetuned (not retrained from scratch)")
    print("  ✓ Best and final models tracked and saved")
    print("  ✓ Early stopping for efficient training")
    print("  ✓ GPU acceleration (when available)")
    print("=" * 70)


if __name__ == "__main__":
    main()
