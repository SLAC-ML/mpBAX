"""Example 08: Checkpoint Analysis

This example demonstrates how to load models and data from specific loops
for post-optimization analysis using mpBAX utilities.

Use cases:
- Test how models from different loops perform
- Examine data collection patterns
- Debug optimization issues
- Analyze convergence
- Test algorithm proposals with historical models
"""

import numpy as np
from mpbax.utils import (
    load_models_for_analysis,
    load_data_from_loop,
    load_accumulated_data,
    inspect_checkpoint
)


def main():
    # NOTE: Update this path to your actual checkpoint directory
    checkpoint_dir = 'checkpoints/'

    print("="*70)
    print("Example 08: Checkpoint Analysis")
    print("="*70)

    # ========================================================================
    # Part 1: Inspect checkpoint directory
    # ========================================================================
    print("\n" + "-"*70)
    print("Part 1: Quick Checkpoint Inspection")
    print("-"*70)

    info = inspect_checkpoint(checkpoint_dir)

    print(f"\nCheckpoint Summary:")
    print(f"  Latest loop: {info['latest_loop']}")
    print(f"  Number of oracles: {len(info['oracle_names'])}")
    print(f"  Oracle names: {info['oracle_names']}")

    print(f"\n  Available loops per oracle:")
    for oracle_name, loops in info['available_loops'].items():
        if loops:
            print(f"    {oracle_name}: loops {min(loops)}-{max(loops)} ({len(loops)} total)")
        else:
            print(f"    {oracle_name}: no data")

    # ========================================================================
    # Part 2: Load models from a specific loop
    # ========================================================================
    print("\n" + "-"*70)
    print("Part 2: Load Models from Loop 5")
    print("-"*70)

    # Load models from loop 5
    try:
        result = load_models_for_analysis(checkpoint_dir, loop=5, mode='final')

        print(f"\nLoaded models from loop {result['loop']}:")
        print(f"  Number of models: {len(result['models'])}")
        print(f"  Oracle names: {result['oracle_names']}")

        # Test models with sample data
        print(f"\n  Testing models with sample inputs:")
        for i, (model, oracle_name) in enumerate(zip(result['models'], result['oracle_names'])):
            # Get input dimension from config or model
            if hasattr(model, 'input_dim'):
                input_dim = model.input_dim
            else:
                # Infer from first oracle's data
                try:
                    data = load_data_from_loop(checkpoint_dir, loop=0, oracle_idx=i)
                    input_dim = data['X'].shape[1]
                except:
                    input_dim = 2  # Fallback

            # Generate test data
            X_test = np.random.rand(5, input_dim)
            Y_pred = model.predict(X_test)

            print(f"    {oracle_name}:")
            print(f"      Input shape: {X_test.shape}")
            print(f"      Output shape: {Y_pred.shape}")
            print(f"      Prediction range: [{Y_pred.min():.4f}, {Y_pred.max():.4f}]")

        # Get fn_pred_list for use with Algorithm.propose()
        fn_pred_list = result['fn_pred_list']
        print(f"\n  fn_pred_list ready for Algorithm.propose():")
        print(f"    Type: {type(fn_pred_list)}")
        print(f"    Length: {len(fn_pred_list)}")
        print(f"    Example usage: X_proposed = my_algorithm.propose(fn_pred_list)")

    except FileNotFoundError as e:
        print(f"  ⚠ Loop 5 not found: {e}")
        print(f"  Try with loop 0 or check available loops above.")

    # ========================================================================
    # Part 3: Load data from specific loop
    # ========================================================================
    print("\n" + "-"*70)
    print("Part 3: Load Data from Specific Loop")
    print("-"*70)

    # Examine data from loop 0 (initial sampling)
    oracle_name = info['oracle_names'][0] if info['oracle_names'] else 'oracle_0'

    print(f"\n  Loading data from loop 0 for '{oracle_name}':")
    try:
        data_loop0 = load_data_from_loop(checkpoint_dir, loop=0, oracle_name=oracle_name)

        print(f"    Number of samples: {data_loop0['n_samples']}")
        print(f"    X shape: {data_loop0['X'].shape}")
        print(f"    Y shape: {data_loop0['Y'].shape}")
        print(f"    X range: [{data_loop0['X'].min():.4f}, {data_loop0['X'].max():.4f}]")
        print(f"    Y range: [{data_loop0['Y'].min():.4f}, {data_loop0['Y'].max():.4f}]")

        # Show statistics
        print(f"\n    X statistics per dimension:")
        for dim in range(data_loop0['X'].shape[1]):
            x_dim = data_loop0['X'][:, dim]
            print(f"      Dim {dim}: mean={x_dim.mean():.4f}, std={x_dim.std():.4f}")

    except FileNotFoundError as e:
        print(f"  ⚠ Error: {e}")

    # Compare with later loop
    if info['latest_loop'] >= 1:
        print(f"\n  Loading data from loop {min(5, info['latest_loop'])} for comparison:")
        try:
            compare_loop = min(5, info['latest_loop'])
            data_later = load_data_from_loop(checkpoint_dir, loop=compare_loop, oracle_name=oracle_name)

            print(f"    Number of samples: {data_later['n_samples']}")
            print(f"    X range: [{data_later['X'].min():.4f}, {data_later['X'].max():.4f}]")
            print(f"    Y range: [{data_later['Y'].min():.4f}, {data_later['Y'].max():.4f}]")

            # Compare Y values
            print(f"\n    Comparison:")
            print(f"      Loop 0 Y mean: {data_loop0['Y'].mean():.6f}")
            print(f"      Loop {compare_loop} Y mean: {data_later['Y'].mean():.6f}")
            improvement = (data_later['Y'].mean() - data_loop0['Y'].mean()) / abs(data_loop0['Y'].mean()) * 100
            print(f"      Change: {improvement:+.2f}%")

        except FileNotFoundError:
            pass

    # ========================================================================
    # Part 4: Load accumulated data
    # ========================================================================
    print("\n" + "-"*70)
    print("Part 4: Load Accumulated Data")
    print("-"*70)

    # Load all data up to latest loop
    latest = min(info['latest_loop'], 10)  # Cap at 10 for this example
    print(f"\n  Loading accumulated data up to loop {latest}:")

    try:
        acc_data = load_accumulated_data(checkpoint_dir, up_to_loop=latest, oracle_name=oracle_name)

        print(f"    Total samples: {acc_data['n_samples']}")
        print(f"    X shape: {acc_data['X'].shape}")
        print(f"    Y shape: {acc_data['Y'].shape}")

        print(f"\n    Samples per loop:")
        for loop, count in sorted(acc_data['samples_per_loop'].items()):
            print(f"      Loop {loop}: {count} samples")

        # Analyze loop indices
        print(f"\n    Loop distribution from metadata:")
        unique_loops, counts = np.unique(acc_data['loop_indices'], return_counts=True)
        for loop, count in zip(unique_loops, counts):
            print(f"      Loop {loop}: {count} samples")

        # Find best samples
        best_idx = np.argmin(acc_data['Y'][:, 0])  # Assuming minimization
        print(f"\n    Best sample (overall):")
        print(f"      Index: {best_idx}")
        print(f"      From loop: {acc_data['loop_indices'][best_idx]}")
        print(f"      X: {acc_data['X'][best_idx]}")
        print(f"      Y: {acc_data['Y'][best_idx]}")

    except FileNotFoundError as e:
        print(f"  ⚠ Error: {e}")

    # ========================================================================
    # Part 5: Test algorithm with historical models
    # ========================================================================
    print("\n" + "-"*70)
    print("Part 5: Test Algorithm with Historical Models")
    print("-"*70)

    print(f"\n  This demonstrates how to test your algorithm's")
    print(f"  proposals using models from different loops.")

    try:
        # Load models from a specific loop
        result = load_models_for_analysis(checkpoint_dir, loop=min(3, info['latest_loop']))

        print(f"\n  Loaded models from loop {result['loop']}")

        # Example: Create test algorithm (replace with your actual algorithm)
        print(f"\n  Example usage:")
        print(f"  ```python")
        print(f"  # Load your algorithm class")
        print(f"  from your_module import Algorithm")
        print(f"  ")
        print(f"  # Create algorithm instance")
        print(f"  algorithm = Algorithm(n_propose=[10, 20], ...)")
        print(f"  ")
        print(f"  # Load models from loop 5")
        print(f"  result = load_models_for_analysis('checkpoints/', loop=5)")
        print(f"  fn_pred_list = result['fn_pred_list']")
        print(f"  ")
        print(f"  # Test algorithm proposal")
        print(f"  X_proposed = algorithm.propose(fn_pred_list)")
        print(f"  ")
        print(f"  # Examine proposals")
        print(f"  for i, X in enumerate(X_proposed):")
        print(f"      print(f'Oracle {{i}}: {{X.shape}} samples')")
        print(f"      print(f'  Range: {{X.min():.3f}} to {{X.max():.3f}}')")
        print(f"  ```")

    except FileNotFoundError:
        print(f"  ⚠ Not enough loops yet for this demonstration")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Summary: Checkpoint Analysis Utilities")
    print("="*70)

    print(f"\n1. inspect_checkpoint(): Quick overview of available data")
    print(f"2. load_models_for_analysis(): Load models for testing/analysis")
    print(f"3. load_data_from_loop(): Load data from a single specific loop")
    print(f"4. load_accumulated_data(): Load all data up to a loop")

    print(f"\nThese utilities enable:")
    print(f"  ✓ Post-optimization analysis")
    print(f"  ✓ Testing algorithms with historical models")
    print(f"  ✓ Debugging optimization issues")
    print(f"  ✓ Understanding data collection patterns")
    print(f"  ✓ Analyzing convergence behavior")

    print(f"\n" + "="*70)


if __name__ == '__main__':
    main()
