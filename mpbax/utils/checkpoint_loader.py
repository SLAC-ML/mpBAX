"""Utilities for loading models and data from mpBAX checkpoints for analysis.

This module provides clean interfaces for post-optimization analysis:
- Load trained models from specific loops
- Get fn_pred_list for testing with algorithms
- Load data from specific loops or accumulated data
- Examine optimization progress

Example:
    >>> from mpbax.utils import load_models_for_analysis, load_data_from_loop
    >>>
    >>> # Load models from loop 5
    >>> result = load_models_for_analysis('checkpoints/', loop=5)
    >>> fn_pred_list = result['fn_pred_list']  # For Algorithm.propose()
    >>> models = result['models']              # For inspection
    >>>
    >>> # Load data from loop 5 for first oracle
    >>> data = load_data_from_loop('checkpoints/', loop=5, oracle_idx=0)
    >>> X, Y = data['X'], data['Y']
"""

from pathlib import Path
from typing import Union, Optional, Dict, List, Any
import pickle

from ..core.checkpoint import CheckpointManager
from ..core.data_handler import DataHandler


def load_models_for_analysis(
    checkpoint_dir: Union[str, Path],
    loop: Optional[Union[int, str]] = None,
    mode: str = 'final'
) -> Dict[str, Any]:
    """Load trained models from a specific checkpoint loop.

    This function loads models exactly as the Engine does during training,
    providing fn_pred_list that can be used with Algorithm.propose().

    Args:
        checkpoint_dir: Path to checkpoint directory
        loop: Loop number to load. Options:
            - int: Specific loop number (0, 1, 2, ...)
            - 'latest'/None: Load from latest checkpoint
        mode: Which model to load:
            - 'final': model_{loop}_final.pkl (default)
            - 'best': model_{loop}_best.pkl (if available)

    Returns:
        Dictionary with keys:
            - 'models': List of loaded model instances (can inspect/modify)
            - 'fn_pred_list': List of predict functions for Algorithm.propose()
            - 'oracle_names': List of oracle names
            - 'loop': Loop number that was loaded
            - 'config': Configuration dict used for the run

    Example:
        >>> result = load_models_for_analysis('run_0_da/', loop=5)
        >>> fn_pred_list = result['fn_pred_list']
        >>>
        >>> # Test with algorithm
        >>> X_proposed = my_algorithm.propose(fn_pred_list)
        >>>
        >>> # Or test models directly
        >>> X_test = np.random.rand(10, 2)
        >>> Y_pred = result['models'][0].predict(X_test)

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist
        ValueError: If specified loop doesn't exist
    """
    manager = CheckpointManager(checkpoint_dir)

    # Load checkpoint (returns accumulated data, but we only need models)
    loop_num, data_handlers, models, config, oracle_names = manager.load_checkpoint(
        loop=loop
    )

    # Note: checkpoint_mode is handled differently - models are loaded based on what exists
    # If you need 'best' models specifically, you would need to load them manually

    # Create fn_pred_list exactly as Engine does (line 150 in engine.py)
    fn_pred_list = [model.predict for model in models]

    return {
        'models': models,
        'fn_pred_list': fn_pred_list,
        'oracle_names': oracle_names,
        'loop': loop_num,
        'config': config
    }


def load_data_from_loop(
    checkpoint_dir: Union[str, Path],
    loop: int,
    oracle_idx: Optional[int] = None,
    oracle_name: Optional[str] = None
) -> Dict[str, Any]:
    """Load data from a SINGLE specific loop (not accumulated).

    This is useful for examining what was collected during a particular
    loop iteration.

    Args:
        checkpoint_dir: Path to checkpoint directory
        loop: Loop number to load (0, 1, 2, ...)
        oracle_idx: Index of oracle (0, 1, ...). Specify this OR oracle_name
        oracle_name: Name of oracle. Specify this OR oracle_idx

    Returns:
        Dictionary with keys:
            - 'X': Input data from this loop only, shape (n, d)
            - 'Y': Output data from this loop only, shape (n, k)
            - 'loop_indices': Loop indices (all equal to loop number)
            - 'oracle_name': Name of the oracle
            - 'n_samples': Number of samples in this loop

    Example:
        >>> # Load data from loop 5 for 'chrom' oracle
        >>> data = load_data_from_loop('run_0_da/', loop=5, oracle_name='chrom')
        >>> print(f"Loop 5 collected {data['n_samples']} samples")
        >>> print(f"X range: {data['X'].min():.3f} to {data['X'].max():.3f}")

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If neither or both oracle_idx and oracle_name specified
    """
    if (oracle_idx is None) == (oracle_name is None):
        raise ValueError("Specify exactly one of oracle_idx or oracle_name")

    checkpoint_dir = Path(checkpoint_dir)

    # Get oracle name
    if oracle_name is None:
        # Load config to get oracle name from index
        manager = CheckpointManager(checkpoint_dir)
        _, _, _, config, oracle_names = manager.load_checkpoint(resume_from=0)
        oracle_name = oracle_names[oracle_idx]

    # Load data file for this specific loop
    data_file = checkpoint_dir / oracle_name / f'data_{loop}.pkl'

    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Loop {loop} may not exist or oracle name '{oracle_name}' is incorrect."
        )

    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    X = data_dict['X']
    Y = data_dict['Y']
    loop_indices = data_dict.get('loop_indices', None)

    return {
        'X': X,
        'Y': Y,
        'loop_indices': loop_indices,
        'oracle_name': oracle_name,
        'n_samples': X.shape[0]
    }


def load_accumulated_data(
    checkpoint_dir: Union[str, Path],
    up_to_loop: int,
    oracle_idx: Optional[int] = None,
    oracle_name: Optional[str] = None
) -> Dict[str, Any]:
    """Load accumulated data from loop 0 to specified loop.

    This loads ALL data from the start up to and including the specified loop,
    exactly as the Engine uses for model training.

    Args:
        checkpoint_dir: Path to checkpoint directory
        up_to_loop: Load data up to and including this loop
        oracle_idx: Index of oracle. Specify this OR oracle_name
        oracle_name: Name of oracle. Specify this OR oracle_idx

    Returns:
        Dictionary with keys:
            - 'X': Accumulated input data, shape (n_total, d)
            - 'Y': Accumulated output data, shape (n_total, k)
            - 'loop_indices': Loop index for each sample
            - 'oracle_name': Name of the oracle
            - 'n_samples': Total number of samples
            - 'samples_per_loop': Dict mapping loop -> sample count

    Example:
        >>> # Load all data up to loop 10
        >>> data = load_accumulated_data('run_0_da/', up_to_loop=10, oracle_idx=0)
        >>> print(f"Total samples: {data['n_samples']}")
        >>> print(f"Samples per loop: {data['samples_per_loop']}")
        >>>
        >>> # Analyze which loop contributed most
        >>> import numpy as np
        >>> unique, counts = np.unique(data['loop_indices'], return_counts=True)
        >>> for loop, count in zip(unique, counts):
        ...     print(f"Loop {loop}: {count} samples")

    Raises:
        FileNotFoundError: If checkpoint directory doesn't exist
        ValueError: If neither or both oracle_idx and oracle_name specified
    """
    if (oracle_idx is None) == (oracle_name is None):
        raise ValueError("Specify exactly one of oracle_idx or oracle_name")

    checkpoint_dir = Path(checkpoint_dir)

    # Get oracle name if needed
    if oracle_name is None:
        manager = CheckpointManager(checkpoint_dir)
        _, _, _, config, oracle_names = manager.load_checkpoint(loop=0)
        oracle_name = oracle_names[oracle_idx]

    # Load first data file to get dimensions
    data_file_0 = checkpoint_dir / oracle_name / 'data_0.pkl'
    if not data_file_0.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_0}")

    with open(data_file_0, 'rb') as f:
        data_dict_0 = pickle.load(f)
    input_dim = data_dict_0['X'].shape[1]

    # Create data handler with correct input_dim
    data_handler = DataHandler(input_dim=input_dim)

    samples_per_loop = {}
    for loop in range(up_to_loop + 1):
        data_file = checkpoint_dir / oracle_name / f'data_{loop}.pkl'

        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Expected continuous loops from 0 to {up_to_loop}"
            )

        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f)

        X = data_dict['X']
        Y = data_dict['Y']
        loop_indices = data_dict.get('loop_indices', None)

        # Add to data handler
        if loop_indices is not None and len(loop_indices) > 0:
            # Use the loop index from the data
            loop_num = loop_indices[0]
        else:
            # Fallback to using the file's loop number
            loop_num = loop

        data_handler.add_data(X, Y, loop=loop_num)
        samples_per_loop[loop] = X.shape[0]

    # Get accumulated data
    X_all, Y_all, metadata = data_handler.get_data_with_metadata()

    return {
        'X': X_all,
        'Y': Y_all,
        'loop_indices': metadata['loop_indices'],
        'oracle_name': oracle_name,
        'n_samples': X_all.shape[0],
        'samples_per_loop': samples_per_loop
    }


# Convenience function for quick checkpoint inspection
def inspect_checkpoint(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    """Quick inspection of a checkpoint directory.

    Provides summary information about available checkpoints without
    loading the full data or models.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Dictionary with keys:
            - 'latest_loop': Latest available loop number
            - 'oracle_names': List of oracle names
            - 'oracle_dirs': Dict mapping oracle name -> directory path
            - 'available_loops': Dict mapping oracle name -> list of loops
            - 'config': Configuration dict (if available)

    Example:
        >>> info = inspect_checkpoint('run_0_da/')
        >>> print(f"Latest loop: {info['latest_loop']}")
        >>> print(f"Oracles: {info['oracle_names']}")
        >>> for oracle, loops in info['available_loops'].items():
        ...     print(f"  {oracle}: loops {min(loops)}-{max(loops)}")
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find oracle directories
    oracle_dirs = {}
    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            oracle_dirs[subdir.name] = subdir

    oracle_names = list(oracle_dirs.keys())

    # Find available loops for each oracle
    available_loops = {}
    for oracle_name, oracle_dir in oracle_dirs.items():
        loops = []
        for data_file in oracle_dir.glob('data_*.pkl'):
            loop_num = int(data_file.stem.split('_')[1])
            loops.append(loop_num)
        available_loops[oracle_name] = sorted(loops)

    # Determine latest loop (minimum across all oracles)
    latest_loop = min(max(loops) if loops else 0 for loops in available_loops.values())

    # Try to load config
    config = None
    config_file = checkpoint_dir / 'config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

    return {
        'latest_loop': latest_loop,
        'oracle_names': oracle_names,
        'oracle_dirs': oracle_dirs,
        'available_loops': available_loops,
        'config': config
    }
