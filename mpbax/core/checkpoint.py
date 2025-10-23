"""Checkpoint manager for saving and loading run state."""

import os
import pickle
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mpbax.core.data_handler import DataHandler
from mpbax.core.model import BaseModel


def _sanitize_oracle_name(name: str) -> str:
    """Sanitize oracle name for use as directory name.

    Replaces spaces and special characters with underscores.
    Ensures the name is safe to use as a filesystem path.

    Args:
        name: Oracle name from config

    Returns:
        Sanitized name safe for filesystem use

    Example:
        >>> _sanitize_oracle_name("My Oracle #1")
        'My_Oracle_1'
    """
    # Replace spaces and non-alphanumeric chars (except hyphen) with underscore
    sanitized = re.sub(r'[^\w\-]', '_', name)
    # Remove duplicate underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


class CheckpointManager:
    """Manages checkpointing for optimization runs.

    Checkpoint structure:
    - checkpoint_dir/
      - config.yaml          # Run configuration
      - state.pkl            # Run state (current loop, seed, etc.)
      - oracle_0/            # Oracle 0
        - data_0.pkl
        - data_1.pkl
        - ...
        - model_0.pkl
        - model_1.pkl
        - ...
      - oracle_1/            # Oracle 1
        - ...
    """

    def __init__(self, checkpoint_dir: str):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to save/load checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)

    def save_checkpoint(
        self,
        loop: int,
        data_handlers: List[DataHandler],
        models: List[BaseModel],
        config: Dict,
        oracle_names: List[str]
    ) -> None:
        """Save checkpoint for current loop.

        Args:
            loop: Current loop number
            data_handlers: List of DataHandler instances (one per oracle)
            models: List of Model instances (one per oracle)
            config: Configuration dictionary
            oracle_names: List of oracle names
        """
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config (only once)
        config_path = self.checkpoint_dir / "config.yaml"
        if not config_path.exists():
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

        # Save state
        state = {
            'current_loop': loop,
            'n_oracles': len(oracle_names),
            'oracle_names': oracle_names
        }
        state_path = self.checkpoint_dir / "state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

        # Get checkpoint mode from config
        checkpoint_mode = config.get('model', {}).get('checkpoint_mode', 'final')

        # Save data and models for each oracle
        for i, (data_handler, model, oracle_name) in enumerate(zip(data_handlers, models, oracle_names)):
            oracle_name_sanitized = _sanitize_oracle_name(oracle_name)
            oracle_dir = self.checkpoint_dir / oracle_name_sanitized
            oracle_dir.mkdir(exist_ok=True)

            # Save data_loop
            data_path = oracle_dir / f"data_{loop}.pkl"
            data_handler.save(str(data_path))

            # Save models based on checkpoint_mode
            if checkpoint_mode in ['final', 'both']:
                # Save final model (used for resumption)
                model_path = oracle_dir / f"model_{loop}_final.pkl"
                model.save(str(model_path))

            if checkpoint_mode in ['best', 'both']:
                # Save best model if available
                best_model = model.get_best_model_snapshot()
                if best_model is not None:
                    best_model_path = oracle_dir / f"model_{loop}_best.pkl"
                    best_model.save(str(best_model_path))

    def get_latest_loop(self) -> Optional[int]:
        """Get the latest completed loop number.

        Returns:
            Latest loop number, or None if no checkpoints exist
        """
        state_path = self.checkpoint_dir / "state.pkl"
        if not state_path.exists():
            return None

        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        return state['current_loop']

    def load_checkpoint(
        self,
        loop: Optional[int] = None
    ) -> Tuple[int, List[DataHandler], List[BaseModel], Dict, List[str]]:
        """Load checkpoint from specified or latest loop.

        Args:
            loop: Loop number to load from. If None, loads latest.

        Returns:
            Tuple of (loop, data_handlers, models, config, oracle_names)

        Raises:
            ValueError: If checkpoint doesn't exist
        """
        # Check if checkpoint directory exists
        if not self.checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory {self.checkpoint_dir} does not exist")

        # Load state to get oracle info
        state_path = self.checkpoint_dir / "state.pkl"
        if not state_path.exists():
            raise ValueError("No checkpoint state found")

        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        # Determine which loop to load
        if loop is None:
            loop = state['current_loop']

        # Load config
        config_path = self.checkpoint_dir / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load data and models for each oracle
        n_oracles = state['n_oracles']
        oracle_names = state['oracle_names']
        data_handlers = []
        models = []

        for i in range(n_oracles):
            # Try oracle name first, fall back to oracle_{i} for backward compatibility
            oracle_name_sanitized = _sanitize_oracle_name(oracle_names[i])
            oracle_dir = self.checkpoint_dir / oracle_name_sanitized

            if not oracle_dir.exists():
                # Backward compatibility: try old naming scheme
                oracle_dir = self.checkpoint_dir / f"oracle_{i}"
                if not oracle_dir.exists():
                    raise ValueError(
                        f"Oracle directory not found: {oracle_name_sanitized} or oracle_{i}"
                    )

            # Load all data from 0 to loop
            data_handler_combined = None
            for data_loop in range(loop + 1):
                data_path = oracle_dir / f"data_{data_loop}.pkl"
                if not data_path.exists():
                    raise ValueError(f"Data file {data_path} not found")

                data_handler = DataHandler.load(str(data_path))

                if data_handler_combined is None:
                    data_handler_combined = data_handler
                else:
                    # Combine data
                    X, Y = data_handler.get_data()
                    if X is not None:
                        data_handler_combined.add_data(X, Y)

            data_handlers.append(data_handler_combined)

            # Load model for this loop
            # Try new format first (model_loop_final.pkl), fallback to old (model_loop.pkl)
            model_path_final = oracle_dir / f"model_{loop}_final.pkl"
            model_path_old = oracle_dir / f"model_{loop}.pkl"

            if model_path_final.exists():
                model = BaseModel.load(str(model_path_final))
            elif model_path_old.exists():
                model = BaseModel.load(str(model_path_old))
            else:
                raise ValueError(f"Model file not found: {model_path_final} or {model_path_old}")

            models.append(model)

        return loop, data_handlers, models, config, oracle_names

    def list_checkpoints(self) -> List[int]:
        """List all available checkpoint loop numbers.

        Returns:
            List of loop numbers with checkpoints
        """
        if not self.checkpoint_dir.exists():
            return []

        # Try to get oracle directory - first try from state, then fall back to oracle_0
        oracle_dir = None
        state_path = self.checkpoint_dir / "state.pkl"
        if state_path.exists():
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            if 'oracle_names' in state and state['oracle_names']:
                oracle_name_sanitized = _sanitize_oracle_name(state['oracle_names'][0])
                oracle_dir = self.checkpoint_dir / oracle_name_sanitized

        # Backward compatibility: fall back to oracle_0
        if oracle_dir is None or not oracle_dir.exists():
            oracle_dir = self.checkpoint_dir / "oracle_0"

        if not oracle_dir.exists():
            return []

        # Get loops from first oracle's data files
        loops = []
        for data_file in sorted(oracle_dir.glob("data_*.pkl")):
            loop_num = int(data_file.stem.split('_')[1])
            loops.append(loop_num)

        return sorted(loops)

    def delete_checkpoints_after(self, loop: int) -> None:
        """Delete all checkpoints after specified loop.

        Useful for rollback operations.

        Args:
            loop: Keep checkpoints up to and including this loop
        """
        state_path = self.checkpoint_dir / "state.pkl"
        if not state_path.exists():
            return

        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        n_oracles = state['n_oracles']
        oracle_names = state.get('oracle_names', [])

        # Delete data and model files after specified loop
        for i in range(n_oracles):
            # Try oracle name first, fall back to oracle_{i}
            if i < len(oracle_names):
                oracle_name_sanitized = _sanitize_oracle_name(oracle_names[i])
                oracle_dir = self.checkpoint_dir / oracle_name_sanitized
                if not oracle_dir.exists():
                    # Backward compatibility
                    oracle_dir = self.checkpoint_dir / f"oracle_{i}"
            else:
                oracle_dir = self.checkpoint_dir / f"oracle_{i}"

            if not oracle_dir.exists():
                continue

            for data_file in oracle_dir.glob(f"data_*.pkl"):
                loop_num = int(data_file.stem.split('_')[1])
                if loop_num > loop:
                    data_file.unlink()

            for model_file in oracle_dir.glob(f"model_*.pkl"):
                loop_num = int(model_file.stem.split('_')[1])
                if loop_num > loop:
                    model_file.unlink()

        # Update state
        state['current_loop'] = loop
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
