"""Checkpoint manager for saving and loading run state."""

import os
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mpbax.core.data_handler import DataHandler
from mpbax.core.model import BaseModel


class CheckpointManager:
    """Manages checkpointing for optimization runs.

    Checkpoint structure:
    - checkpoint_dir/
      - config.yaml          # Run configuration
      - state.pkl            # Run state (current loop, seed, etc.)
      - obj_0/               # Objective 0
        - data_0.pkl
        - data_1.pkl
        - ...
        - model_0.pkl
        - model_1.pkl
        - ...
      - obj_1/               # Objective 1
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
        obj_names: List[str]
    ) -> None:
        """Save checkpoint for current loop.

        Args:
            loop: Current loop number
            data_handlers: List of DataHandler instances (one per objective)
            models: List of Model instances (one per objective)
            config: Configuration dictionary
            obj_names: List of objective names
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
            'n_objectives': len(obj_names),
            'obj_names': obj_names
        }
        state_path = self.checkpoint_dir / "state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

        # Save data and models for each objective
        for i, (data_handler, model, obj_name) in enumerate(zip(data_handlers, models, obj_names)):
            obj_dir = self.checkpoint_dir / f"obj_{i}"
            obj_dir.mkdir(exist_ok=True)

            # Save data_loop
            data_path = obj_dir / f"data_{loop}.pkl"
            data_handler.save(str(data_path))

            # Save model_loop
            model_path = obj_dir / f"model_{loop}.pkl"
            model.save(str(model_path))

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
            Tuple of (loop, data_handlers, models, config, obj_names)

        Raises:
            ValueError: If checkpoint doesn't exist
        """
        # Check if checkpoint directory exists
        if not self.checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory {self.checkpoint_dir} does not exist")

        # Load state to get objective info
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

        # Load data and models for each objective
        n_objectives = state['n_objectives']
        obj_names = state['obj_names']
        data_handlers = []
        models = []

        for i in range(n_objectives):
            obj_dir = self.checkpoint_dir / f"obj_{i}"

            # Load all data from 0 to loop
            data_handler_combined = None
            for data_loop in range(loop + 1):
                data_path = obj_dir / f"data_{data_loop}.pkl"
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
            model_path = obj_dir / f"model_{loop}.pkl"
            if not model_path.exists():
                raise ValueError(f"Model file {model_path} not found")

            model = BaseModel.load(str(model_path))
            models.append(model)

        return loop, data_handlers, models, config, obj_names

    def list_checkpoints(self) -> List[int]:
        """List all available checkpoint loop numbers.

        Returns:
            List of loop numbers with checkpoints
        """
        if not self.checkpoint_dir.exists():
            return []

        # Get loops from first objective's data files
        obj_0_dir = self.checkpoint_dir / "obj_0"
        if not obj_0_dir.exists():
            return []

        loops = []
        for data_file in sorted(obj_0_dir.glob("data_*.pkl")):
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

        n_objectives = state['n_objectives']

        # Delete data and model files after specified loop
        for i in range(n_objectives):
            obj_dir = self.checkpoint_dir / f"obj_{i}"
            if not obj_dir.exists():
                continue

            for data_file in obj_dir.glob(f"data_*.pkl"):
                loop_num = int(data_file.stem.split('_')[1])
                if loop_num > loop:
                    data_file.unlink()

            for model_file in obj_dir.glob(f"model_*.pkl"):
                loop_num = int(model_file.stem.split('_')[1])
                if loop_num > loop:
                    model_file.unlink()

        # Update state
        state['current_loop'] = loop
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
