"""Tests for Checkpoint Manager."""

import numpy as np
import tempfile
import shutil
from pathlib import Path

from mpbax.core.checkpoint import CheckpointManager
from mpbax.core.data_handler import DataHandler
from mpbax.core.model import DummyModel


def test_save_and_load():
    """Test basic save and load functionality."""
    print("Testing checkpoint save and load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        # Create test data and models for 2 objectives
        data_handlers = []
        models = []
        obj_names = ["obj_0", "obj_1"]

        for i in range(2):
            # Data handler with some data
            dh = DataHandler(input_dim=2)
            X = np.array([[1.0 * i, 2.0 * i], [3.0 * i, 4.0 * i]])
            Y = np.array([[5.0 * i], [6.0 * i]])
            dh.add_data(X, Y)
            data_handlers.append(dh)

            # Model
            model = DummyModel(input_dim=2)
            model.train(X, Y)
            models.append(model)

        # Config
        config = {"seed": 42, "max_loops": 10}

        # Save checkpoint for loop 0
        manager.save_checkpoint(
            loop=0,
            data_handlers=data_handlers,
            models=models,
            config=config,
            obj_names=obj_names
        )

        # Verify checkpoint was saved
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "config.yaml").exists()
        assert (checkpoint_dir / "state.pkl").exists()
        assert (checkpoint_dir / "obj_0" / "data_0.pkl").exists()
        assert (checkpoint_dir / "obj_0" / "model_0.pkl").exists()
        assert (checkpoint_dir / "obj_1" / "data_0.pkl").exists()
        assert (checkpoint_dir / "obj_1" / "model_0.pkl").exists()

        # Load checkpoint
        loop, loaded_dhs, loaded_models, loaded_config, loaded_names = manager.load_checkpoint()

        assert loop == 0
        assert len(loaded_dhs) == 2
        assert len(loaded_models) == 2
        assert loaded_config == config
        assert loaded_names == obj_names

        # Verify data
        for i, dh in enumerate(loaded_dhs):
            X, Y = dh.get_data()
            assert X.shape == (2, 2)
            assert Y.shape == (2, 1)

    print("Checkpoint save/load test passed!\n")


def test_multiple_loops():
    """Test saving and loading multiple loops."""
    print("Testing multiple loop checkpoints...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        obj_names = ["obj_0"]
        config = {"seed": 42}

        # Simulate 3 loops
        for loop in range(3):
            # Create data handler with incremental data
            dh = DataHandler(input_dim=2)
            X = np.array([[float(loop), float(loop)]])
            Y = np.array([[float(loop * 10)]])
            dh.add_data(X, Y)

            # Create model
            model = DummyModel(input_dim=2)
            model.train(X, Y)

            # Save checkpoint
            manager.save_checkpoint(
                loop=loop,
                data_handlers=[dh],
                models=[model],
                config=config,
                obj_names=obj_names
            )

        # Get latest loop
        latest = manager.get_latest_loop()
        assert latest == 2

        # List all checkpoints
        loops = manager.list_checkpoints()
        assert loops == [0, 1, 2]

        # Load from loop 1
        loop, dhs, models, _, _ = manager.load_checkpoint(loop=1)
        assert loop == 1

        # Should have accumulated data from loops 0 and 1
        X, Y = dhs[0].get_data()
        assert X.shape[0] == 2  # 2 data points total
        assert Y.shape[0] == 2

        # Load latest (loop 2)
        loop, dhs, models, _, _ = manager.load_checkpoint()
        assert loop == 2

        # Should have accumulated data from loops 0, 1, and 2
        X, Y = dhs[0].get_data()
        assert X.shape[0] == 3  # 3 data points total

    print("Multiple loop checkpoint test passed!\n")


def test_rollback():
    """Test rollback functionality."""
    print("Testing checkpoint rollback...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        obj_names = ["obj_0"]
        config = {"seed": 42}

        # Create checkpoints for loops 0, 1, 2
        for loop in range(3):
            dh = DataHandler(input_dim=2)
            X = np.array([[float(loop), float(loop)]])
            Y = np.array([[float(loop)]])
            dh.add_data(X, Y)

            model = DummyModel(input_dim=2)
            model.train(X, Y)

            manager.save_checkpoint(
                loop=loop,
                data_handlers=[dh],
                models=[model],
                config=config,
                obj_names=obj_names
            )

        # Verify we have 3 checkpoints
        assert manager.list_checkpoints() == [0, 1, 2]

        # Rollback to loop 1 (delete loop 2)
        manager.delete_checkpoints_after(loop=1)

        # Verify loop 2 is deleted
        assert manager.list_checkpoints() == [0, 1]
        assert manager.get_latest_loop() == 1

        # Verify files are actually deleted
        obj_0_dir = checkpoint_dir / "obj_0"
        assert not (obj_0_dir / "data_2.pkl").exists()
        assert not (obj_0_dir / "model_2.pkl").exists()

    print("Rollback test passed!\n")


def test_multi_objective():
    """Test checkpointing with multiple objectives."""
    print("Testing multi-objective checkpointing...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir))

        obj_names = ["objective_1", "objective_2", "objective_3"]
        config = {"seed": 42, "n_objectives": 3}

        # Create data and models for 3 objectives
        data_handlers = []
        models = []

        for i in range(3):
            dh = DataHandler(input_dim=i + 2)  # Different dimensions
            X = np.random.rand(5, i + 2)
            Y = np.random.rand(5, 1)
            dh.add_data(X, Y)
            data_handlers.append(dh)

            model = DummyModel(input_dim=i + 2)
            model.train(X, Y)
            models.append(model)

        # Save checkpoint
        manager.save_checkpoint(
            loop=0,
            data_handlers=data_handlers,
            models=models,
            config=config,
            obj_names=obj_names
        )

        # Load and verify
        loop, loaded_dhs, loaded_models, loaded_config, loaded_names = manager.load_checkpoint()

        assert len(loaded_dhs) == 3
        assert len(loaded_models) == 3
        assert loaded_names == obj_names

        # Verify each objective has correct dimension
        for i, dh in enumerate(loaded_dhs):
            assert dh.input_dim == i + 2
            X, Y = dh.get_data()
            assert X.shape == (5, i + 2)

    print("Multi-objective checkpoint test passed!\n")


if __name__ == "__main__":
    test_save_and_load()
    test_multiple_loops()
    test_rollback()
    test_multi_objective()
    print("All checkpoint tests passed!")
