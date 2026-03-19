"""Tests for CheckpointManager."""

import numpy as np
import tempfile
from pathlib import Path

from mpbax.core.checkpoint import CheckpointManager
from mpbax.core.data_handler import DataHandler
from mpbax.core.model import DummyModel


def test_basic_save_load():
    """Test basic checkpoint save and load."""
    print("Testing basic save and load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Create mock data for 2 oracles
        dh1 = DataHandler(input_dim=2)
        dh1.add_data(np.random.rand(10, 2), np.random.rand(10, 1), loop=0)

        dh2 = DataHandler(input_dim=3)
        dh2.add_data(np.random.rand(10, 3), np.random.rand(10, 1), loop=0)

        model1 = DummyModel(input_dim=2)
        model2 = DummyModel(input_dim=3)

        config = {'seed': 42, 'max_loops': 10}

        # Save checkpoint
        manager.save_checkpoint(
            loop=0,
            data_handlers=[dh1, dh2],
            models=[model1, model2],
            config=config,
            oracle_names=['oracle1', 'oracle2']
        )

        # Load checkpoint
        loop, dh_loaded, models_loaded, config_loaded, names = manager.load_checkpoint(loop=0)

        assert loop == 0
        assert len(dh_loaded) == 2
        assert len(models_loaded) == 2
        assert config_loaded['seed'] == 42
        assert names == ['oracle1', 'oracle2']

    print("  ✓ basic save/load passed")


def test_multiple_loops():
    """Test saving and loading multiple loop checkpoints."""
    print("Testing multiple loops...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Save checkpoints for loops 0, 1, 2
        for loop_idx in range(3):
            dh = DataHandler(input_dim=2)
            dh.add_data(
                np.random.rand(10, 2),
                np.random.rand(10, 1),
                loop=loop_idx
            )

            model = DummyModel(input_dim=2)
            config = {'seed': 42, 'max_loops': 10}

            manager.save_checkpoint(
                loop=loop_idx,
                data_handlers=[dh],
                models=[model],
                config=config,
                oracle_names=['test']
            )

        # List checkpoints
        loops = manager.list_checkpoints()
        assert loops == [0, 1, 2]

        # Get latest
        latest = manager.get_latest_loop()
        assert latest == 2

        # Load specific loop
        loop, dh_loaded, _, _, _ = manager.load_checkpoint(loop=1)
        assert loop == 1

    print("  ✓ multiple loops passed")


def test_rollback():
    """Test checkpoint rollback functionality."""
    print("Testing rollback...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Create checkpoints for loops 0, 1, 2, 3
        for loop_idx in range(4):
            dh = DataHandler(input_dim=2)
            dh.add_data(np.random.rand(10, 2), np.random.rand(10, 1), loop=loop_idx)

            model = DummyModel(input_dim=2)
            config = {'seed': 42}

            manager.save_checkpoint(
                loop=loop_idx,
                data_handlers=[dh],
                models=[model],
                config=config,
                oracle_names=['test']
            )

        # Verify we have 4 checkpoints
        assert manager.list_checkpoints() == [0, 1, 2, 3]

        # Rollback to loop 1 (delete loops 2 and 3)
        manager.delete_checkpoints_after(loop=1)

        # Verify only loops 0 and 1 remain
        assert manager.list_checkpoints() == [0, 1]
        assert manager.get_latest_loop() == 1

        # Verify files are actually deleted
        oracle_dir = Path(tmpdir) / "test"
        assert not (oracle_dir / "data_2.pkl").exists()
        assert not (oracle_dir / "data_3.pkl").exists()

    print("  ✓ rollback passed")


def test_oracle_naming():
    """Test checkpoint directory naming from oracle names."""
    print("Testing oracle naming...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        dh = DataHandler(input_dim=2)
        dh.add_data(np.random.rand(10, 2), np.random.rand(10, 1), loop=0)

        model = DummyModel(input_dim=2)
        config = {'seed': 42}

        # Use custom oracle name
        oracle_name = "my_custom_oracle"
        manager.save_checkpoint(
            loop=0,
            data_handlers=[dh],
            models=[model],
            config=config,
            oracle_names=[oracle_name]
        )

        # Verify directory was created with oracle name
        oracle_dir = Path(tmpdir) / oracle_name
        assert oracle_dir.exists()
        assert (oracle_dir / "data_0.pkl").exists()

    print("  ✓ oracle naming passed")


def test_load_latest():
    """Test loading latest checkpoint."""
    print("Testing load latest...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        # Create checkpoints for loops 0, 1, 2
        for loop_idx in range(3):
            dh = DataHandler(input_dim=2)
            dh.add_data(np.random.rand(5, 2), np.random.rand(5, 1), loop=loop_idx)

            model = DummyModel(input_dim=2)
            config = {'seed': 42}

            manager.save_checkpoint(
                loop=loop_idx,
                data_handlers=[dh],
                models=[model],
                config=config,
                oracle_names=['test']
            )

        # Load latest (should be loop 2)
        loop, _, _, _, _ = manager.load_checkpoint()
        assert loop == 2

        # Load latest explicitly
        loop2, _, _, _, _ = manager.load_checkpoint(loop='latest')
        assert loop2 == 2

    print("  ✓ load latest passed")


def test_data_always_saved():
    """Test that save_data and save_models can be called independently."""
    print("Testing save_data/save_models split...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        config = {'seed': 42}

        # Save data for loops 0, 1, 2 but models only for 0 and 2
        for loop_idx in range(3):
            dh = DataHandler(input_dim=2)
            dh.add_data(np.random.rand(5, 2), np.random.rand(5, 1), loop=loop_idx)
            model = DummyModel(input_dim=2)

            # Always save data
            manager.save_data(loop_idx, [dh], config, ['test'])

            # Only save model on even loops (simulating freq=2)
            if loop_idx % 2 == 0:
                manager.save_models(loop_idx, [model], config, ['test'])

        # Verify all data files exist
        oracle_dir = Path(tmpdir) / "test"
        assert (oracle_dir / "data_0.pkl").exists()
        assert (oracle_dir / "data_1.pkl").exists()
        assert (oracle_dir / "data_2.pkl").exists()

        # Verify model files only at freq intervals
        assert (oracle_dir / "model_0_final.pkl").exists()
        assert not (oracle_dir / "model_1_final.pkl").exists()
        assert (oracle_dir / "model_2_final.pkl").exists()

        # state.pkl should track latest model checkpoint (loop 2)
        assert manager.get_latest_loop() == 2

    print("  ✓ save_data/save_models split passed")


def test_checkpoint_mode_with_training_key():
    """Test that checkpoint_mode works with 'training' config key."""
    print("Testing checkpoint_mode with training key...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)

        dh = DataHandler(input_dim=2)
        dh.add_data(np.random.rand(5, 2), np.random.rand(5, 1), loop=0)
        model = DummyModel(input_dim=2)

        # Use 'training' key (new preferred name)
        config = {'training': {'checkpoint_mode': 'final'}}

        manager.save_checkpoint(
            loop=0,
            data_handlers=[dh],
            models=[model],
            config=config,
            oracle_names=['test']
        )

        oracle_dir = Path(tmpdir) / "test"
        assert (oracle_dir / "model_0_final.pkl").exists()

    print("  ✓ checkpoint_mode with training key passed")


def run_all_tests():
    """Run all CheckpointManager tests."""
    print("\n" + "="*60)
    print("CheckpointManager Tests")
    print("="*60 + "\n")

    test_basic_save_load()
    test_multiple_loops()
    test_rollback()
    test_oracle_naming()
    test_load_latest()
    test_data_always_saved()
    test_checkpoint_mode_with_training_key()

    print("\n" + "="*60)
    print("All CheckpointManager tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
