"""Tests for DataHandler."""

import numpy as np
import tempfile
from pathlib import Path

from mpbax.core.data_handler import DataHandler


def test_add_and_get_data():
    """Test adding and retrieving data."""
    print("Testing add_data and get_data...")

    dh = DataHandler(input_dim=2)

    # Test add_data
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 1)
    dh.add_data(X, Y, loop=0)

    # Test get_data
    X_ret, Y_ret = dh.get_data()
    assert X_ret.shape == (10, 2)
    assert Y_ret.shape == (10, 1)
    assert np.allclose(X, X_ret)

    print("  ✓ add_data and get_data passed")


def test_loop_tracking():
    """Test loop index tracking in metadata."""
    print("Testing loop tracking...")

    dh = DataHandler(input_dim=2)

    # Add data from loop 0
    X1 = np.random.rand(10, 2)
    Y1 = np.random.rand(10, 1)
    dh.add_data(X1, Y1, loop=0)

    # Add data from loop 1
    X2 = np.random.rand(5, 2)
    Y2 = np.random.rand(5, 1)
    dh.add_data(X2, Y2, loop=1)

    # Get data with metadata
    X_all, Y_all, meta = dh.get_data_with_metadata()
    assert X_all.shape == (15, 2)
    assert 'loop_indices' in meta
    assert meta['loop_indices'].shape == (15,)

    # Check loop indices are correct
    assert np.all(meta['loop_indices'][:10] == 0)
    assert np.all(meta['loop_indices'][10:] == 1)

    print("  ✓ loop tracking passed")


def test_save_and_load():
    """Test saving and loading DataHandler."""
    print("Testing save and load...")

    dh = DataHandler(input_dim=2)

    X = np.random.rand(15, 2)
    Y = np.random.rand(15, 1)
    dh.add_data(X, Y, loop=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_data.pkl"
        dh.save(str(path))

        dh_loaded = DataHandler.load(str(path))
        X_loaded, Y_loaded = dh_loaded.get_data()

        assert np.allclose(X, X_loaded)
        assert np.allclose(Y, Y_loaded)

    print("  ✓ save and load passed")


def test_multi_output():
    """Test handling multi-output data (Y with k > 1)."""
    print("Testing multi-output data...")

    dh = DataHandler(input_dim=2)

    # Add multi-output data: Y has shape (n, 3)
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 3)
    dh.add_data(X, Y, loop=0)

    X_ret, Y_ret = dh.get_data()
    assert X_ret.shape == (10, 2)
    assert Y_ret.shape == (10, 3)

    print("  ✓ multi-output data passed")


def test_shape_validation():
    """Test shape validation on add_data."""
    print("Testing shape validation...")

    dh = DataHandler(input_dim=2)

    # Wrong input dimension
    try:
        X = np.random.rand(10, 3)  # Should be (n, 2)
        Y = np.random.rand(10, 1)
        dh.add_data(X, Y, loop=0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # Mismatched sample counts
    try:
        X = np.random.rand(10, 2)
        Y = np.random.rand(5, 1)  # Should be (10, 1)
        dh.add_data(X, Y, loop=0)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print("  ✓ shape validation passed")


def run_all_tests():
    """Run all DataHandler tests."""
    print("\n" + "="*60)
    print("DataHandler Tests")
    print("="*60 + "\n")

    test_add_and_get_data()
    test_loop_tracking()
    test_save_and_load()
    test_multi_output()
    test_shape_validation()

    print("\n" + "="*60)
    print("All DataHandler tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
