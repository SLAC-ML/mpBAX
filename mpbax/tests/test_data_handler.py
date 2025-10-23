"""Tests for Data Handler."""

import numpy as np
import tempfile
import os
from pathlib import Path

from mpbax.core.data_handler import DataHandler


def test_add_data():
    """Test adding data with correct shapes."""
    handler = DataHandler(input_dim=2)

    # Add first batch
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    Y1 = np.array([[5.0], [6.0]])  # (2, 1)
    handler.add_data(X1, Y1)

    assert handler.get_size() == 2

    # Add second batch
    X2 = np.array([[7.0, 8.0]])  # (1, 2)
    Y2 = np.array([[9.0]])  # (1, 1)
    handler.add_data(X2, Y2)

    assert handler.get_size() == 3

    # Check combined data
    X, Y = handler.get_data()
    expected_X = np.array([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
    expected_Y = np.array([[5.0], [6.0], [9.0]])

    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(Y, expected_Y)

    print("test_add_data passed")


def test_invalid_shapes():
    """Test shape validation."""
    handler = DataHandler(input_dim=2)

    # Wrong X dimension
    try:
        X = np.array([[1.0, 2.0, 3.0]])  # (1, 3) - should be (1, 2)
        Y = np.array([[5.0]])
        handler.add_data(X, Y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dimension mismatch" in str(e)
        print(f"Correctly caught dimension mismatch: {e}")

    # Wrong Y shape
    try:
        X = np.array([[1.0, 2.0]])  # (1, 2)
        Y = np.array([5.0])  # (1,) - should be (1, 1)
        handler.add_data(X, Y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "2D array" in str(e)
        print(f"Correctly caught Y shape error: {e}")

    # Mismatched sample sizes
    try:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        Y = np.array([[5.0]])  # (1, 1) - should be (2, 1)
        handler.add_data(X, Y)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "same number of samples" in str(e)
        print(f"Correctly caught sample size mismatch: {e}")

    print("test_invalid_shapes passed")


def test_save_load():
    """Test saving and loading data."""
    handler = DataHandler(input_dim=3)

    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    Y = np.array([[7.0], [8.0]])
    handler.add_data(X, Y)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_data.pkl")
        handler.save(filepath)

        # Load and verify
        loaded_handler = DataHandler.load(filepath)

        assert loaded_handler.input_dim == 3
        assert loaded_handler.get_size() == 2

        X_loaded, Y_loaded = loaded_handler.get_data()
        np.testing.assert_array_equal(X_loaded, X)
        np.testing.assert_array_equal(Y_loaded, Y)

    print("test_save_load passed")


def test_empty_handler():
    """Test handler with no data."""
    handler = DataHandler(input_dim=5)

    assert handler.get_size() == 0
    X, Y = handler.get_data()
    assert X is None
    assert Y is None

    print("test_empty_handler passed")


def test_multi_output():
    """Test handler with multi-output Y."""
    handler = DataHandler(input_dim=2)

    # Add data with Y shape (n, 3)
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y1 = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    handler.add_data(X1, Y1)

    assert handler.get_size() == 2

    # Add more multi-output data
    X2 = np.array([[5.0, 6.0]])
    Y2 = np.array([[11.0, 12.0, 13.0]])
    handler.add_data(X2, Y2)

    assert handler.get_size() == 3

    # Check combined data
    X, Y = handler.get_data()
    expected_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected_Y = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])

    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(Y, expected_Y)

    print("test_multi_output passed")


if __name__ == "__main__":
    test_add_data()
    test_invalid_shapes()
    test_save_load()
    test_empty_handler()
    test_multi_output()
    print("\nAll Data Handler tests passed!")
