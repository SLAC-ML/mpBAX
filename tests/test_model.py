"""Tests for Model interface and implementations."""

import numpy as np
import tempfile
from pathlib import Path

from mpbax.core.model import BaseModel, DummyModel


def test_dummy_model():
    """Test DummyModel basic functionality."""
    print("Testing DummyModel...")

    model = DummyModel(input_dim=2)

    # Train model
    X_train = np.random.rand(20, 2)
    Y_train = np.random.rand(20, 1)
    model.train(X_train, Y_train)

    # Test prediction
    X_test = np.random.rand(5, 2)
    Y_pred = model.predict(X_test)

    assert Y_pred.shape == (5, 1)
    # DummyModel returns mean of training data
    assert np.allclose(Y_pred, Y_train.mean())

    print("  ✓ DummyModel passed")


def test_custom_model():
    """Test custom model implementation."""
    print("Testing custom model...")

    class CustomModel(BaseModel):
        def train(self, X, Y, metadata=None):
            self.mean = Y.mean(axis=0)
            self.is_trained = True

        def predict(self, X):
            if not self.is_trained:
                raise RuntimeError("Model not trained")
            return np.tile(self.mean, (X.shape[0], 1))

    model = CustomModel(input_dim=2)

    X_train = np.random.rand(20, 2)
    Y_train = np.random.rand(20, 1)
    model.train(X_train, Y_train)

    X_test = np.random.rand(5, 2)
    Y_pred = model.predict(X_test)

    assert Y_pred.shape == (5, 1)
    assert np.allclose(Y_pred, Y_train.mean())

    print("  ✓ custom model passed")


def test_multi_output_model():
    """Test model with multi-output data."""
    print("Testing multi-output model...")

    model = DummyModel(input_dim=2)

    # Train with multi-output Y
    X_train = np.random.rand(20, 2)
    Y_train = np.random.rand(20, 3)  # 3 outputs
    model.train(X_train, Y_train)

    # Predict
    X_test = np.random.rand(5, 2)
    Y_pred = model.predict(X_test)

    assert Y_pred.shape == (5, 3)
    # DummyModel returns mean per output dimension
    assert np.allclose(Y_pred, Y_train.mean(axis=0))

    print("  ✓ multi-output model passed")


def test_model_save_load():
    """Test model save/load functionality."""
    print("Testing model save/load...")

    model = DummyModel(input_dim=2)

    X_train = np.random.rand(20, 2)
    Y_train = np.random.rand(20, 1)
    model.train(X_train, Y_train)

    X_test = np.random.rand(5, 2)
    Y_pred = model.predict(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pkl"
        model.save(str(path))

        model_loaded = BaseModel.load(str(path))
        Y_pred_loaded = model_loaded.predict(X_test)

        assert np.allclose(Y_pred, Y_pred_loaded)

    print("  ✓ model save/load passed")


def test_model_with_metadata():
    """Test model training with metadata (loop tracking)."""
    print("Testing model with metadata...")

    class MetadataAwareModel(BaseModel):
        def train(self, X, Y, metadata=None):
            self.mean = Y.mean()
            self.is_trained = True

            # Check metadata
            if metadata and 'loop_indices' in metadata:
                loop_indices = metadata['loop_indices']
                assert loop_indices.shape[0] == X.shape[0]
                self.has_metadata = True
            else:
                self.has_metadata = False

        def predict(self, X):
            return np.full((X.shape[0], 1), self.mean)

    model = MetadataAwareModel(input_dim=2)

    X = np.random.rand(20, 2)
    Y = np.random.rand(20, 1)
    loop_indices = np.array([0]*10 + [1]*10)

    model.train(X, Y, metadata={'loop_indices': loop_indices})

    assert model.has_metadata

    print("  ✓ model with metadata passed")


def run_all_tests():
    """Run all Model tests."""
    print("\n" + "="*60)
    print("Model Tests")
    print("="*60 + "\n")

    test_dummy_model()
    test_custom_model()
    test_multi_output_model()
    test_model_save_load()
    test_model_with_metadata()

    print("\n" + "="*60)
    print("All Model tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
