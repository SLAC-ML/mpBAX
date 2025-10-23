"""Integration tests for core interfaces: Evaluator, Model, and Algorithm."""

import numpy as np
import tempfile
import os

from mpbax.core.evaluator import Evaluator
from mpbax.core.model import DummyModel, BaseModel
from mpbax.core.algorithm import RandomSampling, GreedySampling


# Define a simple oracle function for testing
def simple_oracle(X: np.ndarray) -> np.ndarray:
    """Simple test function: sum of squares.

    Args:
        X: Input with shape (n, d)

    Returns:
        Y: Output with shape (n, 1)
    """
    Y = np.sum(X**2, axis=1, keepdims=True)
    return Y


def multi_output_oracle(X: np.ndarray) -> np.ndarray:
    """Oracle with multiple outputs.

    Args:
        X: Input with shape (n, d)

    Returns:
        Y: Output with shape (n, 3)
    """
    Y = np.hstack([
        np.sum(X**2, axis=1, keepdims=True),   # sum of squares
        np.sum(X, axis=1, keepdims=True),      # sum
        np.ones((X.shape[0], 1))               # constant
    ])
    return Y


def test_evaluator():
    """Test Evaluator wrapper."""
    print("Testing Evaluator...")

    evaluator = Evaluator(fn_oracle=simple_oracle, input_dim=2, name="test_obj")

    # Test valid evaluation
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    Y = evaluator.evaluate(X)

    assert Y.shape == (2, 1)
    expected_Y = np.array([[5.0], [25.0]])  # 1^2+2^2=5, 3^2+4^2=25
    np.testing.assert_array_almost_equal(Y, expected_Y)

    # Check evaluation count
    assert evaluator.get_eval_count() == 2

    # Test another evaluation
    X2 = np.array([[0.0, 0.0]])
    Y2 = evaluator.evaluate(X2)
    assert evaluator.get_eval_count() == 3

    # Test invalid input
    try:
        X_invalid = np.array([[1.0, 2.0, 3.0]])  # Wrong dimension
        evaluator.evaluate(X_invalid)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dimension mismatch" in str(e)
        print(f"  Correctly caught error: {e}")

    print("  Evaluator (single-output) tests passed!")


def test_evaluator_multi_output():
    """Test Evaluator with multi-output oracle."""
    print("  Testing Evaluator with multi-output...")

    evaluator = Evaluator(fn_oracle=multi_output_oracle, input_dim=2, name="multi_out")

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Y = evaluator.evaluate(X)

    # Should accept multi-output Y
    assert Y.shape == (2, 3)
    assert evaluator.get_eval_count() == 2

    print("  Evaluator (multi-output) tests passed!\n")


def test_model():
    """Test Model interface with DummyModel."""
    print("Testing Model...")

    model = DummyModel(input_dim=2)

    # Prepare training data
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y_train = np.array([[10.0], [20.0], [30.0]])

    # Train model
    model.train(X_train, Y_train)
    assert model.is_trained

    # Test prediction
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])
    Y_pred = model.predict(X_test)

    assert Y_pred.shape == (2, 1)
    expected_mean = 20.0  # Mean of [10, 20, 30]
    np.testing.assert_array_almost_equal(Y_pred, [[expected_mean], [expected_mean]])

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.pkl")
        model.save(filepath)

        loaded_model = BaseModel.load(filepath)
        Y_pred_loaded = loaded_model.predict(X_test)
        np.testing.assert_array_equal(Y_pred, Y_pred_loaded)

    print("  Model (single-output) tests passed!")


def test_model_multi_output():
    """Test Model with multi-output data."""
    print("  Testing Model with multi-output...")

    model = DummyModel(input_dim=2)

    # Multi-output training data
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Y_train = np.array([[10.0, 1.0, 0.5], [20.0, 2.0, 0.6], [30.0, 3.0, 0.7]])

    # Train model
    model.train(X_train, Y_train)
    assert model.is_trained

    # Test prediction
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])
    Y_pred = model.predict(X_test)

    assert Y_pred.shape == (2, 3)
    # Mean per column: [20.0, 2.0, 0.6]
    expected_mean = np.array([20.0, 2.0, 0.6])
    np.testing.assert_array_almost_equal(Y_pred[0], expected_mean)
    np.testing.assert_array_almost_equal(Y_pred[1], expected_mean)

    print("  Model (multi-output) tests passed!\n")


def test_algorithm():
    """Test Algorithm interface with RandomSampling and GreedySampling."""
    print("Testing Algorithm...")

    # Test RandomSampling with single oracle
    print("  Testing RandomSampling (single oracle)...")
    random_algo = RandomSampling(input_dims=[3], n_propose=5, seed=42)

    # Create dummy predict function
    def dummy_pred(X):
        return np.sum(X, axis=1, keepdims=True)

    X_list = random_algo.propose([dummy_pred])
    assert len(X_list) == 1
    assert X_list[0].shape == (5, 3)
    assert np.all(X_list[0] >= 0) and np.all(X_list[0] <= 1)

    # Test reproducibility
    random_algo2 = RandomSampling(input_dims=[3], n_propose=5, seed=42)
    X_list2 = random_algo2.propose([dummy_pred])
    np.testing.assert_array_equal(X_list[0], X_list2[0])

    # Test RandomSampling with multiple oracles (different dimensions)
    print("  Testing RandomSampling (multi-oracle)...")
    random_algo_multi = RandomSampling(input_dims=[2, 3, 4], n_propose=5, seed=42)

    def pred1(X): return np.sum(X, axis=1, keepdims=True)
    def pred2(X): return np.sum(X**2, axis=1, keepdims=True)
    def pred3(X): return np.mean(X, axis=1, keepdims=True)

    X_list_multi = random_algo_multi.propose([pred1, pred2, pred3])
    assert len(X_list_multi) == 3
    assert X_list_multi[0].shape == (5, 2)
    assert X_list_multi[1].shape == (5, 3)
    assert X_list_multi[2].shape == (5, 4)

    # Test GreedySampling
    print("  Testing GreedySampling...")
    greedy_algo = GreedySampling(input_dims=[2], n_propose=3, seed=42, n_candidates=100)

    # Predict function that prefers low values
    def pred_minimize(X):
        return np.sum(X**2, axis=1, keepdims=True)

    X_list_greedy = greedy_algo.propose([pred_minimize])
    assert len(X_list_greedy) == 1
    assert X_list_greedy[0].shape == (3, 2)

    # Verify greedy selection works by comparing to average random samples
    random_samples = np.random.rand(100, 2)
    random_vals = pred_minimize(random_samples)
    greedy_vals = pred_minimize(X_list_greedy[0])

    assert np.mean(greedy_vals) < np.mean(random_vals), \
        "Greedy should select points with better (lower) predicted values than random"

    print("Algorithm tests passed!\n")


def test_integration():
    """Test all components together in a mini workflow."""
    print("Testing integration of all components...")

    # Setup
    evaluator = Evaluator(fn_oracle=simple_oracle, input_dim=2, name="test")
    model = DummyModel(input_dim=2)
    algorithm = RandomSampling(input_dims=[2], n_propose=3, seed=42)

    # Step 1: Generate initial data
    X0 = np.array([[0.5, 0.5], [1.0, 1.0]])

    # Step 2: Evaluate
    Y0 = evaluator.evaluate(X0)
    assert Y0.shape == (2, 1)

    # Step 3: Train model
    model.train(X0, Y0)

    # Step 4: Propose new candidates using model's predict function
    X_list = algorithm.propose([model.predict])
    assert len(X_list) == 1
    X_next = X_list[0]
    assert X_next.shape == (3, 2)

    # Step 5: Evaluate new candidates
    Y_next = evaluator.evaluate(X_next)
    assert Y_next.shape == (3, 1)

    # Verify evaluation count
    assert evaluator.get_eval_count() == 5  # 2 initial + 3 new

    print("Integration test passed!\n")


if __name__ == "__main__":
    test_evaluator()
    test_evaluator_multi_output()
    test_model()
    test_model_multi_output()
    test_algorithm()
    test_integration()
    print("All interface tests passed!")
