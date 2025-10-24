"""Tests for Evaluator."""

import numpy as np

from mpbax.core.evaluator import Evaluator


def test_basic_evaluation():
    """Test basic oracle evaluation."""
    print("Testing basic evaluation...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    evaluator = Evaluator(oracle, input_dim=3, name="test_oracle")

    # Test evaluation
    X = np.random.rand(5, 3)
    Y = evaluator.evaluate(X)

    assert Y.shape == (5, 1)
    assert evaluator.get_eval_count() == 5

    # Evaluate more data
    X2 = np.random.rand(3, 3)
    Y2 = evaluator.evaluate(X2)

    assert Y2.shape == (3, 1)
    assert evaluator.get_eval_count() == 8

    print("  ✓ basic evaluation passed")


def test_shape_validation():
    """Test input shape validation."""
    print("Testing shape validation...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    evaluator = Evaluator(oracle, input_dim=3, name="test_oracle")

    # Wrong dimension
    try:
        X_wrong = np.random.rand(5, 2)  # Should be (n, 3)
        evaluator.evaluate(X_wrong)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dimension mismatch" in str(e)

    print("  ✓ shape validation passed")


def test_multi_output_oracle():
    """Test oracle returning multiple outputs."""
    print("Testing multi-output oracle...")

    def oracle_multi(X):
        # Return 3 outputs per input
        sum_sq = np.sum(X**2, axis=1, keepdims=True)
        sum_val = np.sum(X, axis=1, keepdims=True)
        max_val = np.max(X, axis=1, keepdims=True)
        return np.hstack([sum_sq, sum_val, max_val])

    evaluator = Evaluator(oracle_multi, input_dim=2, name="multi_oracle")

    X = np.random.rand(5, 2)
    Y = evaluator.evaluate(X)

    assert Y.shape == (5, 3)
    assert evaluator.get_eval_count() == 5

    print("  ✓ multi-output oracle passed")


def test_eval_count():
    """Test evaluation counter."""
    print("Testing evaluation counter...")

    def oracle(X):
        return np.sum(X**2, axis=1, keepdims=True)

    evaluator = Evaluator(oracle, input_dim=2, name="test")

    assert evaluator.get_eval_count() == 0

    evaluator.evaluate(np.random.rand(10, 2))
    assert evaluator.get_eval_count() == 10

    evaluator.evaluate(np.random.rand(5, 2))
    assert evaluator.get_eval_count() == 15

    print("  ✓ evaluation counter passed")


def run_all_tests():
    """Run all Evaluator tests."""
    print("\n" + "="*60)
    print("Evaluator Tests")
    print("="*60 + "\n")

    test_basic_evaluation()
    test_shape_validation()
    test_multi_output_oracle()
    test_eval_count()

    print("\n" + "="*60)
    print("All Evaluator tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
