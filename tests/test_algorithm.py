"""Tests for Algorithm interface and implementations."""

import numpy as np

from mpbax.core.algorithm import RandomSampling, GreedySampling


def test_random_sampling_single_oracle():
    """Test RandomSampling with single oracle."""
    print("Testing RandomSampling (single oracle)...")

    def mock_predict(X):
        return np.sum(X**2, axis=1, keepdims=True)

    algo = RandomSampling(input_dims=[2], n_propose=10, seed=42)
    X_list = algo.propose([mock_predict])

    assert len(X_list) == 1
    assert X_list[0].shape == (10, 2)
    assert np.all(X_list[0] >= 0) and np.all(X_list[0] <= 1)

    print("  ✓ RandomSampling (single oracle) passed")


def test_random_sampling_multi_oracle():
    """Test RandomSampling with multiple oracles."""
    print("Testing RandomSampling (multi-oracle)...")

    def mock_predict(X):
        return np.sum(X**2, axis=1, keepdims=True)

    algo = RandomSampling(input_dims=[2, 3, 4], n_propose=10, seed=42)
    X_list = algo.propose([mock_predict, mock_predict, mock_predict])

    assert len(X_list) == 3
    assert X_list[0].shape == (10, 2)
    assert X_list[1].shape == (10, 3)
    assert X_list[2].shape == (10, 4)

    print("  ✓ RandomSampling (multi-oracle) passed")


def test_random_sampling_reproducibility():
    """Test RandomSampling seed reproducibility."""
    print("Testing RandomSampling reproducibility...")

    def mock_predict(X):
        return np.sum(X**2, axis=1, keepdims=True)

    algo1 = RandomSampling(input_dims=[2], n_propose=10, seed=42)
    X_list1 = algo1.propose([mock_predict])

    algo2 = RandomSampling(input_dims=[2], n_propose=10, seed=42)
    X_list2 = algo2.propose([mock_predict])

    assert np.allclose(X_list1[0], X_list2[0])

    print("  ✓ RandomSampling reproducibility passed")


def test_greedy_sampling_single_oracle():
    """Test GreedySampling with single oracle."""
    print("Testing GreedySampling (single oracle)...")

    def mock_predict(X):
        return np.sum(X**2, axis=1, keepdims=True)

    algo = GreedySampling(input_dims=[2], n_propose=5, n_candidates=50, seed=42)
    X_list = algo.propose([mock_predict])

    assert len(X_list) == 1
    assert X_list[0].shape == (5, 2)

    print("  ✓ GreedySampling (single oracle) passed")


def test_greedy_sampling_selects_best():
    """Test that GreedySampling selects better candidates than random."""
    print("Testing GreedySampling selection quality...")

    def mock_predict(X):
        return np.sum(X**2, axis=1, keepdims=True)

    # GreedySampling should select points with lower predicted values
    algo_greedy = GreedySampling(input_dims=[2], n_propose=10, n_candidates=1000, seed=42)
    X_greedy_list = algo_greedy.propose([mock_predict])
    X_greedy = X_greedy_list[0]

    # Compare to random sampling
    algo_random = RandomSampling(input_dims=[2], n_propose=10, seed=42)
    X_random_list = algo_random.propose([mock_predict])
    X_random = X_random_list[0]

    # Greedy should have lower average predicted value
    Y_greedy = mock_predict(X_greedy)
    Y_random = mock_predict(X_random)

    assert Y_greedy.mean() < Y_random.mean()

    print("  ✓ GreedySampling selection quality passed")


def test_greedy_sampling_multi_oracle():
    """Test GreedySampling with multiple oracles."""
    print("Testing GreedySampling (multi-oracle)...")

    def mock_predict1(X):
        return np.sum(X**2, axis=1, keepdims=True)

    def mock_predict2(X):
        return np.sum(X, axis=1, keepdims=True)

    algo = GreedySampling(input_dims=[2, 3], n_propose=5, n_candidates=50, seed=42)
    X_list = algo.propose([mock_predict1, mock_predict2])

    assert len(X_list) == 2
    assert X_list[0].shape == (5, 2)
    assert X_list[1].shape == (5, 3)

    print("  ✓ GreedySampling (multi-oracle) passed")


def run_all_tests():
    """Run all Algorithm tests."""
    print("\n" + "="*60)
    print("Algorithm Tests")
    print("="*60 + "\n")

    test_random_sampling_single_oracle()
    test_random_sampling_multi_oracle()
    test_random_sampling_reproducibility()
    test_greedy_sampling_single_oracle()
    test_greedy_sampling_selects_best()
    test_greedy_sampling_multi_oracle()

    print("\n" + "="*60)
    print("All Algorithm tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
