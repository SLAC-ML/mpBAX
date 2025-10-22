"""Algorithm interface for proposing candidates."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable


class BaseAlgorithm(ABC):
    """Abstract base class for optimization algorithms.

    Algorithms propose next candidates X to evaluate based on predict functions.
    """

    def __init__(self, n_propose: int, input_dim: int, seed: int = 42):
        """Initialize algorithm.

        Args:
            n_propose: Number of candidates to propose per iteration
            input_dim: Input dimensionality d
            seed: Random seed for reproducibility
        """
        self.n_propose = n_propose
        self.input_dim = input_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def propose(self, fn_pred_list: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
        """Propose next candidates to evaluate.

        Args:
            fn_pred_list: List of predict functions, one per objective.
                          Each fn_pred has signature: X (n, d) -> Y (n, 1)

        Returns:
            X: Proposed candidates with shape (n_propose, d)
        """
        pass


class RandomSampling(BaseAlgorithm):
    """Simple random sampling algorithm.

    Proposes random samples uniformly in [0, 1]^d.
    Useful for testing and as a baseline.
    """

    def __init__(self, n_propose: int, input_dim: int, seed: int = 42):
        """Initialize RandomSampling.

        Args:
            n_propose: Number of candidates to propose per iteration
            input_dim: Input dimensionality d
            seed: Random seed for reproducibility
        """
        super().__init__(n_propose, input_dim, seed)

    def propose(self, fn_pred_list: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
        """Propose random candidates in [0, 1]^d.

        Args:
            fn_pred_list: List of predict functions (unused for random sampling)

        Returns:
            X: Random candidates with shape (n_propose, input_dim)
        """
        X = self.rng.rand(self.n_propose, self.input_dim)
        return X


class GreedySampling(BaseAlgorithm):
    """Greedy sampling algorithm for multi-objective optimization.

    For single objective: samples n_propose candidates and picks ones with best predicted values.
    For multi-objective: uses a simple weighted sum approach.

    Note: This is a simple baseline. More sophisticated algorithms can be implemented
    by subclassing BaseAlgorithm.
    """

    def __init__(self, n_propose: int, input_dim: int, seed: int = 42, n_candidates: int = 1000):
        """Initialize GreedySampling.

        Args:
            n_propose: Number of candidates to propose per iteration
            input_dim: Input dimensionality d
            seed: Random seed for reproducibility
            n_candidates: Number of random candidates to evaluate before selecting best
        """
        super().__init__(n_propose, input_dim, seed)
        self.n_candidates = n_candidates

    def propose(self, fn_pred_list: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
        """Propose candidates by greedy selection.

        Generates random candidates, evaluates them with predict functions,
        and selects the best ones.

        Args:
            fn_pred_list: List of predict functions, one per objective

        Returns:
            X: Best candidates with shape (n_propose, input_dim)
        """
        # Generate random candidates
        X_candidates = self.rng.rand(self.n_candidates, self.input_dim)

        # Evaluate with all predict functions
        scores = np.zeros((self.n_candidates, len(fn_pred_list)))
        for i, fn_pred in enumerate(fn_pred_list):
            Y_pred = fn_pred(X_candidates)  # (n_candidates, 1)
            scores[:, i] = Y_pred.flatten()

        # For multi-objective, use simple weighted sum (equal weights)
        # Use argsort to get indices of smallest scores (minimization)
        total_score = np.mean(scores, axis=1)

        # Select top n_propose candidates (lowest scores)
        top_indices = np.argsort(total_score)[:self.n_propose]
        X_proposed = X_candidates[top_indices]

        return X_proposed
