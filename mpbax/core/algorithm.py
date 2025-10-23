"""Algorithm interface for proposing candidates."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable


class BaseAlgorithm(ABC):
    """Abstract base class for optimization algorithms.

    Algorithms propose next candidates to evaluate based on predict functions.

    Note: Subclasses define their own __init__ with algorithm-specific parameters.
    The framework does not enforce any specific constructor signature.
    """

    @abstractmethod
    def propose(self, fn_pred_list: List[Callable[[np.ndarray], np.ndarray]]) -> List[np.ndarray]:
        """Propose next candidates to evaluate.

        Args:
            fn_pred_list: List of predict functions, one per oracle.
                          Each fn_pred has signature: X (n, d) -> Y (n, k)

        Returns:
            List of X arrays, one per oracle. Must have len(X_list) == len(fn_pred_list).
            Each X has shape (n_propose_i, d_i) where dimensions can vary per oracle.
        """
        pass


class RandomSampling(BaseAlgorithm):
    """Simple random sampling algorithm.

    Proposes random samples uniformly in [0, 1]^d for each oracle independently.
    Useful for testing and as a baseline.
    """

    def __init__(self, input_dims: List[int], n_propose: int, seed: int = 42):
        """Initialize RandomSampling.

        Args:
            input_dims: List of input dimensionalities, one per oracle
            n_propose: Number of candidates to propose per oracle
            seed: Random seed for reproducibility
        """
        self.input_dims = input_dims
        self.n_propose = n_propose
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def propose(self, fn_pred_list: List[Callable[[np.ndarray], np.ndarray]]) -> List[np.ndarray]:
        """Propose random candidates in [0, 1]^d for each oracle.

        Args:
            fn_pred_list: List of predict functions (unused for random sampling)

        Returns:
            List of X arrays, one per oracle. Each X has shape (n_propose, input_dim_i).
        """
        if len(fn_pred_list) != len(self.input_dims):
            raise ValueError(
                f"Number of predict functions ({len(fn_pred_list)}) must match "
                f"number of input dimensions ({len(self.input_dims)})"
            )

        X_list = []
        for input_dim in self.input_dims:
            X = self.rng.rand(self.n_propose, input_dim)
            X_list.append(X)

        return X_list


class GreedySampling(BaseAlgorithm):
    """Greedy sampling algorithm for multi-objective optimization.

    For each oracle independently:
    - Generates random candidates
    - Evaluates with that oracle's predict function
    - Selects best candidates

    Note: This simple version optimizes each oracle independently.
    More sophisticated algorithms can implement true multi-objective optimization.
    """

    def __init__(self, input_dims: List[int], n_propose: int, n_candidates: int = 1000, seed: int = 42):
        """Initialize GreedySampling.

        Args:
            input_dims: List of input dimensionalities, one per oracle
            n_propose: Number of candidates to propose per oracle
            n_candidates: Number of random candidates to evaluate before selecting best
            seed: Random seed for reproducibility
        """
        self.input_dims = input_dims
        self.n_propose = n_propose
        self.n_candidates = n_candidates
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def propose(self, fn_pred_list: List[Callable[[np.ndarray], np.ndarray]]) -> List[np.ndarray]:
        """Propose candidates by greedy selection for each oracle.

        Generates random candidates for each oracle, evaluates with its predict function,
        and selects the best ones.

        Args:
            fn_pred_list: List of predict functions, one per oracle

        Returns:
            List of X arrays, one per oracle. Each X has shape (n_propose, input_dim_i).
        """
        if len(fn_pred_list) != len(self.input_dims):
            raise ValueError(
                f"Number of predict functions ({len(fn_pred_list)}) must match "
                f"number of input dimensions ({len(self.input_dims)})"
            )

        X_list = []
        for input_dim, fn_pred in zip(self.input_dims, fn_pred_list):
            # Generate random candidates for this oracle's input space
            X_candidates = self.rng.rand(self.n_candidates, input_dim)

            # Evaluate with this oracle's predict function
            Y_pred = fn_pred(X_candidates)  # (n_candidates, k)

            # For multi-output, average across output dimensions for scoring
            if Y_pred.ndim == 2 and Y_pred.shape[1] > 1:
                scores = np.mean(Y_pred, axis=1)  # (n_candidates,)
            else:
                scores = Y_pred.flatten()  # (n_candidates,)

            # Select top n_propose candidates (lowest scores for minimization)
            top_indices = np.argsort(scores)[:self.n_propose]
            X_proposed = X_candidates[top_indices]

            X_list.append(X_proposed)

        return X_list
