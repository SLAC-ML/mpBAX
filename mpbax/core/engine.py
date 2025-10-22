"""Engine for orchestrating the optimization workflow."""

import numpy as np
import yaml
from pathlib import Path
from typing import List, Callable, Optional, Dict

from mpbax.core.evaluator import Evaluator
from mpbax.core.data_handler import DataHandler
from mpbax.core.model import BaseModel
from mpbax.core.algorithm import BaseAlgorithm
from mpbax.core.checkpoint import CheckpointManager


class Engine:
    """Main engine for mpBAX optimization framework.

    Orchestrates the workflow:
    1. Generate/load initial data
    2. Evaluate with oracle functions
    3. Train models
    4. Propose new candidates
    5. Checkpoint and repeat
    """

    def __init__(
        self,
        config_path: str,
        fn_oracles: List[Callable[[np.ndarray], np.ndarray]],
        model_class: type,
        algorithm: BaseAlgorithm,
        fn_generate: Optional[Callable[[int, int], np.ndarray]] = None
    ):
        """Initialize Engine.

        Args:
            config_path: Path to config.yaml file
            fn_oracles: List of oracle functions, one per objective
            model_class: Class for creating models (must be subclass of BaseModel)
            algorithm: Algorithm instance for proposing candidates
            fn_generate: Optional function to generate initial samples.
                        Signature: (n_samples, input_dim) -> X (n_samples, input_dim)
                        If None, uses uniform random in [0, 1]^d
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set random seed
        self.seed = self.config.get('seed', 42)
        np.random.seed(self.seed)

        # Store functions and classes
        self.fn_oracles = fn_oracles
        self.model_class = model_class
        self.algorithm = algorithm
        self.fn_generate = fn_generate or self._default_generate

        # Initialize components
        self.n_objectives = len(fn_oracles)
        self.obj_configs = self.config['objectives']
        assert len(self.obj_configs) == self.n_objectives, \
            f"Config has {len(self.obj_configs)} objectives, but {self.n_objectives} oracles provided"

        # Create evaluators
        self.evaluators = []
        for i, (fn_oracle, obj_config) in enumerate(zip(fn_oracles, self.obj_configs)):
            evaluator = Evaluator(
                fn_oracle=fn_oracle,
                input_dim=obj_config['input_dim'],
                name=obj_config['name']
            )
            self.evaluators.append(evaluator)

        # Initialize checkpoint manager
        checkpoint_dir = self.config['checkpoint']['dir']
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Run state
        self.current_loop = 0
        self.data_handlers = []
        self.models = []

    def run(self) -> None:
        """Run the optimization loop."""
        # Check if resuming from checkpoint
        resume_from = self.config['checkpoint'].get('resume_from')

        if resume_from is not None:
            self._resume_from_checkpoint(resume_from)
        else:
            self._initialize_fresh_run()

        # Main optimization loop
        max_loops = self.config['max_loops']
        checkpoint_freq = self.config['checkpoint']['freq']

        while self.current_loop < max_loops:
            print(f"\n=== Loop {self.current_loop} ===")

            # Step 1: Propose and evaluate new candidates for each objective
            if self.current_loop > 0:
                print("Proposing new candidates...")

                # Propose candidates independently for each objective
                # since they may have different input dimensions
                for i, (evaluator, obj_config) in enumerate(zip(self.evaluators, self.obj_configs)):
                    # Create algorithm instance for this objective's dimension
                    # (if objectives have different dimensions)
                    from mpbax.core.algorithm import RandomSampling, GreedySampling

                    if isinstance(self.algorithm, RandomSampling):
                        obj_algorithm = RandomSampling(
                            n_propose=self.algorithm.n_propose,
                            input_dim=obj_config['input_dim'],
                            seed=self.seed + self.current_loop + i  # Unique seed per objective
                        )
                    elif isinstance(self.algorithm, GreedySampling):
                        obj_algorithm = GreedySampling(
                            n_propose=self.algorithm.n_propose,
                            input_dim=obj_config['input_dim'],
                            seed=self.seed + self.current_loop + i,
                            n_candidates=self.algorithm.n_candidates
                        )
                    else:
                        # For custom algorithms, assume they can handle dimension properly
                        obj_algorithm = self.algorithm

                    # Propose candidates using only this objective's predict function
                    X_new = obj_algorithm.propose([self.models[i].predict])

                    # Evaluate new candidates
                    Y_new = evaluator.evaluate(X_new)

                    # Create data handler for this loop's data
                    dh_new = DataHandler(input_dim=obj_config['input_dim'])
                    dh_new.add_data(X_new, Y_new)

                    # Store for checkpointing
                    self.data_handlers[i] = dh_new

            # Step 3: Get accumulated data for training
            accumulated_data = self._get_accumulated_data()

            # Step 4: Train models
            print("Training models...")
            for i, obj_config in enumerate(self.obj_configs):
                X_train, Y_train = accumulated_data[i]

                # Create new model for this loop
                model = self.model_class(input_dim=obj_config['input_dim'])
                model.train(X_train, Y_train)
                self.models[i] = model

            # Step 5: Save checkpoint
            if self.current_loop % checkpoint_freq == 0:
                print(f"Saving checkpoint at loop {self.current_loop}...")
                obj_names = [obj['name'] for obj in self.obj_configs]
                self.checkpoint_manager.save_checkpoint(
                    loop=self.current_loop,
                    data_handlers=self.data_handlers,
                    models=self.models,
                    config=self.config,
                    obj_names=obj_names
                )

            # Print progress
            self._print_progress()

            # Move to next loop
            self.current_loop += 1

        print(f"\n=== Optimization completed after {max_loops} loops ===")
        self._print_final_summary()

    def _initialize_fresh_run(self) -> None:
        """Initialize a fresh optimization run with random data."""
        print("Initializing fresh run...")

        n_initial = self.config['n_initial']

        # Initialize data handlers and models
        self.data_handlers = []
        self.models = []

        # Generate and evaluate initial data for each objective
        for i, (evaluator, obj_config) in enumerate(zip(self.evaluators, self.obj_configs)):
            print(f"Generating initial data for {obj_config['name']}...")

            # Generate initial samples
            X0 = self.fn_generate(n_initial, obj_config['input_dim'])

            # Evaluate
            Y0 = evaluator.evaluate(X0)

            # Create data handler
            dh = DataHandler(input_dim=obj_config['input_dim'])
            dh.add_data(X0, Y0)
            self.data_handlers.append(dh)

            # Initialize empty model (will be trained in first loop)
            model = self.model_class(input_dim=obj_config['input_dim'])
            self.models.append(model)

        self.current_loop = 0

    def _resume_from_checkpoint(self, resume_from: str) -> None:
        """Resume from a checkpoint.

        Args:
            resume_from: "latest" or specific loop number
        """
        if resume_from == "latest":
            loop_to_resume = None
            print("Resuming from latest checkpoint...")
        else:
            loop_to_resume = int(resume_from)
            print(f"Resuming from checkpoint at loop {loop_to_resume}...")

        # Load checkpoint
        loop, data_handlers, models, _, _ = \
            self.checkpoint_manager.load_checkpoint(loop=loop_to_resume)

        # Restore state
        self.current_loop = loop + 1  # Continue from next loop
        self.data_handlers = data_handlers
        self.models = models

        print(f"Resumed from loop {loop}, continuing with loop {self.current_loop}")

    def _get_accumulated_data(self) -> List[tuple]:
        """Get accumulated data for all objectives from all loops.

        Returns:
            List of (X, Y) tuples, one per objective
        """
        accumulated_data = []

        for i in range(self.n_objectives):
            # Load all data from loop 0 to current_loop
            X_all = None
            Y_all = None

            for loop in range(self.current_loop + 1):
                # For current loop, use in-memory data handler
                if loop == self.current_loop:
                    X, Y = self.data_handlers[i].get_data()
                else:
                    # Load data for previous loops from checkpoint
                    obj_dir = Path(self.config['checkpoint']['dir']) / f"obj_{i}"
                    data_path = obj_dir / f"data_{loop}.pkl"

                    if data_path.exists():
                        dh = DataHandler.load(str(data_path))
                        X, Y = dh.get_data()
                    else:
                        X, Y = None, None

                if X is not None:
                    if X_all is None:
                        X_all = X
                        Y_all = Y
                    else:
                        X_all = np.vstack([X_all, X])
                        Y_all = np.vstack([Y_all, Y])

            accumulated_data.append((X_all, Y_all))

        return accumulated_data

    def _default_generate(self, n: int, d: int) -> np.ndarray:
        """Default generator: uniform random in [0, 1]^d.

        Args:
            n: Number of samples
            d: Dimensionality

        Returns:
            X with shape (n, d)
        """
        return np.random.rand(n, d)

    def _print_progress(self) -> None:
        """Print progress information."""
        print(f"\nLoop {self.current_loop} summary:")
        for i, evaluator in enumerate(self.evaluators):
            print(f"  {evaluator.name}: {evaluator.get_eval_count()} evaluations")

    def _print_final_summary(self) -> None:
        """Print final summary."""
        print("\nFinal summary:")
        for i, evaluator in enumerate(self.evaluators):
            print(f"  {evaluator.name}: {evaluator.get_eval_count()} total evaluations")
