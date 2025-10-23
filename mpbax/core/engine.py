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
        algorithm: Optional[BaseAlgorithm] = None,
        fn_generate: Optional[Callable[[int, int], np.ndarray]] = None
    ):
        """Initialize Engine.

        Args:
            config_path: Path to config.yaml file
            fn_oracles: List of oracle functions, one per oracle
            model_class: Class for creating models (must be subclass of BaseModel)
            algorithm: Optional algorithm instance. If None, instantiates from config.
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
        self.fn_generate = fn_generate or self._default_generate

        # Initialize components
        n_oracles = len(fn_oracles)
        self.obj_configs = self.config['oracles']
        assert len(self.obj_configs) == n_oracles, \
            f"Config has {len(self.obj_configs)} oracles, but {n_oracles} fn_oracles provided"

        # Instantiate algorithm if not provided
        if algorithm is None:
            self.algorithm = self._instantiate_algorithm()
        else:
            self.algorithm = algorithm

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

            # Step 1: Propose and evaluate new candidates
            if self.current_loop > 0:
                print("Proposing new candidates...")

                # Collect all predict functions
                fn_preds = [model.predict for model in self.models]

                # Call propose ONCE with all predict functions
                X_list = self.algorithm.propose(fn_preds)

                # Validate: must return one X per oracle
                if len(X_list) != len(self.evaluators):
                    raise ValueError(
                        f"Algorithm returned {len(X_list)} X arrays, "
                        f"but expected {len(self.evaluators)} (one per oracle)"
                    )

                # Evaluate each X with corresponding oracle
                for i, (X_new, evaluator, obj_config) in enumerate(
                    zip(X_list, self.evaluators, self.obj_configs)
                ):
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
        """Get accumulated data for all oracles from all loops.

        Returns:
            List of (X, Y) tuples, one per oracle
        """
        accumulated_data = []

        n_oracles = len(self.evaluators)
        for i in range(n_oracles):
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

    def _instantiate_algorithm(self) -> BaseAlgorithm:
        """Instantiate algorithm from config.

        Returns:
            Instantiated algorithm

        Raises:
            ValueError: If algorithm config is invalid or class not found
        """
        if 'algorithm' not in self.config:
            raise ValueError("Config must contain 'algorithm' section when algorithm=None")

        algo_config = self.config['algorithm']
        algo_class_name = algo_config.get('class')
        algo_params = algo_config.get('params', {})

        if not algo_class_name:
            raise ValueError("Algorithm config must specify 'class'")

        # Handle built-in algorithms
        if algo_class_name in ['RandomSampling', 'GreedySampling']:
            from mpbax.core.algorithm import RandomSampling, GreedySampling

            if algo_class_name == 'RandomSampling':
                algo_class = RandomSampling
            else:
                algo_class = GreedySampling

        # Handle custom algorithms via import path
        else:
            # Import custom class from module path
            # e.g., "mymodule.MyAlgorithm"
            try:
                module_path, class_name = algo_class_name.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_path)
                algo_class = getattr(module, class_name)
            except (ValueError, ImportError, AttributeError) as e:
                raise ValueError(
                    f"Failed to import algorithm class '{algo_class_name}': {e}"
                )

        # Instantiate algorithm with params
        try:
            algorithm = algo_class(**algo_params)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate {algo_class_name} with params {algo_params}: {e}"
            )

        return algorithm

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
