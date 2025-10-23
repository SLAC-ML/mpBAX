"""Engine for orchestrating the optimization workflow."""

import numpy as np
import yaml
from pathlib import Path
from typing import List, Callable, Optional, Dict, Union

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

    def __init__(self, config: Union[str, Dict]):
        """Initialize Engine from config only.

        Args:
            config: Either a dict containing config or a str path to config.yaml file
                   Config must specify all components: oracles (with functions and models),
                   algorithm, and optionally generators.
        """
        # Load config
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"config must be str or dict, got {type(config)}")

        # Set random seed
        self.seed = self.config.get('seed', 42)
        np.random.seed(self.seed)

        # Get oracle configs
        self.oracle_configs = self.config['oracles']
        n_oracles = len(self.oracle_configs)

        # Validate oracle configs
        for i, oracle_config in enumerate(self.oracle_configs):
            required_fields = ['name', 'input_dim', 'n_initial', 'function', 'model']
            for field in required_fields:
                if field not in oracle_config:
                    raise ValueError(f"Oracle {i} missing required field: '{field}'")

            # Validate function config
            if 'class' not in oracle_config['function']:
                raise ValueError(f"Oracle {i} function config missing 'class' field")

            # Validate model config
            if 'class' not in oracle_config['model']:
                raise ValueError(f"Oracle {i} model config missing 'class' field")

        # Validate oracle names are unique
        oracle_names = [cfg['name'] for cfg in self.oracle_configs]
        if len(oracle_names) != len(set(oracle_names)):
            raise ValueError(f"Oracle names must be unique, got: {oracle_names}")

        # Instantiate components from config
        self.fn_oracles = self._instantiate_oracle_functions()
        self.fn_generate_list = self._instantiate_generators()
        self.algorithm = self._instantiate_algorithm()

        # Create evaluators
        self.evaluators = []
        for fn_oracle, oracle_config in zip(self.fn_oracles, self.oracle_configs):
            evaluator = Evaluator(
                fn_oracle=fn_oracle,
                input_dim=oracle_config['input_dim'],
                name=oracle_config['name']
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
                for i, (X_new, evaluator, oracle_config) in enumerate(
                    zip(X_list, self.evaluators, self.oracle_configs)
                ):
                    # Evaluate new candidates
                    Y_new = evaluator.evaluate(X_new)

                    # Create data handler for this loop's data
                    dh_new = DataHandler(input_dim=oracle_config['input_dim'])
                    dh_new.add_data(X_new, Y_new, loop=self.current_loop)

                    # Store for checkpointing
                    self.data_handlers[i] = dh_new

            # Step 3: Get accumulated data for training
            accumulated_data = self._get_accumulated_data()

            # Step 4: Train models
            print("Training models...")
            model_mode = self.config.get('model', {}).get('mode', 'retrain')

            for i, oracle_config in enumerate(self.oracle_configs):
                X_train, Y_train, metadata = accumulated_data[i]

                # Check training mode
                if model_mode == 'finetune' and self.current_loop > 0:
                    # Finetuning: continue from previous model
                    model = self.models[i]  # Reuse existing model instance
                    model.train(X_train, Y_train, metadata=metadata)
                else:
                    # Retraining: create new model from scratch (default)
                    model = self.model_class(input_dim=oracle_config['input_dim'])
                    model.train(X_train, Y_train, metadata=metadata)
                    self.models[i] = model

            # Step 5: Save checkpoint
            if self.current_loop % checkpoint_freq == 0:
                print(f"Saving checkpoint at loop {self.current_loop}...")
                oracle_names = [obj['name'] for obj in self.oracle_configs]
                self.checkpoint_manager.save_checkpoint(
                    loop=self.current_loop,
                    data_handlers=self.data_handlers,
                    models=self.models,
                    config=self.config,
                    oracle_names=oracle_names
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

        # Initialize data handlers and models
        self.data_handlers = []

        # Generate and evaluate initial data for each oracle
        for i, (evaluator, oracle_config) in enumerate(zip(self.evaluators, self.oracle_configs)):
            print(f"Generating initial data for {oracle_config['name']}...")

            # Get n_initial for this oracle
            n_initial = oracle_config['n_initial']

            # Generate initial samples using per-oracle generator
            X0 = self.fn_generate_list[i](n_initial, oracle_config['input_dim'])

            # Evaluate
            Y0 = evaluator.evaluate(X0)

            # Create data handler
            dh = DataHandler(input_dim=oracle_config['input_dim'])
            dh.add_data(X0, Y0, loop=0)  # Initial data is from loop 0
            self.data_handlers.append(dh)

        # Instantiate models (will be trained in first loop)
        self.models = self._instantiate_models()

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
            List of (X, Y, metadata) tuples, one per oracle
        """
        accumulated_data = []

        n_oracles = len(self.evaluators)
        for i in range(n_oracles):
            # Load all data from loop 0 to current_loop
            X_all = None
            Y_all = None
            loop_indices_all = []

            for loop in range(self.current_loop + 1):
                # For current loop, use in-memory data handler
                if loop == self.current_loop:
                    X, Y, meta = self.data_handlers[i].get_data_with_metadata()
                else:
                    # Load data for previous loops from checkpoint
                    oracle_name = self.oracle_configs[i]['name']
                    from mpbax.core.checkpoint import _sanitize_oracle_name
                    oracle_name_sanitized = _sanitize_oracle_name(oracle_name)
                    oracle_dir = Path(self.config['checkpoint']['dir']) / oracle_name_sanitized

                    # Backward compatibility: try old naming if new doesn't exist
                    if not oracle_dir.exists():
                        oracle_dir = Path(self.config['checkpoint']['dir']) / f"oracle_{i}"

                    data_path = oracle_dir / f"data_{loop}.pkl"

                    if data_path.exists():
                        dh = DataHandler.load(str(data_path))
                        X, Y, meta = dh.get_data_with_metadata()
                    else:
                        X, Y, meta = None, None, {}

                if X is not None:
                    if X_all is None:
                        X_all = X
                        Y_all = Y
                    else:
                        X_all = np.vstack([X_all, X])
                        Y_all = np.vstack([Y_all, Y])

                    # Accumulate loop indices if available
                    if 'loop_indices' in meta:
                        loop_indices_all.append(meta['loop_indices'])

            # Build metadata
            metadata = {}
            if loop_indices_all:
                metadata['loop_indices'] = np.concatenate(loop_indices_all)

            accumulated_data.append((X_all, Y_all, metadata))

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

    def _instantiate_oracle_functions(self) -> List[Callable]:
        """Instantiate oracle functions from config.

        For each oracle, reads function.class (import path) and function.params.
        Imports the function/factory and instantiates with params.

        Returns:
            List of oracle functions with signature fn(X) -> Y

        Raises:
            ValueError: If function config is invalid or import fails
        """
        fn_oracles = []

        for i, oracle_config in enumerate(self.oracle_configs):
            fn_config = oracle_config['function']
            fn_class_name = fn_config['class']
            fn_params = fn_config.get('params', {})

            try:
                # Import function/factory
                module_path, obj_name = fn_class_name.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_path)
                fn_or_factory = getattr(module, obj_name)

                # If params provided, call as factory; otherwise use directly
                if fn_params:
                    fn_oracle = fn_or_factory(**fn_params)
                else:
                    fn_oracle = fn_or_factory

                fn_oracles.append(fn_oracle)

            except (ValueError, ImportError, AttributeError, TypeError) as e:
                raise ValueError(
                    f"Failed to instantiate oracle function '{fn_class_name}' "
                    f"for oracle {i} ('{oracle_config['name']}'): {e}"
                )

        return fn_oracles

    def _instantiate_generators(self) -> List[Callable]:
        """Instantiate generator functions from oracle configs.

        For each oracle, reads optional generate config.
        Returns list of generator functions.

        Returns:
            List of generator functions with signature fn(n_samples, input_dim) -> X

        Raises:
            ValueError: If generator config is invalid or import fails
        """
        fn_generate_list = []

        for i, oracle_config in enumerate(self.oracle_configs):
            gen_config = oracle_config.get('generate')

            if gen_config is None:
                # Use default generator
                fn_generate_list.append(self._default_generate)
            else:
                # Import custom generator
                gen_class_name = gen_config['class']
                gen_params = gen_config.get('params', {})

                try:
                    # Import generator/factory
                    module_path, obj_name = gen_class_name.rsplit('.', 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    gen_or_factory = getattr(module, obj_name)

                    # If params provided, call as factory; otherwise use directly
                    if gen_params:
                        fn_generate = gen_or_factory(**gen_params)
                    else:
                        fn_generate = gen_or_factory

                    fn_generate_list.append(fn_generate)

                except (ValueError, ImportError, AttributeError, TypeError) as e:
                    raise ValueError(
                        f"Failed to instantiate generator '{gen_class_name}' "
                        f"for oracle {i} ('{oracle_config['name']}'): {e}"
                    )

        return fn_generate_list

    def _instantiate_models(self) -> List[BaseModel]:
        """Instantiate models from oracle configs.

        For each oracle, reads model.class and model.params.
        Handles built-in models and custom models via import path.

        Returns:
            List of model instances

        Raises:
            ValueError: If model config is invalid or import fails
        """
        models = []

        for i, oracle_config in enumerate(self.oracle_configs):
            model_config = oracle_config['model']
            model_class_name = model_config['class']
            model_params = model_config.get('params', {})

            # Handle built-in models
            if model_class_name in ['DummyModel']:
                from mpbax.core.model import DummyModel
                model_class = DummyModel
            elif model_class_name in ['DANetModel']:
                from mpbax.plugins.models.da_net_model import DANetModel
                model_class = DANetModel
            else:
                # Import custom model via full path
                try:
                    module_path, class_name = model_class_name.rsplit('.', 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    model_class = getattr(module, class_name)
                except (ValueError, ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Failed to import model class '{model_class_name}' "
                        f"for oracle {i} ('{oracle_config['name']}'): {e}"
                    )

            # Instantiate model with input_dim and params
            try:
                model = model_class(input_dim=oracle_config['input_dim'], **model_params)
                models.append(model)
            except TypeError as e:
                raise ValueError(
                    f"Failed to instantiate {model_class_name} with params {model_params} "
                    f"for oracle {i} ('{oracle_config['name']}'): {e}"
                )

        return models

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
