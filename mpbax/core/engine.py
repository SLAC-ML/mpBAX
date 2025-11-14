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
            oracle_name = oracle_config.get('name', f'oracle_{i}')

            # Basic required fields (input_dim and n_initial now optional at oracle level)
            required_fields = ['name', 'function', 'model']
            for field in required_fields:
                if field not in oracle_config:
                    raise ValueError(f"Oracle {i} missing required field: '{field}'")

            # Validate function config
            if 'class' not in oracle_config['function']:
                raise ValueError(f"Oracle {i} ({oracle_name}) function config missing 'class' field")

            # Validate model config
            if 'class' not in oracle_config['model']:
                raise ValueError(f"Oracle {i} ({oracle_name}) model config missing 'class' field")

            # Validate input_dim is specified somewhere (oracle level OR model.params OR generate.params)
            input_dim_oracle = oracle_config.get('input_dim')
            input_dim_model = oracle_config.get('model', {}).get('params', {}).get('input_dim')
            input_dim_generate = oracle_config.get('generate', {}).get('params', {}).get('d')

            if input_dim_oracle is None and input_dim_model is None and input_dim_generate is None:
                raise ValueError(
                    f"Oracle {i} ({oracle_name}): input_dim must be specified in one of: "
                    "oracle config, model.params.input_dim, or generate.params.d"
                )

            # Validate n_initial is specified somewhere (oracle level OR generate.params)
            n_initial_oracle = oracle_config.get('n_initial')
            n_initial_generate = oracle_config.get('generate', {}).get('params', {}).get('n')

            if n_initial_oracle is None and n_initial_generate is None:
                raise ValueError(
                    f"Oracle {i} ({oracle_name}): n_initial must be specified in either "
                    "oracle config or generate.params.n"
                )

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
            # Get input_dim from oracle config, model.params, or generate.params
            input_dim = oracle_config.get('input_dim')
            if input_dim is None:
                model_params = oracle_config.get('model', {}).get('params', {})
                input_dim = model_params.get('input_dim')
            if input_dim is None:
                gen_params = oracle_config.get('generate', {}).get('params', {})
                input_dim = gen_params.get('d')

            evaluator = Evaluator(
                fn_oracle=fn_oracle,
                input_dim=input_dim,
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
                    # Get input_dim from X_new if not in config
                    input_dim = oracle_config.get('input_dim')
                    if input_dim is None:
                        input_dim = X_new.shape[1]
                    dh_new = DataHandler(input_dim=input_dim)
                    dh_new.add_data(X_new, Y_new, loop=self.current_loop)

                    # Store for checkpointing
                    self.data_handlers[i] = dh_new

            # Step 3: Get accumulated data for training
            accumulated_data = self._get_accumulated_data()

            # Step 4: Train models
            print("Training models...")
            # Support both 'training' (new) and 'model' (deprecated) for backward compatibility
            training_config = self.config.get('training')
            if training_config is None:
                training_config = self.config.get('model', {})
                if training_config:
                    import warnings
                    warnings.warn(
                        "Using 'model' for top-level training config is deprecated. "
                        "Please rename to 'training' in your config.",
                        DeprecationWarning,
                        stacklevel=2
                    )
            model_mode = training_config.get('mode', 'retrain')

            if model_mode == 'finetune' and self.current_loop > 0:
                # Finetuning: continue from previous model instances
                for i, oracle_config in enumerate(self.oracle_configs):
                    X_train, Y_train, metadata = accumulated_data[i]
                    model = self.models[i]  # Reuse existing model instance
                    model.train(X_train, Y_train, metadata=metadata)
            else:
                # Retraining: create new models from scratch (default)
                # Or first loop in finetune mode
                self.models = self._instantiate_models()
                for i, oracle_config in enumerate(self.oracle_configs):
                    X_train, Y_train, metadata = accumulated_data[i]
                    model = self.models[i]
                    model.train(X_train, Y_train, metadata=metadata)

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

            # Handle flexible generator calling
            gen_config = oracle_config.get('generate')

            # Check if user provided a custom generator (has 'class' field)
            has_custom_generator = gen_config is not None and 'class' in gen_config

            if has_custom_generator and 'params' in gen_config:
                # Custom generator with params - call with those params
                gen_params = gen_config['params']
                X0 = self.fn_generate_list[i](**gen_params)
            else:
                # Default generator or generator without params
                # Get n_initial (from oracle or generate.params.n)
                n_initial = oracle_config.get('n_initial')
                if n_initial is None and gen_config and 'params' in gen_config:
                    n_initial = gen_config['params'].get('n')
                if n_initial is None:
                    raise ValueError(
                        f"Oracle {i} ({oracle_config['name']}): n_initial not found"
                    )

                # Get input_dim (from oracle, model.params, or generate.params.d)
                input_dim = oracle_config.get('input_dim')
                if input_dim is None:
                    model_params = oracle_config.get('model', {}).get('params', {})
                    input_dim = model_params.get('input_dim')
                if input_dim is None and gen_config and 'params' in gen_config:
                    input_dim = gen_config['params'].get('d')

                if input_dim is not None:
                    # Call with (n, d) - standard signature
                    X0 = self.fn_generate_list[i](n_initial, input_dim)
                else:
                    # Call with just n - problem-specific generator
                    X0 = self.fn_generate_list[i](n_initial)

            # Evaluate
            Y0 = evaluator.evaluate(X0)

            # Create data handler - get input_dim from X0 if not in config
            input_dim_for_dh = oracle_config.get('input_dim')
            if input_dim_for_dh is None:
                input_dim_for_dh = X0.shape[1]
            dh = DataHandler(input_dim=input_dim_for_dh)
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

        For each oracle, reads function.class (import path string OR function instance)
        and function.params. Supports both:
        - String: imports function/factory from module path
        - Instance: uses the function/factory directly

        Returns:
            List of oracle functions with signature fn(X) -> Y

        Raises:
            ValueError: If function config is invalid or import/instantiation fails
        """
        fn_oracles = []

        for i, oracle_config in enumerate(self.oracle_configs):
            fn_config = oracle_config['function']
            fn_class_or_instance = fn_config['class']

            try:
                # Support both string (import path) and direct instance
                if isinstance(fn_class_or_instance, str):
                    # String: import function/factory from module path
                    module_path, obj_name = fn_class_or_instance.rsplit('.', 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    fn_or_factory = getattr(module, obj_name)
                else:
                    # Instance: use the function/factory directly
                    fn_or_factory = fn_class_or_instance

                # If params key exists in config, call as factory; otherwise use directly
                # Check for key existence to distinguish:
                #   - No 'params' key: use function directly
                #   - 'params': {} : call factory with no args
                #   - 'params': {x: 1}: call factory with args
                if 'params' in fn_config:
                    fn_params = fn_config['params']
                    fn_oracle = fn_or_factory(**fn_params)
                else:
                    fn_oracle = fn_or_factory

                fn_oracles.append(fn_oracle)

            except (ValueError, ImportError, AttributeError, TypeError) as e:
                # Generate appropriate error message
                if isinstance(fn_class_or_instance, str):
                    desc = f"string '{fn_class_or_instance}'"
                else:
                    desc = f"instance {fn_class_or_instance}"

                raise ValueError(
                    f"Failed to instantiate oracle function from {desc} "
                    f"for oracle {i} ('{oracle_config['name']}'): {e}"
                )

        return fn_oracles

    def _instantiate_generators(self) -> List[Callable]:
        """Instantiate generator functions from oracle configs.

        For each oracle, reads optional generate config.
        Supports both:
        - String: imports generator/factory from module path
        - Instance: uses the generator/factory directly
        - None: uses default uniform random generator

        Returns:
            List of generator functions with signature fn(n_samples, input_dim) -> X

        Raises:
            ValueError: If generator config is invalid or import/instantiation fails
        """
        fn_generate_list = []

        for i, oracle_config in enumerate(self.oracle_configs):
            gen_config = oracle_config.get('generate')

            if gen_config is None or 'class' not in gen_config:
                # Use default generator (when generate is None or just has params for default)
                fn_generate_list.append(self._default_generate)
            else:
                # Import custom generator or use instance
                gen_class_or_instance = gen_config['class']

                try:
                    # Support both string (import path) and direct instance
                    if isinstance(gen_class_or_instance, str):
                        # String: import generator/factory from module path
                        module_path, obj_name = gen_class_or_instance.rsplit('.', 1)
                        import importlib
                        module = importlib.import_module(module_path)
                        gen_or_factory = getattr(module, obj_name)
                    else:
                        # Instance: use the generator directly
                        gen_or_factory = gen_class_or_instance

                    # For generators, never call as factory during instantiation
                    # Store the generator function itself - params will be passed during call
                    fn_generate = gen_or_factory

                    fn_generate_list.append(fn_generate)

                except (ValueError, ImportError, AttributeError, TypeError) as e:
                    # Generate appropriate error message
                    if isinstance(gen_class_or_instance, str):
                        desc = f"string '{gen_class_or_instance}'"
                    else:
                        desc = f"instance {gen_class_or_instance}"

                    raise ValueError(
                        f"Failed to instantiate generator from {desc} "
                        f"for oracle {i} ('{oracle_config['name']}'): {e}"
                    )

        return fn_generate_list

    def _instantiate_models(self) -> List[BaseModel]:
        """Instantiate models from oracle configs.

        For each oracle, reads model.class and model.params.
        Supports both:
        - String: handles built-in models or imports from module path
        - Class: uses the model class directly

        Returns:
            List of model instances

        Raises:
            ValueError: If model config is invalid or import/instantiation fails
        """
        models = []

        for i, oracle_config in enumerate(self.oracle_configs):
            model_config = oracle_config['model']
            model_class_or_name = model_config['class']
            model_params = model_config.get('params', {})

            # Support both string (import path) and direct class
            if isinstance(model_class_or_name, str):
                # String: handle built-in models or import custom
                if model_class_or_name in ['DummyModel']:
                    from mpbax.core.model import DummyModel
                    model_class = DummyModel
                elif model_class_or_name in ['DANetModel']:
                    from mpbax.plugins.models.da_net_model import DANetModel
                    model_class = DANetModel
                else:
                    # Import custom model via full path
                    try:
                        module_path, class_name = model_class_or_name.rsplit('.', 1)
                        import importlib
                        module = importlib.import_module(module_path)
                        model_class = getattr(module, class_name)
                    except (ValueError, ImportError, AttributeError) as e:
                        raise ValueError(
                            f"Failed to import model class '{model_class_or_name}' "
                            f"for oracle {i} ('{oracle_config['name']}'): {e}"
                        )
            else:
                # Class instance: use directly
                model_class = model_class_or_name

            # Instantiate model with input_dim and params
            # Handle flexible input_dim placement: model.params.input_dim OR oracle.input_dim
            try:
                if 'input_dim' in model_params:
                    # input_dim already in model params - use as is
                    model = model_class(**model_params)
                else:
                    # Fall back to oracle-level input_dim (backward compatibility)
                    input_dim = oracle_config.get('input_dim')
                    if input_dim is None:
                        # Try to get from generate.params.d as last resort
                        input_dim = oracle_config.get('generate', {}).get('params', {}).get('d')
                    if input_dim is None:
                        raise ValueError(
                            f"input_dim not found for model in oracle {i} ({oracle_config['name']})"
                        )
                    model = model_class(input_dim=input_dim, **model_params)
                models.append(model)
            except TypeError as e:
                # Generate appropriate error message
                if isinstance(model_class_or_name, str):
                    desc = f"'{model_class_or_name}'"
                else:
                    desc = f"{model_class_or_name}"

                raise ValueError(
                    f"Failed to instantiate model {desc} with params {model_params} "
                    f"for oracle {i} ('{oracle_config['name']}'): {e}"
                )

        return models

    def _instantiate_algorithm(self) -> BaseAlgorithm:
        """Instantiate algorithm from config.

        Supports both:
        - String: handles built-in algorithms or imports from module path
        - Class: uses the algorithm class directly

        Returns:
            Instantiated algorithm

        Raises:
            ValueError: If algorithm config is invalid or class not found/instantiation fails
        """
        if 'algorithm' not in self.config:
            raise ValueError("Config must contain 'algorithm' section")

        algo_config = self.config['algorithm']
        algo_class_or_name = algo_config.get('class')
        algo_params = algo_config.get('params', {})

        if not algo_class_or_name:
            raise ValueError("Algorithm config must specify 'class'")

        # Support both string (import path) and direct class
        if isinstance(algo_class_or_name, str):
            # String: handle built-in algorithms or import custom
            if algo_class_or_name in ['RandomSampling', 'GreedySampling']:
                from mpbax.core.algorithm import RandomSampling, GreedySampling

                if algo_class_or_name == 'RandomSampling':
                    algo_class = RandomSampling
                else:
                    algo_class = GreedySampling

            # Handle custom algorithms via import path
            else:
                # Import custom class from module path
                # e.g., "mymodule.MyAlgorithm"
                try:
                    module_path, class_name = algo_class_or_name.rsplit('.', 1)
                    import importlib
                    module = importlib.import_module(module_path)
                    algo_class = getattr(module, class_name)
                except (ValueError, ImportError, AttributeError) as e:
                    raise ValueError(
                        f"Failed to import algorithm class '{algo_class_or_name}': {e}"
                    )
        else:
            # Class instance: use directly
            algo_class = algo_class_or_name

        # Instantiate algorithm with params
        try:
            algorithm = algo_class(**algo_params)
        except TypeError as e:
            # Generate appropriate error message
            if isinstance(algo_class_or_name, str):
                desc = f"'{algo_class_or_name}'"
            else:
                desc = f"{algo_class_or_name}"

            raise ValueError(
                f"Failed to instantiate algorithm {desc} with params {algo_params}: {e}"
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
