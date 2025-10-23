# mpBAX - Multipoint Bayesian Algorithm Execution

A lightweight framework for model-based multi-objective optimization with built-in checkpointing and reproducibility.

## Features

- **Simple & Robust**: Minimal dependencies, straightforward architecture
- **Multi-Oracle Support**: Optimize multiple independent oracles with different input dimensions
- **Multi-Output Support**: Oracles can return multiple outputs (Y with shape (n, k) where k ≥ 1)
- **Checkpointing**: Automatic checkpointing with resume and rollback capabilities
- **Reproducible**: Built-in seed management for reproducible results
- **Modular Design**: Easy to extend with custom models and algorithms

## Installation

No installation required - just add the `mpbax` directory to your PYTHONPATH:

```bash
export PYTHONPATH=/path/to/mpBAX:$PYTHONPATH
```

## Quick Start

### Single-Objective Optimization

```python
import numpy as np
import yaml
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling

# Define your oracle function
def my_oracle(X: np.ndarray) -> np.ndarray:
    """X has shape (n, d), returns Y with shape (n, k) where k >= 1"""
    return np.sum(X**2, axis=1, keepdims=True)

# Create config
config = {
    'seed': 42,
    'max_loops': 10,
    'n_initial': 20,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1, 'resume_from': None},
    'oracles': [{'name': 'my_obj', 'input_dim': 2}],
    'algorithm': {
        'class': 'GreedySampling',
        'params': {
            'input_dims': [2],
            'n_propose': 10,
            'n_candidates': 1000
        }
    }
}

# Run optimization - pass config dict directly (no need to save to file)
engine = Engine(
    config=config,  # Can be dict or path to YAML file
    fn_oracles=[my_oracle],
    model_class=DummyModel,
    algorithm=None  # Auto-instantiate from config
)
engine.run()
```

### Multi-Oracle Optimization

```python
# Define multiple oracle functions
def oracle_1(X: np.ndarray) -> np.ndarray:  # 2D input
    return np.sum(X**2, axis=1, keepdims=True)

def oracle_2(X: np.ndarray) -> np.ndarray:  # 3D input
    return np.sum(X, axis=1, keepdims=True)

# Update config for multiple oracles
config['oracles'] = [
    {'name': 'obj1', 'input_dim': 2},
    {'name': 'obj2', 'input_dim': 3}
]
config['algorithm'] = {
    'class': 'GreedySampling',
    'params': {
        'input_dims': [2, 3],  # Must match oracle dimensions
        'n_propose': 8,
        'n_candidates': 500
    }
}

# Run with multiple oracles
engine = Engine(
    config=config,
    fn_oracles=[oracle_1, oracle_2],
    model_class=DummyModel,
    algorithm=None  # Auto-instantiate from config
)
engine.run()
```

### Multi-Output Oracle

```python
# Oracle that returns multiple outputs per evaluation
def multi_output_oracle(X: np.ndarray) -> np.ndarray:
    """Returns Y with shape (n, 3) - three outputs per input"""
    sum_squares = np.sum(X**2, axis=1, keepdims=True)
    sum_vals = np.sum(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)
    return np.hstack([sum_squares, sum_vals, max_vals])

# Config for single oracle with multiple outputs
config = {
    'seed': 42,
    'max_loops': 5,
    'n_initial': 10,
    'checkpoint': {'dir': 'checkpoints_multi_output', 'freq': 1, 'resume_from': None},
    'oracles': [{'name': 'multi_output_oracle', 'input_dim': 2}],
    'algorithm': {
        'class': 'GreedySampling',
        'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 500}
    }
}

# Run optimization
engine = Engine(
    config=config,
    fn_oracles=[multi_output_oracle],
    model_class=DummyModel,
    algorithm=None
)
engine.run()
```

## Checkpointing Features

### Resume from Latest Checkpoint

```python
config['checkpoint']['resume_from'] = 'latest'
```

### Resume from Specific Loop

```python
config['checkpoint']['resume_from'] = 5  # Resume from loop 5
```

### Rollback to Previous Loop

```python
from mpbax.core.checkpoint import CheckpointManager

manager = CheckpointManager('checkpoints')
manager.delete_checkpoints_after(loop=3)  # Keep only loops 0-3
```

## Directory Structure

```
mpBAX/
├── CLAUDE.md           # Development guidelines
├── README.md           # This file
├── config.yaml         # Configuration template
├── mpbax/
│   ├── core/           # Core framework modules
│   │   ├── algorithm.py      # Algorithm interface
│   │   ├── checkpoint.py     # Checkpoint manager
│   │   ├── data_handler.py   # Data management
│   │   ├── engine.py         # Main orchestrator
│   │   ├── evaluator.py      # Oracle wrapper
│   │   └── model.py          # Model interface
│   ├── examples/       # Example scripts
│   │   ├── example_single_objective.py
│   │   ├── example_multi_objective.py
│   │   ├── example_multi_output.py
│   │   └── example_checkpoint_workflow.py
│   ├── tests/          # Unit tests
│   │   ├── test_data_handler.py
│   │   ├── test_interfaces.py
│   │   ├── test_checkpoint.py
│   │   └── test_engine.py
│   └── utils/          # Utility functions
```

## Running Examples

```bash
# Single-objective example
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/examples/example_single_objective.py

# Multi-oracle example
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/examples/example_multi_objective.py

# Multi-output oracle example
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/examples/example_multi_output.py

# Checkpoint workflow demo
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/examples/example_checkpoint_workflow.py
```

## Running Tests

```bash
# Test data handler
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/tests/test_data_handler.py

# Test interfaces
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/tests/test_interfaces.py

# Test checkpoint manager
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/tests/test_checkpoint.py

# Test engine
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/tests/test_engine.py
```

## Key Concepts

### Data Shape Conventions

- **Input X**: Shape `(n, d)` where `n` = number of samples, `d` = input dimensionality
- **Output Y**: Shape `(n, k)` where `n` = number of samples, `k` = number of outputs (k ≥ 1)

All components enforce these shapes for consistency. Multi-output oracles (k > 1) are fully supported.

### Workflow

1. **Initialize**: Generate initial data X0 or load from checkpoint
2. **Evaluate**: X → Oracle → Y (expensive simulation)
3. **Train**: (X, Y) → Model → predict()
4. **Propose**: predict() → Algorithm → X_next (new candidates)
5. **Checkpoint**: Save data and model
6. **Repeat** until termination condition

### Multi-Oracle Design

Each oracle operates independently with:
- Its own oracle function (fn_oracle)
- Its own input space (can have different dimensions)
- Its own data handler (stores X, Y pairs)
- Its own model (trains on oracle-specific data)
- Its own predict function (fn_pred)

The algorithm's `propose()` method receives all predict functions and returns a list of X arrays (one per oracle) to enable model-based multi-oracle optimization.

## Configuration Options

```yaml
seed: 42                    # Random seed for reproducibility
max_loops: 10               # Maximum number of optimization loops
n_initial: 20               # Number of initial samples

checkpoint:
  dir: "checkpoints"        # Checkpoint directory
  freq: 1                   # Checkpoint every N loops
  resume_from: null         # null, "latest", or loop number

oracles:
  - name: "oracle_1"        # Oracle name
    input_dim: 2            # Input dimensionality
  # Add more oracles for multi-oracle optimization

algorithm:
  class: "GreedySampling"   # Algorithm class name (built-in or custom)
  params:                   # Algorithm-specific parameters
    input_dims: [2]         # List of input dimensions (one per oracle)
    n_propose: 10           # Number of candidates to propose per loop
    n_candidates: 1000      # Number of candidate points to evaluate
```

## Extending the Framework

### Custom Model

```python
from mpbax.core.model import BaseModel

class MyModel(BaseModel):
    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Your training logic
        # X shape: (n, d), Y shape: (n, k) where k >= 1
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Your prediction logic
        # Input: X shape (n, d)
        # Returns: Y shape (n, k) matching oracle output dimensions
        pass
```

### Custom Algorithm

```python
from mpbax.core.algorithm import BaseAlgorithm
from typing import List, Callable
import numpy as np

class MyAlgorithm(BaseAlgorithm):
    def __init__(self, input_dims: List[int], n_propose: int, **kwargs):
        """Define your own constructor with needed parameters.

        Args:
            input_dims: List of input dimensions (one per oracle)
            n_propose: Number of candidates to propose per loop
            **kwargs: Additional algorithm-specific parameters
        """
        self.input_dims = input_dims
        self.n_propose = n_propose
        # Your initialization logic

    def propose(self, fn_pred_list: List[Callable]) -> List[np.ndarray]:
        """Propose new candidates for each oracle.

        Args:
            fn_pred_list: List of predict functions (one per oracle)

        Returns:
            List of X arrays, one per oracle
            X_i has shape (n_propose, input_dims[i])
        """
        # Your candidate proposal logic
        # Must return len(X_list) == len(fn_pred_list)
        X_list = []
        for i, fn_pred in enumerate(fn_pred_list):
            # Generate candidates for oracle i
            X_i = ...  # Shape (n_propose, input_dims[i])
            X_list.append(X_i)
        return X_list
```

## Design Principles

- **Simplicity First**: No databases, no async, no over-engineering
- **Minimal Dependencies**: numpy, pyyaml, pickle
- **Testable Components**: Each module independently tested
- **Clear Interfaces**: Consistent function signatures
- **Reproducible**: Seed management throughout

## Dependencies

- Python 3.7+
- NumPy
- PyYAML

## License

See project license file.
