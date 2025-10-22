# mpBAX - Multipoint Bayesian Algorithm Execution

A lightweight framework for model-based multi-objective optimization with built-in checkpointing and reproducibility.

## Features

- **Simple & Robust**: Minimal dependencies, straightforward architecture
- **Multi-Objective Support**: Optimize multiple independent objectives with different input dimensions
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
    """X has shape (n, d), returns Y with shape (n, 1)"""
    return np.sum(X**2, axis=1, keepdims=True)

# Create config
config = {
    'seed': 42,
    'max_loops': 10,
    'n_initial': 20,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1, 'resume_from': None},
    'n_propose': 10,
    'objectives': [{'name': 'my_obj', 'input_dim': 2}]
}

# Save config
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)

# Create algorithm
algorithm = GreedySampling(n_propose=10, input_dim=2, seed=42)

# Run optimization
engine = Engine(
    config_path='config.yaml',
    fn_oracles=[my_oracle],
    model_class=DummyModel,
    algorithm=algorithm
)
engine.run()
```

### Multi-Objective Optimization

```python
# Define multiple oracle functions
def oracle_1(X: np.ndarray) -> np.ndarray:  # 2D input
    return np.sum(X**2, axis=1, keepdims=True)

def oracle_2(X: np.ndarray) -> np.ndarray:  # 3D input
    return np.sum(X, axis=1, keepdims=True)

# Update config for multiple objectives
config['objectives'] = [
    {'name': 'obj1', 'input_dim': 2},
    {'name': 'obj2', 'input_dim': 3}
]

# Run with multiple oracles
engine = Engine(
    config_path='config.yaml',
    fn_oracles=[oracle_1, oracle_2],
    model_class=DummyModel,
    algorithm=algorithm
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

# Multi-objective example
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/examples/example_multi_objective.py

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

- **Input X**: Shape `(n, d)` where `n` = number of samples, `d` = dimensionality
- **Output Y**: Shape `(n, 1)` where `n` = number of samples

All components enforce these shapes for consistency.

### Workflow

1. **Initialize**: Generate initial data X0 or load from checkpoint
2. **Evaluate**: X → Oracle → Y (expensive simulation)
3. **Train**: (X, Y) → Model → predict()
4. **Propose**: predict() → Algorithm → X_next (new candidates)
5. **Checkpoint**: Save data and model
6. **Repeat** until termination condition

### Multi-Objective Design

Each objective operates independently with:
- Its own oracle function
- Its own input space (can have different dimensions)
- Its own data handler
- Its own model

The algorithm receives all predict functions to enable model-based multi-objective optimization.

## Configuration Options

```yaml
seed: 42                    # Random seed for reproducibility
max_loops: 10               # Maximum number of optimization loops
n_initial: 20               # Number of initial samples
n_propose: 10               # Number of candidates to propose per loop

checkpoint:
  dir: "checkpoints"        # Checkpoint directory
  freq: 1                   # Checkpoint every N loops
  resume_from: null         # null, "latest", or loop number

objectives:
  - name: "obj_1"           # Objective name
    input_dim: 2            # Input dimensionality
  # Add more objectives for multi-objective optimization
```

## Extending the Framework

### Custom Model

```python
from mpbax.core.model import BaseModel

class MyModel(BaseModel):
    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        # Your training logic
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Your prediction logic (must return shape (n, 1))
        pass
```

### Custom Algorithm

```python
from mpbax.core.algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def propose(self, fn_pred_list):
        # Your candidate proposal logic
        # Returns X with shape (n_propose, input_dim)
        pass
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
