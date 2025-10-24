# mpBAX - Multipoint Bayesian Algorithm Execution

A lightweight framework for model-based multi-objective optimization with built-in checkpointing and reproducibility.

> **⚠️ v2.0 Release**: This version introduces a **config-first architecture** with breaking API changes. Everything (oracle functions, models, generators) is now specified in config. See [Migration Guide](#migrating-from-v1x) below.

## Features

- **Config-First Design** (NEW in v2.0): Entire experiment specified in config file
- **Per-Oracle Flexibility** (NEW): Each oracle can have different n_initial, models, generators

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

### Option 1: Single-File Example (Python API)

For simple examples, define everything in one file using direct instances:

```python
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling

# Define oracle function directly in this file
def my_oracle(X: np.ndarray) -> np.ndarray:
    """X has shape (n, d), returns Y with shape (n, k) where k >= 1"""
    return np.sum(X**2, axis=1, keepdims=True)

# Pass function and class instances directly in config
config = {
    'seed': 42,
    'max_loops': 10,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1},
    'oracles': [{
        'name': 'my_objective',
        'input_dim': 2,
        'n_initial': 20,
        'function': {
            'class': my_oracle,  # Pass function directly!
        },
        'model': {
            'class': DummyModel,  # Pass class directly!
        }
    }],
    'algorithm': {
        'class': GreedySampling,  # Pass class directly!
        'params': {'input_dims': [2], 'n_propose': 10, 'n_candidates': 1000}
    }
}

engine = Engine(config)
engine.run()
```

### Option 2: Modular Approach (YAML + Python)

For larger projects, define oracle functions in separate modules and reference via import paths:

```python
# myproject/oracles.py
import numpy as np

def my_oracle(X: np.ndarray) -> np.ndarray:
    return np.sum(X**2, axis=1, keepdims=True)
```

```yaml
# myproject/config.yaml
seed: 42
max_loops: 10
checkpoint:
  dir: 'checkpoints'
  freq: 1

oracles:
  - name: 'my_objective'
    input_dim: 2
    n_initial: 20
    function:
      class: 'myproject.oracles.my_oracle'  # Import path string
    model:
      class: 'DummyModel'  # Built-in model name

algorithm:
  class: 'GreedySampling'
  params:
    input_dims: [2]
    n_propose: 10
    n_candidates: 1000
```

```python
# myproject/run.py
from mpbax.core.engine import Engine

engine = Engine('config.yaml')  # Load from YAML file
engine.run()
```

**Key Differences:**
- **Option 1 (Instances)**: Best for simple examples, rapid prototyping, single-file scripts
- **Option 2 (Strings)**: Best for large projects, YAML configs, reproducibility from config files

### Multi-Oracle Optimization

```python
# Define multiple oracle functions in your module
# myproject/oracles.py
def oracle_1(X: np.ndarray) -> np.ndarray:  # 2D input
    return np.sum(X**2, axis=1, keepdims=True)

def oracle_2(X: np.ndarray) -> np.ndarray:  # 3D input
    return np.sum(X, axis=1, keepdims=True)

# Config with multiple oracles
config = {
    'seed': 42,
    'max_loops': 10,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1},
    'oracles': [
        {
            'name': 'obj1',
            'input_dim': 2,
            'n_initial': 20,
            'function': {'class': 'myproject.oracles.oracle_1'},
            'model': {'class': 'DummyModel'}
        },
        {
            'name': 'obj2',
            'input_dim': 3,
            'n_initial': 30,  # Different n_initial per oracle!
            'function': {'class': 'myproject.oracles.oracle_2'},
            'model': {'class': 'DummyModel'}
        }
    ],
    'algorithm': {
        'class': 'GreedySampling',
        'params': {
            'input_dims': [2, 3],
            'n_propose': 8,
            'n_candidates': 500
        }
    }
}

# Pure config-first API
engine = Engine(config)
engine.run()
```

### Multi-Output Oracle

```python
# myproject/oracles.py
import numpy as np

def multi_output_oracle(X: np.ndarray) -> np.ndarray:
    """Returns Y with shape (n, 3) - three outputs per input"""
    sum_squares = np.sum(X**2, axis=1, keepdims=True)
    sum_vals = np.sum(X, axis=1, keepdims=True)
    max_vals = np.max(X, axis=1, keepdims=True)
    return np.hstack([sum_squares, sum_vals, max_vals])

# myproject/run.py
from mpbax.core.engine import Engine

# Config for single oracle with multiple outputs
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {'dir': 'checkpoints_multi_output', 'freq': 1, 'resume_from': None},
    'oracles': [
        {
            'name': 'multi_output_oracle',
            'input_dim': 2,
            'n_initial': 10,
            'function': {
                'class': 'myproject.oracles.multi_output_oracle'
            },
            'model': {
                'class': 'DummyModel',
                'params': {}
            }
        }
    ],
    'algorithm': {
        'class': 'GreedySampling',
        'params': {'input_dims': [2], 'n_propose': 5, 'n_candidates': 500}
    }
}

# NEW API: Engine takes only config
engine = Engine(config)
engine.run()
```

## Model Training Modes

### Finetuning vs. Retraining

mpBAX supports two model training modes:

**Retrain Mode (Default)**
```python
config['model'] = {
    'mode': 'retrain',  # Train from scratch each loop
    'checkpoint_mode': 'final'
}
```
- Creates new model instance every loop
- Trains on all accumulated data from scratch
- Simple and suitable for most models

**Finetune Mode**
```python
config['model'] = {
    'mode': 'finetune',  # Continue from previous model
    'checkpoint_mode': 'both'  # Save both best and final
}
```
- Reuses model from previous loop
- Continues training without resetting
- Preserves normalization params (e.g., initial X_mu, X_sigma)
- Ideal for neural networks and incremental learning

### Model with Normalization Example

```python
class ModelWithNormalization(BaseModel):
    def train(self, X, Y, metadata=None):
        # Compute normalization ONLY on first call
        if self.X_mu is None:
            self.X_mu = np.mean(X, axis=0, keepdims=True)
            self.X_sigma = np.std(X, axis=0, keepdims=True) + 1e-8
        # Use initial normalization for all subsequent calls
        X_norm = (X - self.X_mu) / self.X_sigma
        # ... train model ...
```

**Note:** The DANetModel plugin automatically implements this pattern. See DA_Net Plugin section for details.

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

### Checkpoint Directory Structure

Checkpoints are organized by oracle names for clarity:

```
checkpoints/
├── config.yaml          # Run configuration
├── state.pkl            # Run state
├── quadratic/           # Named by oracle name!
│   ├── data_0.pkl
│   ├── data_1.pkl
│   ├── model_0_best.pkl
│   ├── model_0_final.pkl
│   └── ...
```

For multi-oracle optimization:

```
checkpoints_multi/
├── obj1_2d/    # Oracle 1 named "obj1_2d"
├── obj2_3d/    # Oracle 2 named "obj2_3d"
```

**Important:**
- Oracle names must be unique
- Special characters are sanitized (e.g., "My Oracle #1" → "My_Oracle_1")
- Old checkpoints using `oracle_0/`, `oracle_1/` naming still load correctly (backward compatible)

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
│   │   ├── example_model_with_normalization.py
│   │   ├── example_da_net_optimization.py
│   │   └── example_checkpoint_workflow.py
│   ├── plugins/        # Model and algorithm plugins
│   │   └── models/
│   │       └── da_net_model.py  # PyTorch neural network plugin
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

# Model with normalization and finetuning
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/examples/example_model_with_normalization.py

# DA_Net plugin (PyTorch neural network)
PYTHONPATH=/path/to/mpBAX:$PYTHONPATH python mpbax/examples/example_da_net_optimization.py

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

**v2.0 Complete Configuration Structure:**

```yaml
# Run settings
seed: 42                    # Random seed for reproducibility
max_loops: 10               # Maximum number of optimization loops

# Checkpointing
checkpoint:
  dir: "checkpoints"        # Checkpoint directory
  freq: 1                   # Checkpoint every N loops
  resume_from: null         # null, "latest", or specific loop number

# Model training settings (global)
model:
  mode: "retrain"           # "retrain" (default) or "finetune"
                            # retrain: Create new model each loop
                            # finetune: Continue from previous model (preserves state)
  checkpoint_mode: "final"  # "final" (default), "best", or "both"
                            # final: Save model after training
                            # best: Save best model if tracked
                            # both: Save both final and best

# Algorithm configuration
algorithm:
  class: "GreedySampling"   # Algorithm class name (built-in or full import path)
  params:                   # Algorithm-specific parameters
    input_dims: [2, 3]      # List of input dimensions (one per oracle)
    n_propose: 10           # Number of candidates to propose per loop
    n_candidates: 1000      # Number of candidate points to evaluate

# Oracle configurations (list) - NEW in v2.0!
oracles:
  - name: "objective_1"     # Oracle name (used for checkpoint directories)
    input_dim: 2            # Input dimensionality for this oracle
    n_initial: 20           # Number of initial samples for this oracle

    # Oracle function (required)
    function:
      class: "myproject.oracles.oracle_1"  # Import path to function
      params: {}            # Optional: parameters for factory functions

    # Initial data generator (optional)
    generate: null          # null = use default uniform [0,1]^d
    # OR custom generator:
    # generate:
    #   class: "myproject.generators.lhs_generator"
    #   params:
    #     criterion: "maximin"

    # Model configuration (required)
    model:
      class: "DummyModel"   # Model class (built-in or full import path)
      params: {}            # Model-specific hyperparameters

  # Add more oracles for multi-oracle optimization
  - name: "objective_2"
    input_dim: 3            # Different dimension!
    n_initial: 30           # Different initial samples!
    function:
      class: "myproject.oracles.oracle_2"
    model:
      class: "DANetModel"   # Different model!
      params:
        epochs: 150
        epochs_iter: 10
        weight_new_data: 10.0
```

**Key v2.0 Changes:**
- `n_initial` moved to per-oracle configuration (different oracles can have different amounts)
- `function` section specifies oracle function import path (required)
- `generate` section allows per-oracle custom generators (optional)
- `model` section allows per-oracle model configuration with hyperparameters (required)

## Extending the Framework

### Custom Model

```python
# myproject/models.py
import numpy as np
from mpbax.core.model import BaseModel

class MyModel(BaseModel):
    def train(self, X: np.ndarray, Y: np.ndarray, metadata: dict = None) -> None:
        """Train the model.

        Args:
            X: Input data, shape (n, d)
            Y: Output data, shape (n, k) where k >= 1
            metadata: Optional dict with additional data information
                - 'loop_indices': Array indicating which loop each sample came from
        """
        # Your training logic
        # Can use metadata['loop_indices'] for sample weighting if desired
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs.

        Args:
            X: Input data, shape (n, d)

        Returns:
            Y: Predictions, shape (n, k) matching oracle output dimensions
        """
        # Your prediction logic
        pass
```

**Using custom model in config:**

```yaml
oracles:
  - name: "my_objective"
    input_dim: 4
    n_initial: 20
    function:
      class: "myproject.oracles.my_oracle"
    model:
      class: "myproject.models.MyModel"  # Full import path
      params:
        my_param1: 100
        my_param2: 0.01
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

## Model Plugins

mpBAX includes a plugin system for pre-built model implementations. Plugins provide production-ready models with optimized architectures and training procedures.

### DA_Net Plugin (PyTorch Neural Network)

The DA_Net plugin provides a deep neural network for multi-output regression with advanced features:

**Features:**
- 7-layer fully connected architecture with 800 neurons per layer
- Automatic input normalization (preserved in finetune mode)
- Early stopping with best model tracking
- GPU acceleration when available
- Multiple forward modes (fc, split, sine)
- Dropout and batch normalization

**Installation:**
```bash
pip install torch scikit-learn
```

**Usage with NEW config-first API:**
```python
# myproject/oracles.py
import numpy as np

def my_oracle(X):
    return np.sum(X**2, axis=1, keepdims=True)

# myproject/run.py
from mpbax.core.engine import Engine

# Config with DA_Net model - all params in config!
config = {
    'seed': 42,
    'max_loops': 5,
    'checkpoint': {'dir': 'checkpoints_danet', 'freq': 1, 'resume_from': None},
    'oracles': [
        {
            'name': 'my_obj',
            'input_dim': 2,
            'n_initial': 20,
            'function': {
                'class': 'myproject.oracles.my_oracle'
            },
            'model': {
                'class': 'DANetModel',  # Built-in model
                'params': {
                    'epochs': 150,        # Initial training epochs
                    'epochs_iter': 10,    # Finetuning epochs (NEW!)
                    'n_neur': 800,
                    'dropout': 0.1,
                    'lr': 1e-4,
                    'weight_new_data': 10.0,
                    'early_stop_patience': 10
                }
            }
        }
    ],
    'model': {
        'mode': 'finetune',  # Preserve normalization across loops
        'checkpoint_mode': 'both'  # Save best and final models
    },
    'algorithm': {
        'class': 'GreedySampling',
        'params': {'input_dims': [2], 'n_propose': 10, 'n_candidates': 500}
    }
}

# NEW API: Engine takes only config!
engine = Engine(config)
engine.run()
```

**Hyperparameters:**

All DA_Net hyperparameters are specified in the config under `oracles[i].model.params`:

```yaml
model:
  class: 'DANetModel'
  params:
    epochs: 150              # Training epochs for initial loop (pretraining)
    epochs_iter: 10          # Training epochs for later loops (finetuning) - NEW!
    n_neur: 800              # Neurons per hidden layer
    dropout: 0.1             # Dropout probability
    lr: 1e-4                 # Learning rate
    model_type: 'split'      # Forward mode: 'fc', 'split', 'sine'
    test_ratio: 0.05         # Train/test split ratio
    batch_size: 1000         # Training batch size
    early_stop_patience: 10  # Early stopping patience (None = disabled)
    device: null             # 'cuda', 'cpu', or null for auto-detect
    weight_new_data: 10.0    # Weight multiplier for recent loop data
    verbose: true            # Print training progress
```

**Forward Modes:**
- `'fc'`: Standard fully connected (all features processed equally)
- `'split'`: Concatenates last 2 input features before final layer (useful for spatial coordinates)
- `'sine'`: Uses sine activation instead of ReLU (experimental)

**Sample Weights for Recent Data:**

DANetModel automatically assigns higher weights to recently acquired data:
- Data from most recent loop: weight = `weight_new_data` (default: 10.0)
- Data from older loops: weight = 1.0

This emphasizes recent observations in the loss function, similar to importance weighting in active learning:

```python
# Example: Loop 3 with accumulated data
# Loop 0: 20 samples (weight=1.0)
# Loop 1: 10 samples (weight=1.0)
# Loop 2: 10 samples (weight=1.0)
# Loop 3: 10 samples (weight=10.0) ← Recent data emphasized
```

Set `weight_new_data=1.0` to disable weighting and treat all data equally.

**See also:** `mpbax/examples/example_da_net_optimization.py` for complete example

## Migrating from v1.x

v2.0 introduces a **config-first architecture**. Here's how to migrate:

### OLD v1.x Code

```python
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel

def my_oracle(X):
    return np.sum(X**2, axis=1, keepdims=True)

config = {
    'seed': 42,
    'max_loops': 10,
    'n_initial': 20,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1},
    'oracles': [{'name': 'my_obj', 'input_dim': 2}],
    'algorithm': {
        'class': 'GreedySampling',
        'params': {'input_dims': [2], 'n_propose': 10, 'n_candidates': 1000}
    }
}

# OLD API
engine = Engine(
    config=config,
    fn_oracles=[my_oracle],
    model_class=DummyModel,
    algorithm=None
)
engine.run()
```

### NEW v2.0 Code

**Step 1:** Move oracle function to importable module

```python
# myproject/oracles.py
import numpy as np

def my_oracle(X):
    return np.sum(X**2, axis=1, keepdims=True)
```

**Step 2:** Update config structure

```python
# myproject/run.py
from mpbax.core.engine import Engine

config = {
    'seed': 42,
    'max_loops': 10,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1},
    'oracles': [
        {
            'name': 'my_obj',
            'input_dim': 2,
            'n_initial': 20,  # Moved here!
            'function': {      # NEW!
                'class': 'myproject.oracles.my_oracle'
            },
            'model': {         # NEW!
                'class': 'DummyModel',
                'params': {}
            }
        }
    ],
    'algorithm': {
        'class': 'GreedySampling',
        'params': {'input_dims': [2], 'n_propose': 10, 'n_candidates': 1000}
    }
}

# NEW API - much simpler!
engine = Engine(config)
engine.run()
```

### Key Changes

1. **Oracle functions**: Must be in importable modules (not inline)
2. **n_initial**: Moved from top-level to per-oracle config
3. **function**: New required field specifying import path
4. **model**: New required field with class and params per oracle
5. **Engine API**: Now takes only `config` parameter

### Benefits

- ✅ Fully reproducible from config file
- ✅ Per-oracle flexibility (different n_initial, models, generators)
- ✅ Cleaner API: `Engine(config)` - that's it!
- ✅ Ready for CLI: `mpbax run config.yaml` (future feature)

## Design Principles

- **Simplicity First**: No databases, no async, no over-engineering
- **Minimal Dependencies**: numpy, pyyaml, pickle
- **Testable Components**: Each module independently tested
- **Clear Interfaces**: Consistent function signatures
- **Reproducible**: Seed management throughout

## Dependencies

**Core Dependencies:**
- Python 3.7+
- NumPy
- PyYAML

**Optional Dependencies:**
- PyTorch (for DA_Net plugin)
- scikit-learn (for DA_Net plugin's train/test split)

## License

See project license file.
