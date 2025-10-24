# mpBAX - Multipoint Bayesian Algorithm Execution

A lightweight framework for model-based multi-objective optimization with built-in checkpointing and reproducibility.

**Key Features:**
- **Config-First Design**: Entire experiment specified in config (dict or YAML)
- **Multi-Oracle Support**: Optimize multiple independent objectives with different input dimensions
- **Checkpointing**: Automatic save/resume/rollback functionality
- **Reproducible**: Built-in seed management
- **Extensible**: Easy to add custom models, algorithms, and generators

## Installation

No installation required - add mpBAX to your PYTHONPATH:

```bash
export PYTHONPATH=/path/to/mpBAX:$PYTHONPATH
```

**Dependencies:**
- Core: `numpy`, `pyyaml`
- Optional: `torch` (for DANetModel plugin)

## Quick Start

Create a single-file optimization script:

```python
import numpy as np
from mpbax.core.engine import Engine
from mpbax.core.model import DummyModel
from mpbax.core.algorithm import GreedySampling

# Define oracle function
def quadratic(X: np.ndarray) -> np.ndarray:
    """X has shape (n, d), returns Y with shape (n, 1)"""
    return np.sum((X - 0.5)**2, axis=1, keepdims=True)

# Configure optimization
config = {
    'seed': 42,
    'max_loops': 10,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1},
    'oracles': [{
        'name': 'quadratic',
        'input_dim': 2,
        'n_initial': 20,
        'function': {'class': quadratic},    # Pass function directly
        'model': {'class': DummyModel}       # Pass class directly
    }],
    'algorithm': {
        'class': GreedySampling,
        'params': {'input_dims': [2], 'n_propose': 10, 'n_candidates': 1000}
    }
}

# Run optimization
engine = Engine(config)
engine.run()
```

That's it! mpBAX will:
1. Generate 20 initial samples
2. Evaluate them with `quadratic` function
3. Train a model on the data
4. Use the model to propose 10 new candidates
5. Repeat for 10 loops total
6. Save checkpoints every loop

**For larger projects**, use YAML configs and import paths instead of instances:

```yaml
# config.yaml
oracles:
  - function:
      class: 'myproject.oracles.quadratic'  # Import path string
```

See [examples/](examples/) for more examples.

## Key Concepts

### Oracle Function Requirements

Oracle functions are the core of mpBAX - they represent your expensive simulation or objective function.

**Shape Convention:**
- **Input X**: Shape `(n, d)` where `n` = number of samples, `d` = input dimension
- **Output Y**: Shape `(n, k)` where `k ≥ 1` (number of outputs per sample)

**Example - Single Output:**
```python
def my_oracle(X: np.ndarray) -> np.ndarray:
    """
    Args:
        X: Input with shape (n, d)
    Returns:
        Y: Output with shape (n, 1)
    """
    return np.sum(X**2, axis=1, keepdims=True)  # Shape (n, 1)
```

**Example - Multi-Output:**
```python
def multi_output_oracle(X: np.ndarray) -> np.ndarray:
    """
    Args:
        X: Input with shape (n, d)
    Returns:
        Y: Output with shape (n, 3)  # Three outputs per input
    """
    y1 = np.sum(X**2, axis=1, keepdims=True)
    y2 = np.sum(X, axis=1, keepdims=True)
    y3 = np.max(X, axis=1, keepdims=True)
    return np.hstack([y1, y2, y3])  # Shape (n, 3)
```

**Important:** Always use `keepdims=True` or reshape to ensure Y is 2D, never 1D.

### Config-First Architecture

Everything is specified in a config dict/YAML:
- **Oracle functions**: Import paths (YAML) or direct instances (Python)
- **Models**: Per-oracle model configuration with hyperparameters
- **Generators**: Custom initial sample generators (optional)
- **Algorithm**: Proposal strategy with parameters

### Multi-Oracle Optimization

Each oracle operates independently with its own:
- Input dimension (can differ between oracles)
- Number of initial samples
- Model and hyperparameters
- Data storage and checkpoints

Algorithms receive predictions from all models and propose candidates for each oracle.

### Checkpointing System

Automatic checkpointing enables:
- **Resume**: Continue from latest checkpoint
- **Rollback**: Delete checkpoints after a specific loop
- **Reproducibility**: Complete experiment state saved

Directory structure:
```
checkpoints/
├── config.yaml
├── state.pkl
├── oracle_name_1/
│   ├── data_0.pkl, data_1.pkl, ...
│   ├── model_0_final.pkl, model_1_final.pkl, ...
│   └── model_0_best.pkl, ...  (if checkpoint_mode='both')
└── oracle_name_2/
    └── ...
```

## Configuration Reference

### Top-Level Config

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Random seed for reproducibility |
| `max_loops` | int | Maximum optimization loops |
| `checkpoint` | dict | Checkpointing configuration |
| `model` | dict | Global model training settings |
| `oracles` | list | Oracle configurations (see below) |
| `algorithm` | dict | Algorithm configuration |

### Checkpoint Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | str | required | Checkpoint directory path |
| `freq` | int | 1 | Save frequency (1=every loop) |
| `resume_from` | str/int/null | null | Resume point ('latest', loop number, or null) |

### Model Config (Global)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | str | 'retrain' | 'retrain' or 'finetune' |
| `checkpoint_mode` | str | 'final' | 'final', 'best', or 'both' |

### Oracle Config

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Oracle identifier (used for checkpoint dirs) |
| `input_dim` | int | Input dimensionality |
| `n_initial` | int | Number of initial samples |
| `function` | dict | `{'class': func_or_path, 'params': {}}` |
| `generate` | dict/null | Custom generator or null for uniform [0,1]^d |
| `model` | dict | `{'class': model_class_or_path, 'params': {}}` |

### Algorithm Config

| Field | Type | Description |
|-------|------|-------------|
| `class` | str/class | Algorithm class or import path |
| `params` | dict | Algorithm parameters |

Built-in algorithms:
- `RandomSampling`: Random proposals
- `GreedySampling`: Greedy selection from candidates (requires `n_candidates`)

## Advanced Features

### Training Modes

**Retrain Mode** (default):
- Creates fresh model instance each loop
- Trains on all accumulated data from scratch
- Simple and suitable for most models

```python
config['model'] = {'mode': 'retrain'}
```

**Finetune Mode**:
- Reuses model instance from previous loop
- Continues training without resetting weights
- Preserves normalization parameters (e.g., X_mu, X_sigma from loop 0)
- Ideal for neural networks

```python
config['model'] = {'mode': 'finetune'}
```

### Checkpoint Modes

- `'final'`: Save model after training (used for resumption)
- `'best'`: Save best model (if model implements `get_best_model_snapshot()`)
- `'both'`: Save both final and best

### Sample Weighting

Models can access loop tracking metadata to weight recent data:

```python
def train(self, X, Y, metadata=None):
    if metadata and 'loop_indices' in metadata:
        loop_indices = metadata['loop_indices']
        max_loop = np.max(loop_indices)
        # Weight recent data higher
        weights = np.where(loop_indices == max_loop, 10.0, 1.0)
```

See [DANetModel](mpbax/plugins/models/README.md) for example implementation.

### Checkpointing Operations

**Resume from latest:**
```python
config['checkpoint']['resume_from'] = 'latest'
engine = Engine(config)
engine.run()
```

**Resume from specific loop:**
```python
config['checkpoint']['resume_from'] = 5
```

**Rollback:**
```python
from mpbax.core.checkpoint import CheckpointManager

manager = CheckpointManager('checkpoints')
manager.delete_checkpoints_after(loop=3)  # Delete loops 4+
```

## Developer Guide

### Project Structure

```
mpBAX/
├── mpbax/                  # Core package
│   ├── core/               # Core components
│   │   ├── engine.py       # Main orchestrator
│   │   ├── evaluator.py    # Oracle wrapper
│   │   ├── data_handler.py # Data storage
│   │   ├── model.py        # Model interface
│   │   ├── algorithm.py    # Algorithm interface
│   │   └── checkpoint.py   # Checkpointing
│   └── plugins/            # Extensions
│       └── models/         # Model plugins
│           └── da_net.py   # Neural network model
├── examples/               # Example scripts
├── tests/                  # Test suite
├── README.md
└── config.yaml             # Config template
```

### Design Principles

- **Simplicity**: Use the simplest and most robust approach
- **Minimal Dependencies**: numpy, pyyaml, pickle - that's it
- **No Assumptions**: Common API, no isinstance checks
- **Shape Conventions**: X is (n, d), Y is (n, k) where k ≥ 1

### Extending mpBAX

**Custom Model:**

```python
from mpbax.core.model import BaseModel

class MyModel(BaseModel):
    def __init__(self, input_dim, my_param=1.0):
        super().__init__(input_dim)
        self.my_param = my_param

    def train(self, X, Y, metadata=None):
        # Training logic here
        self.is_trained = True

    def predict(self, X):
        # Prediction logic here
        return Y_pred  # Shape (n, k)
```

Use in config:
```python
'model': {
    'class': MyModel,  # or 'myproject.models.MyModel'
    'params': {'my_param': 2.0}
}
```

**Custom Algorithm:**

```python
from mpbax.core.algorithm import BaseAlgorithm

class MyAlgorithm(BaseAlgorithm):
    def __init__(self, input_dims, n_propose, my_param=1.0):
        super().__init__(input_dims, n_propose)
        self.my_param = my_param

    def propose(self, fn_pred_list):
        # fn_pred_list: list of predict functions (one per oracle)
        # Return: list of X arrays (one per oracle)
        X_list = []
        for i, fn_pred in enumerate(fn_pred_list):
            X_candidates = self.generate_candidates(self.input_dims[i])
            # Selection logic here
            X_proposed = self.select_best(X_candidates, fn_pred)
            X_list.append(X_proposed)
        return X_list
```

**Custom Generator:**

```python
def lhs_generator(n, d, criterion='maximin'):
    """Latin Hypercube Sampling generator."""
    # Return X with shape (n, d)
    return X
```

Use in config:
```python
'generate': {
    'class': lhs_generator,  # or import path
    'params': {'criterion': 'maximin'}
}
```

### Testing

Run tests:
```bash
# All tests
python -m pytest tests/

# Specific component
python tests/test_model.py

# With pytest
pytest tests/ -v
```

Test structure:
- `test_data_handler.py` - DataHandler tests
- `test_evaluator.py` - Evaluator tests
- `test_model.py` - Model interface tests
- `test_algorithm.py` - Algorithm tests
- `test_checkpoint.py` - Checkpointing tests
- `test_engine.py` - Integration tests

See [tests/README.md](tests/README.md) for details.

### Plugins

**DANetModel** - PyTorch neural network surrogate:
- Automatic input normalization (preserved across loops)
- Sample weighting for recent data
- Adaptive epochs (initial vs finetuning)
- Early stopping
- GPU support

See [mpbax/plugins/models/README.md](mpbax/plugins/models/README.md) for full documentation.

**Adding a Plugin:**

1. Create file in `mpbax/plugins/{category}/`
2. Inherit from appropriate base class
3. Implement required methods
4. Document in plugin README

## Examples

See [examples/](examples/) directory:

- `01_basic_optimization.py` - Simple single-objective example
- `02_multi_oracle.py` - Multi-objective optimization
- `03_multi_output.py` - Oracle with multiple outputs
- `04_checkpointing.py` - Resume and rollback
- `05_custom_model.py` - Implementing custom models
- `06_danet_model.py` - Using DANetModel plugin

## Multi-Output vs Multi-Oracle

**Multi-Output**: Single oracle returns multiple values
```python
def oracle(X):  # Shape (n, d)
    return Y    # Shape (n, 3) - three outputs per input
```

**Multi-Oracle**: Multiple independent oracles
```python
config['oracles'] = [
    {'name': 'obj1', 'input_dim': 2, ...},
    {'name': 'obj2', 'input_dim': 3, ...}
]
```
