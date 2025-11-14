# mpBAX - Multipoint Bayesian Algorithm Execution

A lightweight framework for model-based multi-objective optimization with built-in checkpointing and reproducibility.

**Key Features:**
- **Config-First Design**: Entire experiment specified in config (dict or YAML)
- **Multi-Oracle Support**: Optimize multiple independent objectives with different input dimensions
- **Checkpointing**: Automatic save/resume/rollback functionality
- **Reproducible**: Built-in seed management
- **Extensible**: Easy to add custom models, algorithms, and generators

## Installation

**Option 1: pip install (Recommended)**

Install in development mode (editable):
```bash
cd /path/to/mpBAX
pip install -e .
```

Install normally:
```bash
pip install .
```

With optional PyTorch support (for DANetModel):
```bash
pip install -e ".[torch]"
```

With development tools (pytest):
```bash
pip install -e ".[dev]"
```

All optional dependencies:
```bash
pip install -e ".[all]"
```

**Option 2: PYTHONPATH (No installation)**

If you prefer not to install:
```bash
export PYTHONPATH=/path/to/mpBAX:$PYTHONPATH
```

**Dependencies:**
- Core: `numpy>=1.20.0`, `pyyaml>=5.0`
- Optional: `torch>=1.13.0` (for DANetModel plugin)
- Dev: `pytest>=7.0` (for running tests)

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

### Flexible Config Patterns

mpBAX v2 introduces flexible parameter placement for cleaner configs:

**Default Generator Shortcut:**
```python
# Use default uniform generator with custom number of samples
'oracles': [{
    'input_dim': 2,
    'generate': {'params': {'n': 20}},  # No need to specify 'class'
    ...
}]
```

**Parameter-Level Placement:**
```python
# Put input_dim where it's used (in model params)
'oracles': [{
    'generate': {'params': {'n': 20, 'd': 2}},
    'model': {
        'class': DummyModel,
        'params': {'input_dim': 2}  # Input dim specified here
    }
}]
```

**Custom Generator with All Params:**
```python
# Pass all params (n, d, custom) to generator
'generate': {
    'class': lhs_generator,
    'params': {'n': 20, 'd': 4, 'criterion': 'maximin'}
}
```

**Cleaner Top-Level Config:**
```python
# Use 'training' instead of 'model' to avoid confusion
config = {
    'training': {'mode': 'finetune'},  # Clear and unambiguous
    'oracles': [...]
}
```

All old config patterns remain supported for backward compatibility.

## Examples

See [examples/](examples/) directory:

- `01_basic_optimization.py` - Simple single-objective example
- `02_multi_oracle.py` - Multi-objective optimization
- `03_multi_output.py` - Oracle with multiple outputs
- `04_checkpointing.py` - Resume and rollback
- `05_custom_model.py` - Implementing custom models
- `06_danet_model.py` - Using DANetModel plugin

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

#### Multi-Output vs Multi-Oracle

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

### Complete Example

Here's a complete config demonstrating all available options:

```yaml
# Top-level settings
seed: 42
max_loops: 10

# Checkpointing configuration
checkpoint:
  dir: 'checkpoints'
  freq: 1
  resume_from: null  # or 'latest' or loop number

# Global training settings (renamed from 'model' to avoid confusion)
training:
  mode: 'retrain'  # 'retrain' or 'finetune'
  checkpoint_mode: 'final'  # 'final', 'best', or 'both'

# Oracle configurations (list - one per oracle)
oracles:
  - name: 'my_objective'
    input_dim: 4        # Can also be in model.params or generate.params.d
    n_initial: 20       # Can also be in generate.params.n

    # Oracle function
    function:
      class: 'myproject.oracles.my_oracle'  # or direct instance
      params: {}        # Empty dict calls factory with no args

    # Optional: Custom generator (null = uniform [0,1]^d)
    generate: null
    # Alternative patterns:
    #   generate:
    #     params: {n: 20}  # Use default generator with custom n
    #   generate:
    #     class: 'myproject.generators.lhs_generator'
    #     params: {n: 20, d: 4, criterion: 'maximin'}  # All params here

    # Model for this oracle
    model:
      class: 'DummyModel'  # or 'DANetModel' or custom
      params: {}        # Can include input_dim here instead of oracle level

# Algorithm configuration
algorithm:
  class: 'GreedySampling'
  params:
    input_dims: [4]
    n_propose: 10
    n_candidates: 1000
```

### Top-Level Config

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Random seed for reproducibility |
| `max_loops` | int | Maximum optimization loops |
| `checkpoint` | dict | Checkpointing configuration |
| `training` | dict | Global training settings (formerly `model`) |
| `oracles` | list | Oracle configurations (see below) |
| `algorithm` | dict | Algorithm configuration |

**Note:** `training` was renamed from `model` to avoid confusion with per-oracle `model` field. The old name is still supported with a deprecation warning.

### Checkpoint Config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | str | required | Checkpoint directory path |
| `freq` | int | 1 | Save frequency (1=every loop) |
| `resume_from` | str/int/null | null | Resume point ('latest', loop number, or null) |

**Usage examples:**

Resume from latest checkpoint:
```python
config['checkpoint']['resume_from'] = 'latest'
```

Resume from specific loop:
```python
config['checkpoint']['resume_from'] = 5
```

Rollback to previous checkpoint:
```python
from mpbax.core.checkpoint import CheckpointManager
manager = CheckpointManager('checkpoints')
manager.delete_checkpoints_after(loop=3)  # Delete loops 4+
```

### Training Config (Global)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | str | 'retrain' | Training mode: 'retrain' or 'finetune' |
| `checkpoint_mode` | str | 'final' | Model checkpoint: 'final', 'best', or 'both' |

**Training Modes:**

**Retrain** (default) - Creates fresh model instance each loop, trains on all accumulated data from scratch. Simple and suitable for most models.
```python
config['training'] = {'mode': 'retrain'}
```

**Finetune** - Reuses model instance from previous loop, continues training without resetting weights. Preserves normalization parameters (e.g., X_mu, X_sigma from loop 0). Ideal for neural networks.
```python
config['training'] = {'mode': 'finetune'}
```

**Checkpoint Modes:**
- `'final'`: Save model after training completes (used for resumption)
- `'best'`: Save best model if tracked via `get_best_model_snapshot()`
- `'both'`: Save both final and best models

### Oracle Config

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Oracle identifier (used for checkpoint dirs) |
| `input_dim` | int (optional) | Input dimensionality (can be in model.params or generate.params.d) |
| `n_initial` | int (optional) | Number of initial samples (can be in generate.params.n) |
| `function` | dict | `{'class': func_or_path, 'params': {}}` |
| `generate` | dict/null | Custom generator or null for uniform [0,1]^d |
| `model` | dict | `{'class': model_class_or_path, 'params': {}}` |

**Flexible Parameter Placement:**

The framework supports multiple ways to specify `input_dim` and `n_initial`:

```yaml
# Pattern 1: Traditional (oracle level)
oracles:
  - input_dim: 4
    n_initial: 20

# Pattern 2: input_dim in model.params
oracles:
  - model:
      params: {input_dim: 4}

# Pattern 3: n in generate.params (default generator shortcut)
oracles:
  - input_dim: 4
    generate:
      params: {n: 20}

# Pattern 4: All params in custom generator
oracles:
  - generate:
      class: 'myproject.generators.custom'
      params: {n: 20, d: 4, scale: 0.5}
    model:
      params: {input_dim: 4}
```

At least one location must specify each parameter. The framework validates that required params exist somewhere in the config.

### Algorithm Config

| Field | Type | Description |
|-------|------|-------------|
| `class` | str/class | Algorithm class or import path |
| `params` | dict | Algorithm parameters |

Built-in algorithms:
- `RandomSampling`: Random proposals
- `GreedySampling`: Greedy selection from candidates (requires `n_candidates`)

## Advanced Features

### Sample Weighting

Models can access loop tracking metadata to weight recent data. This is useful for adaptive optimization where recent samples are more informative about promising regions.

```python
def train(self, X, Y, metadata=None):
    if metadata and 'loop_indices' in metadata:
        loop_indices = metadata['loop_indices']
        max_loop = np.max(loop_indices)
        # Weight recent data higher (e.g., 10x)
        weights = np.where(loop_indices == max_loop, 10.0, 1.0)
        # Use weights in loss function
```

The DataHandler automatically tracks which loop each sample came from, making this information available to models during training.

See [DANetModel](mpbax/plugins/models/README.md) for a complete implementation example using PyTorch with sample weighting.

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

Generators are functions that generate initial samples. They receive all params during the actual call (not during instantiation):

```python
def lhs_generator(n, d, criterion='maximin'):
    """Latin Hypercube Sampling generator.

    Args:
        n: Number of samples to generate
        d: Input dimensionality
        criterion: LHS sampling criterion

    Returns:
        X with shape (n, d)
    """
    # Generate samples using LHS
    return X
```

Use in config:
```python
'generate': {
    'class': lhs_generator,  # or import path
    'params': {
        'n': 20,            # Number of samples
        'd': 4,             # Dimensionality
        'criterion': 'maximin'  # Custom parameter
    }
}
```

The generator receives **all** params specified in the config during the call. This allows flexible signatures beyond the standard `(n, d)` pattern.

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
