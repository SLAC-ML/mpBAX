# DANetModel Plugin

Deep neural network surrogate model for mpBAX using PyTorch.

## Features

- **Deep architecture**: Multi-layer feedforward network with dropout
- **Input normalization**: Fixes normalization parameters from initial data
- **Sample weighting**: Emphasize recent data in loss function
- **Adaptive epochs**: Different training durations for initial vs incremental training
- **Finetune support**: Preserves network state across optimization loops

## Requirements

```bash
pip install torch
```

## Usage

### Basic Example

```python
from mpbax.core.engine import Engine

config = {
    'oracles': [{
        'name': 'my_oracle',
        'input_dim': 5,
        'n_initial': 100,
        'function': {'class': my_oracle_fn},
        'model': {
            'class': 'DANetModel',  # Built-in name
            'params': {
                'epochs': 150,      # Initial training
                'epochs_iter': 10,  # Incremental training
                'n_neur': 800,      # Network width
                'dropout': 0.1
            }
        }
    }],
    'model': {'mode': 'finetune'},  # Recommended for DANetModel
    # ...
}

engine = Engine(config)
engine.run()
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | int | 150 | Training epochs for initial model |
| `epochs_iter` | int | 10 | Training epochs for incremental updates |
| `n_neur` | int | 400 | Number of neurons per hidden layer |
| `dropout` | float | 0.1 | Dropout rate for regularization |
| `weight_new_data` | float | 10.0 | Weight multiplier for recent samples |
| `lr` | float | 0.001 | Adam optimizer learning rate |

## Training Modes

### Finetune Mode (Recommended)

```python
config['model'] = {'mode': 'finetune'}
```

**Benefits:**
- Preserves input normalization (X_mu, X_sigma from loop 0)
- Continues neural network training without resetting
- Faster convergence on new data
- Consistent predictions across loops

**How it works:**
1. Loop 0: Compute X_mu, X_sigma; train network from scratch
2. Loop 1+: Reuse X_mu, X_sigma; continue training existing network

### Retrain Mode

```python
config['model'] = {'mode': 'retrain'}  # or omit (default)
```

Creates fresh network each loop. Generally less efficient for DANetModel.

## Sample Weighting

DANetModel can emphasize recent data in the loss function:

```python
'model': {
    'class': 'DANetModel',
    'params': {
        'weight_new_data': 10.0  # Recent samples weighted 10x
    }
}
```

**Rationale:**
- Optimization explores different regions over time
- Recent samples more relevant to current search area
- Similar to importance weighting in active learning

**Implementation:**
```python
# Loop tracking automatically provided via metadata
weights = np.where(loop_indices == max_loop, 10.0, 1.0)
loss = (weights * (predictions - targets)**2).mean()
```

**Disable weighting:**
```python
'weight_new_data': 1.0  # All samples weighted equally
```

## Adaptive Epochs

Use different epoch counts for initial vs incremental training:

```python
'epochs': 150,      # Thorough initial training
'epochs_iter': 10   # Quick incremental updates
```

**Benefits:**
- Initial model needs more training (fitting from scratch)
- Incremental updates only need small adjustments
- Saves computation time in later loops

## Architecture Details

**Network structure:**
```
Input (normalized)
  ↓
Dense(n_neur) + Dropout
  ↓
Dense(n_neur) + Dropout
  ↓
Dense(output_dim)
```

**Training details:**
- Optimizer: Adam with default lr=0.001
- Loss: MSE (weighted by sample weights if enabled)
- Device: Automatically uses GPU if available, else CPU
- Normalization: Fixed from initial data (finetune mode)

## Checkpointing

DANetModel supports both checkpoint modes:

```python
config['model'] = {
    'mode': 'finetune',
    'checkpoint_mode': 'both'  # or 'final', 'best'
}
```

- **'final'**: Save model after training completes
- **'best'**: Save best model during training (if tracked)
- **'both'**: Save both final and best models

## Example: Full Configuration

```python
config = {
    'seed': 42,
    'max_loops': 50,
    'checkpoint': {'dir': 'checkpoints', 'freq': 1},
    'model': {
        'mode': 'finetune',
        'checkpoint_mode': 'both'
    },
    'oracles': [{
        'name': 'expensive_simulation',
        'input_dim': 10,
        'n_initial': 200,
        'function': {'class': my_simulation},
        'model': {
            'class': 'DANetModel',
            'params': {
                'epochs': 200,
                'epochs_iter': 15,
                'n_neur': 1000,
                'dropout': 0.15,
                'weight_new_data': 10.0,
                'lr': 0.0005
            }
        }
    }],
    'algorithm': {
        'class': 'GreedySampling',
        'params': {
            'input_dims': [10],
            'n_propose': 20,
            'n_candidates': 2000
        }
    }
}
```

## Tips

1. **Start with defaults**: Try default parameters first
2. **Use finetune mode**: Almost always better for DANetModel
3. **Adjust network size**: Increase `n_neur` for complex functions
4. **Monitor training**: Check if epochs are sufficient
5. **Sample weighting**: Reduce if optimization explores uniformly

## See Also

- Main README: Framework overview
- `examples/06_danet_model.py`: Complete working example
- `mpbax/plugins/models/da_net_model.py`: Source code
