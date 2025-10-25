# YAML Configuration Example

This example demonstrates the **YAML-based workflow** for mpBAX, where the entire experiment is specified in a YAML configuration file.

## When to Use This Approach

**Use YAML configs when:**
- Working in production or shared environments
- Need configuration management and version control
- Running experiments on clusters or via job schedulers
- Collaborating with team members who modify configs
- Want to separate code from configuration

**Use Python API (examples 01-06) when:**
- Rapid prototyping and experimentation
- Oracle functions are simple lambdas or closures
- Prefer everything in one script for simplicity
- Doing interactive development

## File Structure

```
yaml_example/
├── oracles.py      # Oracle function definitions
├── config.yaml     # Complete experiment configuration
├── run.py          # Script to run optimization
└── README.md       # This file
```

## Key Differences from Python API

### YAML Approach (This Example)

**config.yaml:**
```yaml
oracles:
  - function:
      class: 'examples.yaml_example.oracles.quadratic'  # Import path string
    model:
      class: 'DummyModel'  # Built-in model name
```

**oracles.py:**
```python
def quadratic(X):
    return np.sum((X - 0.5)**2, axis=1, keepdims=True)
```

**run.py:**
```python
engine = Engine('config.yaml')  # Load from file
engine.run()
```

### Python API Approach (Examples 01-06)

**Single Python file:**
```python
def quadratic(X):
    return np.sum((X - 0.5)**2, axis=1, keepdims=True)

config = {
    'oracles': [{
        'function': {'class': quadratic},  # Pass instance directly
        'model': {'class': DummyModel}     # Pass class directly
    }]
}

engine = Engine(config)  # Pass dict
engine.run()
```

## Running This Example

### Option 1: With Installation (Recommended)

```bash
# Install mpBAX
cd /path/to/mpBAX
pip install -e .

# Run the example
cd examples/yaml_example
python run.py
```

### Option 2: With PYTHONPATH

```bash
# Set PYTHONPATH
export PYTHONPATH=/path/to/mpBAX:$PYTHONPATH

# Run the example
cd examples/yaml_example
python run.py
```

## What Happens

1. `run.py` loads configuration from `config.yaml`
2. Engine imports oracle function using the path: `examples.yaml_example.oracles.quadratic`
3. Runs 10 optimization loops with:
   - 20 initial samples
   - 10 new proposals per loop
   - DummyModel surrogate
   - GreedySampling algorithm
4. Saves checkpoints to `checkpoints_yaml/`
5. Prints best result found

## Modifying the Example

**Change oracle function:**
1. Add new function to `oracles.py`
2. Update `config.yaml` with new import path

**Change model:**
1. Update `model.class` in `config.yaml`
2. Add model parameters under `model.params`

**Add second oracle:**
1. Add new oracle function to `oracles.py`
2. Add second oracle config in `config.yaml` under `oracles`
3. Update algorithm `input_dims` parameter

## Benefits of YAML Workflow

1. **Separation of Concerns**: Code (oracles.py) separate from config (config.yaml)
2. **Version Control**: Track config changes independently from code
3. **Reproducibility**: Complete experiment specified in single YAML file
4. **Modularity**: Reuse oracle functions across different configs
5. **Collaboration**: Team members can modify configs without touching code
6. **Cluster-Friendly**: Easy to submit batch jobs with different configs

## See Also

- Main README.md for complete API reference
- Examples 01-06 for Python API approach
- `mpbax/plugins/models/README.md` for advanced model usage
