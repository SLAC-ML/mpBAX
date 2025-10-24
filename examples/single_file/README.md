# Single-File Examples

These examples demonstrate mpBAX's **instance-based config** approach, where you can define everything in one Python file without needing separate modules.

## When to Use

✅ **Use single-file approach when:**
- Prototyping and experimenting
- Creating simple examples or tutorials
- Working in Jupyter notebooks
- Oracle functions are short and simple
- Project is small-scale

❌ **Use modular approach (separate modules + YAML) when:**
- Building production systems
- Oracle functions are complex or reused
- Need to share configs across projects
- Working with a team
- Need strict reproducibility from config files

## Example: `basic_single_file.py`

Demonstrates the simplest possible mpBAX setup:
- Oracle function defined in same file
- Model and algorithm classes imported directly
- Everything runs in ~30 lines of code

```bash
python basic_single_file.py
```

## Key Features

### Direct Instance Passing

Instead of import path strings:
```python
# Old approach (still supported)
'function': {
    'class': 'myproject.oracles.my_oracle'  # String
}
```

You can pass instances directly:
```python
# New approach
def my_oracle(X):
    return X**2

'function': {
    'class': my_oracle  # Direct reference!
}
```

### Works with Everything

You can pass instances for:
- **Oracle functions**: Regular functions, lambdas, or factory functions
- **Generators**: Functions returning X samples
- **Models**: Any BaseModel subclass
- **Algorithms**: Any BaseAlgorithm subclass

### Mixing Approaches

You can mix instances and strings in the same config:
```python
config = {
    'oracles': [{
        'function': {'class': my_oracle},      # Instance
        'model': {'class': 'DummyModel'}       # String
    }],
    'algorithm': {'class': GreedySampling}     # Instance
}
```

## Benefits

1. **Faster Development**: No need to create separate module files
2. **Easier Learning**: Everything visible in one place
3. **Notebook-Friendly**: Works perfectly in Jupyter notebooks
4. **Still Pythonic**: Direct object references are more natural

## See Also

- `/examples/basic/`: Modular approach with separate files
- `/examples/multi_oracle/`: Multi-objective examples
- Main README.md: Complete API reference
