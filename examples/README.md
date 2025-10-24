# mpBAX Examples

All examples are self-contained single-file scripts demonstrating different mpBAX features.

## Running Examples

Make sure mpBAX is in your PYTHONPATH:
```bash
export PYTHONPATH=/path/to/mpBAX:$PYTHONPATH
```

Then run any example:
```bash
python examples/01_basic_optimization.py
python examples/02_multi_oracle.py
# etc.
```

## Example Overview

### 01_basic_optimization.py
**The simplest example** - Start here!
- Single oracle optimization
- Quadratic function minimization
- DummyModel + GreedySampling
- ~60 lines total

### 02_multi_oracle.py
**Multiple independent oracles**
- Two oracles: Sphere and Rosenbrock functions
- Each oracle has its own model and data
- Single algorithm proposes for both
- Useful for multi-objective optimization

### 03_multi_output.py
**Vector-valued objectives**
- Oracle returns multiple outputs (shape (n, k) with k>1)
- Model predicts all outputs simultaneously
- Useful for multi-task learning scenarios

### 04_checkpointing.py
**Saving and resuming**
- Automatic checkpointing every loop
- Resume from latest checkpoint
- Resume from specific loop number
- Essential for long-running optimizations

### 05_custom_model.py
**Extending the framework**
- Define custom BaseModel subclass
- Implement train() and predict() methods
- Example: Simple GP-like model with RBF kernel
- Shows how easy it is to add new models

### 06_danet_model.py
**Deep learning plugin** (requires PyTorch)
- DANetModel plugin for neural network surrogate
- Demonstrates finetune mode
- Sample weighting (recent data emphasized)
- Adaptive epochs (100 initial, 20 incremental)

### notebook_example.ipynb
**Jupyter notebook version**
- Interactive example in notebook format
- Same basic optimization as Example 01
- Great for exploration and visualization

## What to Try Next

After running the examples, try:

1. **Modify parameters**: Change `n_initial`, `max_loops`, `n_propose`
2. **Change oracle functions**: Try different test functions
3. **Experiment with models**: Swap DummyModel for your custom model
4. **Add visualization**: Plot optimization progress
5. **Real applications**: Replace toy functions with your actual simulations

## Need Help?

- See main README.md for API reference
- Check `tests/` for more code examples
- See `mpbax/plugins/models/README.md` for DANetModel details
