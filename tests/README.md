# mpBAX Tests

Comprehensive test suite for the mpBAX framework.

## Test Structure

Tests are organized by component for better maintainability:

- **test_data_handler.py** - DataHandler: add/get data, loop tracking, save/load, multi-output
- **test_evaluator.py** - Evaluator: oracle evaluation, shape validation, eval counting
- **test_model.py** - Models: DummyModel, custom models, train/predict, save/load
- **test_algorithm.py** - Algorithms: RandomSampling, GreedySampling, proposals
- **test_checkpoint.py** - CheckpointManager: save/load/rollback, oracle naming
- **test_engine.py** - Engine integration: instance/string configs, multi-oracle, finetune/retrain modes

## Running Tests

**With installation (recommended):**

First, install mpBAX in development mode with test dependencies:
```bash
cd /path/to/mpBAX
pip install -e ".[dev]"
```

Then run tests:
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_engine.py -v

# Or run individual files directly
python tests/test_data_handler.py
python tests/test_evaluator.py
```

**Without installation:**

Set PYTHONPATH and run tests:
```bash
export PYTHONPATH=/path/to/mpBAX:$PYTHONPATH

# With pytest
pytest tests/ -v

# Or directly
python tests/test_engine.py
```

## Test Coverage

Tests cover:
- ✓ Core components (DataHandler, Evaluator, Model, Algorithm, CheckpointManager)
- ✓ Engine end-to-end integration
- ✓ Instance-based config (Python API)
- ✓ String-based config (YAML compatibility)
- ✓ Multi-oracle optimization
- ✓ Multi-output oracles
- ✓ Finetune vs retrain modes
- ✓ Loop tracking and metadata
- ✓ Checkpoint/resume/rollback functionality
- ✓ Shape validation and error handling

## Adding New Tests

When adding functionality to mpBAX:

1. Add test to appropriate test file (or create new file if needed)
2. Follow existing test patterns: clear names, assertions, print statements
3. Test both success and error cases
4. Run full test suite to ensure no regressions

Example:
```python
def test_my_feature():
    """Test description."""
    print("Testing my feature...")

    # Test code here
    assert something == expected

    print("  ✓ My feature tests passed")
```

## Test Principles

- **Simple and direct**: Each test function tests one thing
- **Instance-based**: Use direct instances, not string imports (easier to test)
- **Isolated**: Each test is independent, uses temp directories
- **Clear output**: Print statements show progress and what passed
