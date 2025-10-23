"""Model plugins for mpBAX."""

try:
    from .da_net_model import DANetModel
    __all__ = ['DANetModel']
except ImportError as e:
    # PyTorch not available
    import warnings
    warnings.warn(f"Could not import DANetModel: {e}. Install PyTorch to use this plugin.")
    __all__ = []
