"""
DA_Net Model Plugin for mpBAX

This plugin provides a PyTorch-based deep neural network model for multi-output regression.
The model supports:
- Multi-dimensional output (output_dim >= 1)
- Progressive dimension reduction architecture for improved capacity
- Flexible per-dimension sigmoid activation for output normalization
- Input normalization (preserved across loops in finetune mode)
- Early stopping with best model tracking
- Multiple forward modes (fc, split, sine)
- GPU acceleration when available
"""

import numpy as np
import copy
import time
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy base class for when PyTorch is not available
    class nn:
        class Module:
            pass
    class TorchDataset:
        pass

from mpbax.core.model import BaseModel


# ============================================================================
# PyTorch Dataset
# ============================================================================

class Dataset(TorchDataset):
    """PyTorch dataset for training."""

    def __init__(self, X, Y, weights=None):
        """
        Args:
            X: Input features, shape (n, d)
            Y: Target values, shape (n,) or (n, k)
            weights: Optional sample weights, shape (n,)
        """
        self.X = torch.from_numpy(X).float()

        if Y.ndim == 1:
            self.Y = torch.from_numpy(Y).float().unsqueeze(1)
        else:
            self.Y = torch.from_numpy(Y).float()

        if weights is not None:
            self.weights = torch.from_numpy(weights).float()
        else:
            self.weights = torch.ones(self.Y.shape[0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.weights[idx]


# ============================================================================
# Normalization Utilities
# ============================================================================

def get_norm(X, eps=1e-8):
    """Compute mean and std for normalization.

    Args:
        X: Input array, shape (n, d)
        eps: Small constant for numerical stability

    Returns:
        X_mu: Mean, shape (1, d)
        X_sigma: Std, shape (1, d)
    """
    X_mu = np.mean(X, axis=0, keepdims=True)
    X_sigma = np.std(X, axis=0, keepdims=True) + eps
    return X_mu, X_sigma


def normalize(X, X_mu, X_sigma):
    """Apply normalization to X.

    Args:
        X: Input array, shape (n, d)
        X_mu: Mean, shape (1, d)
        X_sigma: Std, shape (1, d)

    Returns:
        X_normalized: Normalized array, shape (n, d)
    """
    return (X - X_mu) / X_sigma


# ============================================================================
# DA_Net Architecture
# ============================================================================

class DA_Net(nn.Module):
    """Deep neural network for multi-output regression.

    Architecture: Progressive dimension reduction with optional dropout and batch norm.
    Supports multiple forward modes: fc, split, sine.
    Supports multi-dimensional output with optional per-dimension sigmoid activation.
    """

    def __init__(self, dropout=0.1, train_noise=0, n_feat=6, n_neur=800,
                 device=None, model_type='fc', out_scale=1, output_dim=1,
                 sigmoid_dims=None):
        """
        Args:
            dropout: Dropout probability
            train_noise: Training noise level (currently unused)
            n_feat: Number of input features
            n_neur: Number of neurons in hidden layers
            device: PyTorch device (cuda/cpu)
            model_type: Forward mode - 'fc', 'split', or 'sine'
            out_scale: Output scaling factor
            output_dim: Number of output dimensions (default=1)
            sigmoid_dims: Sigmoid activation control:
                - None or False: No sigmoid on any dimension (default)
                - True or 'all': Apply sigmoid to all dimensions
                - list/tuple of bool: Per-dimension control (must match output_dim length)
                  Example: [True, False] applies sigmoid to dim 0, not dim 1
        """
        super(DA_Net, self).__init__()

        self.dropout = dropout
        self.train_noise = train_noise
        self.n_feat = n_feat
        self.n_neur = n_neur
        self.device = device if device else torch.device('cpu')
        self.model_type = model_type
        self.out_scale = out_scale
        self.output_dim = output_dim

        # Process sigmoid_dims into a boolean mask
        if sigmoid_dims is None or sigmoid_dims is False:
            self.sigmoid_mask = None
        elif sigmoid_dims is True or sigmoid_dims == 'all':
            self.sigmoid_mask = torch.ones(output_dim, dtype=torch.bool)
        else:
            # List/tuple of booleans
            if len(sigmoid_dims) != output_dim:
                raise ValueError(
                    f"sigmoid_dims list length ({len(sigmoid_dims)}) must match "
                    f"output_dim ({output_dim})"
                )
            self.sigmoid_mask = torch.tensor(sigmoid_dims, dtype=torch.bool)

        # Define layers with progressive dimension reduction
        self.fc1 = nn.Linear(n_feat, n_neur)
        self.fc2 = nn.Linear(n_neur, n_neur)
        self.fc3 = nn.Linear(n_neur, n_neur)
        self.fc4 = nn.Linear(n_neur, int(n_neur/2))  # Progressive reduction
        self.fc5 = nn.Linear(int(n_neur/2), int(n_neur/4))  # Progressive reduction

        if model_type in ['split', 'sine']:
            # Split/sine modes: concatenate spatial coords (last 2 features) after fc5
            self.fc6 = nn.Linear(int(n_neur/4) + 2, int(n_neur/4))
        else:
            self.fc6 = nn.Linear(int(n_neur/4), int(n_neur/4))

        self.fc_out = nn.Linear(int(n_neur/4), output_dim)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(n_neur)  # After fc2 (outputs n_neur)
        self.bn2 = nn.BatchNorm1d(int(n_neur/2))  # After fc4 (outputs n_neur/2)

    def forward_fc(self, x):
        """Standard fully connected forward pass with progressive dimension reduction."""
        x = torch.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.bn1(torch.relu(self.fc2(x)))
        x = self.dropout_layer(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout_layer(x)
        x = self.bn2(torch.relu(self.fc4(x)))
        x = self.dropout_layer(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout_layer(x)
        x = torch.relu(self.fc6(x))
        x = self.dropout_layer(x)
        x = self.fc_out(x)

        # Apply selective sigmoid activation
        if self.sigmoid_mask is not None:
            sigmoid_mask = self.sigmoid_mask.to(x.device)
            x = torch.where(sigmoid_mask, torch.sigmoid(x), x)

        return x * self.out_scale

    def forward_split(self, x):
        """Split mode: concatenate spatial coordinates after fc5 before fc6."""
        xy = x[:, -2:]  # Last 2 features (spatial coords)
        x = torch.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.bn1(torch.relu(self.fc2(x)))
        x = self.dropout_layer(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout_layer(x)
        x = self.bn2(torch.relu(self.fc4(x)))
        x = self.dropout_layer(x)
        x = torch.relu(self.fc5(x))
        x = torch.cat((x, xy), dim=1)  # Concatenate spatial coords after fc5
        x = torch.relu(self.fc6(x))
        x = self.dropout_layer(x)
        x = self.fc_out(x)

        # Apply selective sigmoid activation
        if self.sigmoid_mask is not None:
            sigmoid_mask = self.sigmoid_mask.to(x.device)
            x = torch.where(sigmoid_mask, torch.sigmoid(x), x)

        return x * self.out_scale

    def forward_sine(self, x):
        """Sine activation mode: uses sin() instead of relu, concatenates spatial coords."""
        xy = x[:, -2:]  # Last 2 features (spatial coords)
        x = torch.sin(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.bn1(torch.sin(self.fc2(x)))
        x = self.dropout_layer(x)
        x = torch.sin(self.fc3(x))
        x = self.dropout_layer(x)
        x = self.bn2(torch.sin(self.fc4(x)))
        x = self.dropout_layer(x)
        x = torch.sin(self.fc5(x))
        x = torch.cat((x, xy), dim=1)  # Concatenate spatial coords after fc5
        x = torch.sin(self.fc6(x))
        x = self.dropout_layer(x)
        x = self.fc_out(x)

        # Apply selective sigmoid activation
        if self.sigmoid_mask is not None:
            sigmoid_mask = self.sigmoid_mask.to(x.device)
            x = torch.where(sigmoid_mask, torch.sigmoid(x), x)

        return x * self.out_scale

    def forward(self, x):
        """Forward pass using configured model type."""
        if self.model_type == 'fc':
            return self.forward_fc(x)
        elif self.model_type == 'split':
            return self.forward_split(x)
        elif self.model_type == 'sine':
            return self.forward_sine(x)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")


# ============================================================================
# Loss Function
# ============================================================================

def myloss(y_pred, y_target, weights=None):
    """Weighted MSE loss.

    Args:
        y_pred: Predictions, shape (batch, k)
        y_target: Targets, shape (batch, k)
        weights: Sample weights, shape (batch,)

    Returns:
        loss: Scalar loss value
    """
    if weights is not None:
        # Weighted MSE with normalization by mean weight
        # This keeps loss magnitude similar to unweighted case
        squared_error = (y_pred - y_target) ** 2
        weighted_squared_error = squared_error * weights.unsqueeze(1) / torch.mean(weights)
        loss = torch.mean(weighted_squared_error)
    else:
        loss = torch.mean((y_pred - y_target) ** 2)
    return loss


# ============================================================================
# Training Function
# ============================================================================

def train_NN_re(model, trainloader, testloader, lr=1e-4, epochs=150,
                savefile=None, final_savefile=None, device=None,
                verbose=True, early_stop_patience=None, log_period=10,
                eval_mode_on_test=True, use_batch_loss=False):
    """Train neural network with early stopping.

    Args:
        model: DA_Net model instance
        trainloader: Training data loader
        testloader: Test data loader
        lr: Learning rate
        epochs: Number of training epochs
        savefile: Path to save best model
        final_savefile: Path to save final model (optional)
        device: PyTorch device
        verbose: Print training progress
        early_stop_patience: Patience for early stopping (None = no early stopping)
        log_period: Print log every N epochs
        eval_mode_on_test: Use model.eval() during test evaluation
        use_batch_loss: Log batch-wise loss instead of accumulated loss

    Returns:
        model: Trained model (loaded with best weights if savefile provided)
    """
    if device is None:
        device = model.device

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch, weights_batch in trainloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            weights_batch = weights_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = myloss(Y_pred, Y_batch, weights_batch)
            loss.backward()
            optimizer.step()

            if use_batch_loss:
                train_loss = loss.item()
            else:
                train_loss += loss.item() * X_batch.size(0)

        if not use_batch_loss:
            train_loss /= len(trainloader.dataset)

        # Testing
        if eval_mode_on_test:
            model.eval()

        test_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch, _ in testloader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                Y_pred = model(X_batch)
                loss = myloss(Y_pred, Y_batch)
                test_loss += loss.item() * X_batch.size(0)

        test_loss /= len(testloader.dataset)

        # Logging
        if verbose and (epoch % log_period == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d}/{epochs}: train_loss={train_loss:.6f}, test_loss={test_loss:.6f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            if savefile:
                torch.save(model.state_dict(), savefile)
        else:
            patience_counter += 1

        # Early stopping
        if early_stop_patience is not None and patience_counter >= early_stop_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={early_stop_patience})")
            break

    # Save final model
    if final_savefile:
        torch.save(model.state_dict(), final_savefile)

    # Load best model if saved
    if savefile:
        model.load_state_dict(torch.load(savefile, map_location=device, weights_only=True))
        if verbose:
            print(f"  Loaded best model with test_loss={best_test_loss:.6f}")

    return model


# ============================================================================
# DANetModel: mpBAX BaseModel Interface
# ============================================================================

class DANetModel(BaseModel):
    """DA_Net model plugin for mpBAX framework.

    Features:
    - PyTorch-based deep neural network with progressive dimension reduction
    - Multi-dimensional output support (output_dim >= 1)
    - Flexible per-dimension sigmoid activation for output range control
    - Automatic input normalization (preserved in finetune mode)
    - Early stopping with best model tracking
    - GPU acceleration when available
    - Configurable architecture and training parameters

    Example usage:
        # Single output, no sigmoid
        model = DANetModel(input_dim=4)

        # Two outputs, sigmoid on both (normalize to [0, 1])
        model = DANetModel(input_dim=4, sigmoid_dims=True)

        # Two outputs, sigmoid only on second dimension
        model = DANetModel(input_dim=4, sigmoid_dims=[False, True])
    """

    def __init__(self, input_dim: int,
                 n_neur: int = 800,
                 dropout: float = 0.1,
                 lr: float = 1e-4,
                 epochs: int = 150,
                 epochs_iter: int = 10,
                 model_type: str = 'split',
                 out_scale: float = 1.0,
                 sigmoid_dims=None,
                 test_ratio: float = 0.05,
                 batch_size: int = 1000,
                 early_stop_patience: Optional[int] = 10,
                 random_state: int = 1,
                 device: Optional[str] = None,
                 verbose: bool = True,
                 weight_new_data: float = 10.0):
        """
        Args:
            input_dim: Input dimensionality
            n_neur: Number of neurons in hidden layers
            dropout: Dropout probability
            lr: Learning rate
            epochs: Number of training epochs for initial training (pretraining phase)
            epochs_iter: Number of training epochs for later iterations (finetuning phase)
            model_type: Forward mode - 'fc', 'split', or 'sine'
            out_scale: Output scaling factor
            sigmoid_dims: Sigmoid activation control for output layer:
                - None or False: No sigmoid on any dimension (default)
                - True or 'all': Apply sigmoid to all output dimensions
                - list/tuple of bool: Per-dimension control (e.g., [True, False] for 2D output)
                  Note: length must match output_dim inferred from training data
            test_ratio: Fraction of data for test set
            batch_size: Training batch size
            early_stop_patience: Patience for early stopping (None = disabled)
            random_state: Random seed for train/test split
            device: Device specification ('cuda', 'cpu', or None for auto)
            verbose: Print training progress
            weight_new_data: Weight multiplier for most recent loop's data (default: 10.0)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DANetModel. "
                "Install it with: pip install torch"
            )

        super().__init__(input_dim)

        # Hyperparameters
        self.n_neur = n_neur
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.epochs_iter = epochs_iter
        self.model_type = model_type
        self.out_scale = out_scale
        self.sigmoid_dims = sigmoid_dims
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience
        self.random_state = random_state
        self.verbose = verbose
        self.weight_new_data = weight_new_data

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Normalization params (computed once from initial data)
        self.X_mu = None
        self.X_sigma = None

        # Output dimensionality (inferred from Y during training)
        self.output_dim = None

        # PyTorch model
        self.network = None

        # Best model state
        self.best_test_loss = float('inf')
        self.best_network_state = None

    def train(self, X: np.ndarray, Y: np.ndarray, metadata: dict = None) -> None:
        """Train model with input normalization and early stopping.

        Args:
            X: Input data with shape (n, d)
            Y: Output data with shape (n, k) where k >= 1
            metadata: Optional metadata dict with 'loop_indices' for sample weighting
        """
        self._validate_data(X, Y)

        # Filter out samples with NaN values in Y
        valid_mask = ~np.any(np.isnan(Y), axis=1)
        n_invalid = np.sum(~valid_mask)
        if n_invalid > 0:
            if self.verbose:
                print(f"  [DANetModel] Filtering {n_invalid} samples with NaN labels (keeping {np.sum(valid_mask)}/{len(Y)} samples)")
            X = X[valid_mask]
            Y = Y[valid_mask]
            if metadata and 'loop_indices' in metadata:
                metadata = metadata.copy()
                metadata['loop_indices'] = metadata['loop_indices'][valid_mask]

        if len(X) == 0:
            raise ValueError("No valid training samples after NaN filtering")

        # Infer output dimensionality
        if self.output_dim is None:
            self.output_dim = Y.shape[1]

        # Compute normalization ONLY on first call
        if self.X_mu is None:
            self.X_mu, self.X_sigma = get_norm(X, eps=1e-8)
            if self.verbose:
                print(f"  [DANetModel] Computing initial normalization:")
                print(f"    X_mu shape: {self.X_mu.shape}, X_sigma shape: {self.X_sigma.shape}")
        else:
            if self.verbose:
                print(f"  [DANetModel] Using existing normalization (from initial data)")

        # Normalize inputs
        X_norm = normalize(X, self.X_mu, self.X_sigma)

        # Flatten Y if it's a single output
        if Y.shape[1] == 1:
            Y_flat = Y.flatten()
        else:
            Y_flat = Y

        # Compute sample weights from loop indices
        sample_weights = None
        if metadata and 'loop_indices' in metadata:
            loop_indices = metadata['loop_indices']
            max_loop = np.max(loop_indices)
            # Assign higher weights to samples from the most recent loop
            sample_weights = np.where(
                loop_indices == max_loop,
                self.weight_new_data,
                1.0
            )
            if self.verbose:
                n_new = np.sum(loop_indices == max_loop)
                n_old = len(loop_indices) - n_new
                print(f"  [DANetModel] Sample weights: {n_new} new samples (weight={self.weight_new_data}), {n_old} old samples (weight=1.0)")

        # Train/test split
        from sklearn.model_selection import train_test_split
        if sample_weights is not None:
            X_train, X_test, Y_train, Y_test, weights_train, weights_test = train_test_split(
                X_norm, Y_flat, sample_weights, test_size=self.test_ratio, random_state=self.random_state
            )
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                X_norm, Y_flat, test_size=self.test_ratio, random_state=self.random_state
            )
            weights_train = None
            weights_test = None

        # Create data loaders
        trainset = Dataset(X_train, Y_train, weights=weights_train)
        testset = Dataset(X_test, Y_test, weights=weights_test)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

        # Initialize network if first training
        if self.network is None:
            self.network = DA_Net(
                dropout=self.dropout,
                train_noise=0,
                n_feat=self.input_dim,
                n_neur=self.n_neur,
                device=self.device,
                model_type=self.model_type,
                out_scale=self.out_scale,
                output_dim=self.output_dim,
                sigmoid_dims=self.sigmoid_dims
            ).to(self.device)

            if self.verbose:
                print(f"  [DANetModel] Initialized network on {self.device}")

            # First training: use full epochs
            epochs_to_use = self.epochs
            if self.verbose:
                print(f"  [DANetModel] Initial training for {epochs_to_use} epochs...")
        else:
            # Later training: use iteration epochs
            epochs_to_use = self.epochs_iter
            if self.verbose:
                print(f"  [DANetModel] Finetuning for {epochs_to_use} epochs...")

        # Use temporary files for best/final model during training
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            savefile = os.path.join(tmpdir, 'best_model.pt')

            self.network = train_NN_re(
                self.network,
                trainloader,
                testloader,
                lr=self.lr,
                epochs=epochs_to_use,
                savefile=savefile,
                final_savefile=None,
                device=self.device,
                verbose=self.verbose,
                early_stop_patience=self.early_stop_patience,
                log_period=10,
                eval_mode_on_test=True,
                use_batch_loss=False
            )

            # Save best model state for get_best_model_snapshot()
            if os.path.exists(savefile):
                self.best_network_state = torch.load(savefile, map_location='cpu', weights_only=True)
                # Also compute best test loss
                self.network.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for X_batch, Y_batch, _ in testloader:
                        X_batch = X_batch.to(self.device)
                        Y_batch = Y_batch.to(self.device)
                        Y_pred = self.network(X_batch)
                        loss = myloss(Y_pred, Y_batch)
                        test_loss += loss.item() * X_batch.size(0)
                self.best_test_loss = test_loss / len(testloader.dataset)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using normalized inputs.

        Args:
            X: Input data with shape (n, d)

        Returns:
            Y: Predicted outputs with shape (n, k)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        self._validate_input(X)

        # Normalize using same params as training
        X_norm = normalize(X, self.X_mu, self.X_sigma)

        # Convert to tensor
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)

        # Predict
        self.network.eval()
        with torch.no_grad():
            Y_pred = self.network(X_tensor)

        # Convert back to numpy
        Y_np = Y_pred.cpu().numpy()

        # Ensure output has shape (n, k)
        if Y_np.ndim == 1:
            Y_np = Y_np.reshape(-1, 1)

        return Y_np

    def get_state(self) -> dict:
        """Get model state including normalization params and network weights.

        Returns:
            Dictionary with X_mu, X_sigma, network state, best model info
        """
        state = {
            'X_mu': self.X_mu,
            'X_sigma': self.X_sigma,
            'output_dim': self.output_dim,
            'best_test_loss': self.best_test_loss,
            'is_trained': self.is_trained
        }

        if self.network is not None:
            state['network_state_dict'] = self.network.state_dict()

        if self.best_network_state is not None:
            state['best_network_state'] = self.best_network_state

        return state

    def set_state(self, state_dict: dict) -> None:
        """Restore model state.

        Args:
            state_dict: Dictionary containing model state
        """
        self.X_mu = state_dict.get('X_mu')
        self.X_sigma = state_dict.get('X_sigma')
        self.output_dim = state_dict.get('output_dim')
        self.best_test_loss = state_dict.get('best_test_loss', float('inf'))
        self.is_trained = state_dict.get('is_trained', False)

        if 'network_state_dict' in state_dict:
            # Initialize network if needed
            if self.network is None:
                self.network = DA_Net(
                    dropout=self.dropout,
                    train_noise=0,
                    n_feat=self.input_dim,
                    n_neur=self.n_neur,
                    device=self.device,
                    model_type=self.model_type,
                    out_scale=self.out_scale,
                    output_dim=self.output_dim,
                    sigmoid_dims=self.sigmoid_dims
                ).to(self.device)

            self.network.load_state_dict(state_dict['network_state_dict'])

        if 'best_network_state' in state_dict:
            self.best_network_state = state_dict['best_network_state']

    def get_best_model_snapshot(self) -> Optional['DANetModel']:
        """Get best model snapshot from training.

        Returns:
            Copy of model with best network weights, or None if not available
        """
        if self.best_network_state is None:
            return None

        # Create copy with best weights
        best_model = DANetModel(
            input_dim=self.input_dim,
            n_neur=self.n_neur,
            dropout=self.dropout,
            lr=self.lr,
            epochs=self.epochs,
            epochs_iter=self.epochs_iter,
            model_type=self.model_type,
            out_scale=self.out_scale,
            sigmoid_dims=self.sigmoid_dims,
            test_ratio=self.test_ratio,
            batch_size=self.batch_size,
            early_stop_patience=self.early_stop_patience,
            random_state=self.random_state,
            device=str(self.device),
            verbose=False,
            weight_new_data=self.weight_new_data
        )

        # Copy normalization params
        best_model.X_mu = self.X_mu
        best_model.X_sigma = self.X_sigma
        best_model.output_dim = self.output_dim
        best_model.best_test_loss = self.best_test_loss
        best_model.is_trained = True

        # Initialize and load best network
        best_model.network = DA_Net(
            dropout=self.dropout,
            train_noise=0,
            n_feat=self.input_dim,
            n_neur=self.n_neur,
            device=best_model.device,
            model_type=self.model_type,
            out_scale=self.out_scale,
            output_dim=self.output_dim,
            sigmoid_dims=self.sigmoid_dims
        ).to(best_model.device)

        best_model.network.load_state_dict(self.best_network_state)
        best_model.best_network_state = copy.deepcopy(self.best_network_state)

        return best_model
