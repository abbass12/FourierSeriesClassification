"""
Neural Network Models Module
=============================
Implements three classification models:
- Model A: Raw signal data input
- Model B: Fourier coefficient input
- Model C: Fourier coefficients + jump discontinuity features

Uses PyTorch for GPU-accelerated training.
"""

import copy

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from torch.utils.data import DataLoader, TensorDataset


class SignalClassifier(nn.Module):
    """
    Feed-forward neural network for signal classification.
    Shared architecture for Models A and B (differ only in input_dim).
    """

    def __init__(self, input_dim: int, n_classes: int = 5,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Conv1DSignalClassifier(nn.Module):
    """Compact one-dimensional CNN baseline for raw sampled signals.

    The architecture is intentionally modest and is not a reimplementation of
    InceptionTime. It provides a convolutional baseline under the same split,
    optimizer, and early-stopping protocol as the MLP models.
    """

    def __init__(self, n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x))


class SignalClassifierWithJumps(nn.Module):
    """
    Model C: Two-branch architecture.
    Branch 1: Fourier coefficients -> Dense layers
    Branch 2: Jump features -> Dense layer
    Concatenated -> Shared hidden layers -> Output
    """

    def __init__(self, fourier_dim: int, jump_dim: int,
                 n_classes: int = 5,
                 hidden_dims: List[int] = [256, 128, 64],
                 jump_hidden: int = 32,
                 dropout: float = 0.2):
        super().__init__()

        # Jump branch
        self.jump_branch = nn.Sequential(
            nn.Linear(jump_dim, jump_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined layers
        combined_dim = fourier_dim + jump_hidden
        layers = []
        prev_dim = combined_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.main_network = nn.Sequential(*layers)

    def forward(self, fourier_features: torch.Tensor,
                jump_features: torch.Tensor) -> torch.Tensor:
        jump_out = self.jump_branch(jump_features)
        combined = torch.cat([fourier_features, jump_out], dim=1)
        return self.main_network(combined)


# ============================================================
# Training and Evaluation
# ============================================================

def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, n_epochs: int = 50,
                lr: float = 0.001, patience: int = 10,
                device: str = 'cpu',
                model_type: str = 'AB') -> Dict:
    """
    Train a model with early stopping.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        device: 'cpu' or 'cuda'
        model_type: 'AB' for Models A/B, 'C' for Model C

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            if model_type == 'C':
                fourier_feat, jump_feat, labels = batch
                fourier_feat = fourier_feat.to(device)
                jump_feat = jump_feat.to(device)
                labels = labels.to(device)
                outputs = model(fourier_feat, jump_feat)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if model_type == 'C':
                    fourier_feat, jump_feat, labels = batch
                    fourier_feat = fourier_feat.to(device)
                    jump_feat = jump_feat.to(device)
                    labels = labels.to(device)
                    outputs = model(fourier_feat, jump_feat)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        # Record history
        epoch_train_loss = train_loss / train_total
        epoch_val_loss = val_loss / val_total
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        scheduler.step(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Acc: {epoch_train_acc:.4f} - "
                  f"Val Acc: {epoch_val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   device: str = 'cpu',
                   model_type: str = 'AB') -> Tuple[float, np.ndarray]:
    """
    Evaluate model on test set.

    Returns:
        accuracy: Overall accuracy
        confusion_matrix: 5x5 confusion matrix
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'C':
                fourier_feat, jump_feat, labels = batch
                fourier_feat = fourier_feat.to(device)
                jump_feat = jump_feat.to(device)
                outputs = model(fourier_feat, jump_feat)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)

    # Confusion matrix
    n_classes = 5
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    return accuracy, cm


# ============================================================
# Data Preparation Utilities
# ============================================================

def prepare_dataloader(X: np.ndarray, y: np.ndarray,
                       batch_size: int = 32,
                       shuffle: bool = True) -> DataLoader:
    """Create a DataLoader from numpy arrays (for Models A/B)."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_dataloader_with_jumps(X_fourier: np.ndarray,
                                  X_jumps: np.ndarray,
                                  y: np.ndarray,
                                  batch_size: int = 32,
                                  shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for Model C (separate fourier and jump inputs)."""
    fourier_tensor = torch.FloatTensor(X_fourier)
    jump_tensor = torch.FloatTensor(X_jumps)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(fourier_tensor, jump_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
