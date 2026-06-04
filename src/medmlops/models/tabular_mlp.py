"""PyTorch tabular MLP hero model for CPU-only training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from torch import nn

from medmlops.seeds import set_deterministic


@dataclass(frozen=True)
class MLPConfig:
    """Training configuration for the tabular MLP."""

    hidden: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    epochs: int = 6
    batch_size: int = 1024
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    use_class_weight: bool = True
    seed: int = 42


class TabularMLP(nn.Module):
    """Dense MLP over the sklearn-transformed tabular matrix."""

    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = n_features
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""

        return self.network(x).squeeze(-1)


def to_dense_float32(matrix: Any) -> NDArray[np.float32]:
    """Convert a sklearn sparse or dense output to a float32 numpy array."""

    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return np.asarray(dense, dtype=np.float32)


def _positive_class_weight(y: NDArray[np.float32]) -> torch.Tensor | None:
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives == 0.0:
        return None
    return torch.tensor(negatives / positives, dtype=torch.float32)


def train_tabular_mlp(
    x_train: Any,
    y_train: ArrayLike,
    x_val: Any,
    y_val: ArrayLike,
    config: MLPConfig,
) -> TabularMLP:
    """Train a deterministic CPU MLP and return the fitted model."""

    del x_val, y_val
    set_deterministic(config.seed)

    x_array = to_dense_float32(x_train)
    y_array = np.asarray(y_train, dtype=np.float32)
    model = TabularMLP(
        n_features=x_array.shape[1],
        hidden=config.hidden,
        dropout=config.dropout,
    )
    model.train()

    pos_weight = _positive_class_weight(y_array) if config.use_class_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    x_tensor = torch.from_numpy(x_array)
    y_tensor = torch.from_numpy(y_array)

    for _epoch in range(config.epochs):
        for start in range(0, len(y_tensor), config.batch_size):
            stop = min(start + config.batch_size, len(y_tensor))
            batch_x = x_tensor[start:stop]
            batch_y = y_tensor[start:stop]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict_proba_mlp(model: TabularMLP, matrix: Any) -> NDArray[np.float64]:
    """Predict positive-class probabilities with the MLP."""

    model.eval()
    x_array = to_dense_float32(matrix)
    with torch.inference_mode():
        logits = model(torch.from_numpy(x_array))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)
