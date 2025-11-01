"""Model utilities for circadian phase estimation using CNN + BiGRU."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _get_torch_module():
    """Lazy import torch so linters don't fail when it isn't installed."""

    try:
        module = import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "PyTorch is required for the circadian phase estimator. Install it via "
            "'pip install torch'."
        ) from exc
    return module


torch = _get_torch_module()
nn = torch.nn
Adam = torch.optim.Adam
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset
random_split = torch.utils.data.random_split

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for training the circadian phase estimator."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    gradient_clip: float = 1.0
    seed: int = 42
    device: Optional[str] = None


class PhaseDataset(Dataset):
    """Wrap sliding-window sequences as a PyTorch dataset."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        if sequences.ndim != 3:
            raise ValueError("Sequences must have shape (n_samples, seq_len, n_features)")
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("Targets must be (n_samples, 2) representing sin and cos components")
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.sequences.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.sequences[index], self.targets[index]


class CircadianPhaseEstimator(nn.Module):
    """CNN + BiGRU architecture predicting circadian phase encodings."""

    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(gru_hidden * 2, 2)

    def forward(self, x: Any) -> Any:
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # -> (batch, features, seq_len)
        conv_features = self.cnn(x).transpose(1, 2)  # -> (batch, seq_len, channels)
        gru_out, _ = self.gru(conv_features)
        pooled = gru_out.mean(dim=1)
        pooled = self.dropout(pooled)
        output = self.regressor(pooled)
        return normalize_predictions(output)


def normalize_predictions(raw_output: Any) -> Any:
    """Ensure predictions lie on the unit circle."""

    magnitude = torch.linalg.norm(raw_output, dim=1, keepdim=True).clamp_min(1e-6)
    return raw_output / magnitude


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic behaviour where possible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def circular_mae(predictions: Any, targets: Any) -> Any:
    """Compute circular MAE between predicted and true phase encodings."""

    pred_angles = torch.atan2(predictions[:, 0], predictions[:, 1])
    target_angles = torch.atan2(targets[:, 0], targets[:, 1])
    delta = torch.atan2(torch.sin(pred_angles - target_angles), torch.cos(pred_angles - target_angles))
    return torch.abs(delta).mean()


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Derive evaluation metrics from np arrays of sin/cos predictions."""

    pred_angle = np.arctan2(predictions[:, 0], predictions[:, 1])
    target_angle = np.arctan2(targets[:, 0], targets[:, 1])
    mae_minutes = np.abs(angle_difference(pred_angle, target_angle)) * (720 / math.pi)
    corr = circular_correlation(pred_angle, target_angle)
    return {
        "circular_mae_minutes": float(mae_minutes.mean()),
        "circular_correlation": float(corr),
    }


def angle_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.arctan2(np.sin(a - b), np.cos(a - b))
    return diff


def circular_correlation(alpha: np.ndarray, beta: np.ndarray) -> float:
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    sin_b, cos_b = np.sin(beta), np.cos(beta)
    num = np.mean(sin_a * sin_b) - np.mean(sin_a) * np.mean(sin_b) + np.mean(cos_a * cos_b) - np.mean(cos_a) * np.mean(cos_b)
    den = math.sqrt(
        (
            (np.mean(sin_a**2) - np.mean(sin_a) ** 2 + np.mean(cos_a**2) - np.mean(cos_a) ** 2)
            * (np.mean(sin_b**2) - np.mean(sin_b) ** 2 + np.mean(cos_b**2) - np.mean(cos_b) ** 2)
        )
    )
    return float(num / den) if den != 0 else 0.0


def prepare_datasets(
    sequences: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
) -> Tuple[PhaseDataset, PhaseDataset, PhaseDataset]:
    """Split sequences into train/validation/test datasets chronologically."""

    set_seed(config.seed)
    dataset = PhaseDataset(sequences, targets)
    total = len(dataset)
    test_size = int(total * config.test_ratio)
    val_size = int(total * config.validation_ratio)
    train_size = total - val_size - test_size
    if train_size <= 0:
        raise ValueError("Dataset too small for requested splits.")

    lengths = [train_size, val_size, test_size]
    return tuple(random_split(dataset, lengths, generator=torch.Generator().manual_seed(config.seed)))  # type: ignore[return-value]


def train_model(
    dataset: PhaseDataset,
    config: TrainingConfig,
    validation_dataset: Optional[PhaseDataset] = None,
) -> Tuple[Any, List[Dict[str, float]]]:
    """Train the circadian phase estimator model."""

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = (
        DataLoader(validation_dataset, batch_size=config.batch_size)
        if validation_dataset is not None
        else None
    )

    model = CircadianPhaseEstimator(input_dim=dataset.sequences.shape[2])
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: List[Dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = circular_mae(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(loader.dataset)

        metrics = {"epoch": epoch + 1, "train_loss": avg_loss}
        if val_loader is not None:
            metrics.update(evaluate_model(model, val_loader, device))
        history.append(metrics)
        LOGGER.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", metrics["epoch"], metrics["train_loss"], metrics.get("val_loss", float("nan")))
    return model, history


@torch.no_grad()
def evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
) -> Dict[str, float]:
    """Evaluate model on a given dataloader."""

    model.eval()
    total_loss = 0.0
    preds: List[Any] = []
    trues: List[Any] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        loss = circular_mae(outputs, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        preds.append(outputs.cpu())
        trues.append(batch_y.cpu())

    predictions = torch.cat(preds).numpy()
    targets = torch.cat(trues).numpy()
    metrics = compute_metrics(predictions, targets)
    metrics["val_loss"] = total_loss / len(loader.dataset)
    return metrics


def save_model(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    LOGGER.info("Model weights saved to %s", path)


def load_model(path: Path, input_dim: int) -> nn.Module:
    model = CircadianPhaseEstimator(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
