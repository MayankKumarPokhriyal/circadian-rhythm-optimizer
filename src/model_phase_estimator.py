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


def train_model_core(
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=False,
    )
    best_val = float("inf")
    best_state = None
    patience = 7
    bad_epochs = 0
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
            # Scheduler and early stopping on validation loss
            scheduler.step(metrics["val_loss"])  # type: ignore[arg-type]
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
        history.append(metrics)
        LOGGER.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", metrics["epoch"], metrics["train_loss"], metrics.get("val_loss", float("nan")))
        if val_loader is not None and bad_epochs >= patience:
            LOGGER.info("Early stopping triggered at epoch %d", metrics["epoch"]) 
            break
    # Restore best weights if available
    if best_state is not None:
        model.load_state_dict(best_state)
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


def _infer_target_vectors(df: np.ndarray | None, hours: Optional[np.ndarray], radians: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError  # placeholder (not used)


def _build_sequences_from_df(
    df: "np.ndarray",
    feature_cols: List[str],
    target_vecs: np.ndarray,
    user_ids: np.ndarray,
    seq_len: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences per user without crossing boundaries."""

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    start = 0
    # Assumes df is a structured array-like aligned with user_ids
    for uid in np.unique(user_ids):
        idx = np.where(user_ids == uid)[0]
        if idx.size == 0:
            continue
        uX = df[idx][:, [feature_cols.index(c) for c in feature_cols]] if isinstance(df, np.ndarray) else df.loc[idx, feature_cols].values
        uy = target_vecs[idx]
        for i in range(0, len(idx) - seq_len + 1, step):
            X_list.append(uX[i : i + seq_len])
            y_list.append(uy[i + seq_len - 1])
    return np.stack(X_list), np.stack(y_list)


def train_model(
    features_csv: str | Path,
    save_path: str | Path = "model/circadian_phase_estimator.pt",
    seq_len: int = 240,
    step: int = 5,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """Train from a features CSV and save the model.

    Expects features CSV to include:
    - user_id
    - datetime
    - numerical feature columns
    - one of: phase_rad, phase_hours, or solar_phase_offset (fallback proxy)
    """
    import pandas as pd

    config = config or TrainingConfig()
    df = pd.read_csv(features_csv, parse_dates=["datetime"])  # type: ignore[arg-type]
    df = df.sort_values(["user_id", "datetime"]).reset_index(drop=True)

    # Determine target
    if "phase_rad" in df.columns:
        radians = df["phase_rad"].to_numpy()
    elif "phase_hours" in df.columns:
        radians = (df["phase_hours"].to_numpy() % 24.0) / 24.0 * 2 * math.pi
    elif "solar_phase_offset" in df.columns:
        # Proxy: map [-12, 12) offset to [0, 24)
        hours = ((df["solar_phase_offset"].to_numpy() + 12.0) % 24.0)
        radians = hours / 24.0 * 2 * math.pi
        LOGGER.warning("Using solar_phase_offset as a proxy target; please provide ground-truth phase if available.")
    else:
        raise KeyError("No target column found. Provide 'phase_rad' or 'phase_hours' (or 'solar_phase_offset' as proxy).")

    target_vecs = np.stack([np.sin(radians), np.cos(radians)], axis=1)

    # Feature columns: numeric excluding obvious non-features
    non_feats = {"user_id", "datetime", "phase_rad", "phase_hours", "solar_phase_offset"}
    feature_cols = [c for c in df.columns if c not in non_feats and np.issubdtype(df[c].dtype, np.number)]
    X_array = df[feature_cols].values.astype(np.float32)
    users = df["user_id"].to_numpy()

    # Build sequences per user
    sequences: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for uid, g in df.groupby("user_id", sort=False):
        gX = g[feature_cols].values.astype(np.float32)
        gy = target_vecs[g.index]
        for i in range(0, len(g) - seq_len + 1, step):
            sequences.append(gX[i : i + seq_len])
            targets.append(gy[i + seq_len - 1])
    sequences_np = np.stack(sequences)
    targets_np = np.stack(targets)

    # Split datasets
    train_ds, val_ds, test_ds = prepare_datasets(sequences_np, targets_np, config)

    # Train with early stopping & scheduler
    model, history = train_model_core(train_ds, config, validation_dataset=val_ds)

    # Evaluate on test
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)
    # Collect predictions for plotting
    model.eval()
    preds: List[Any] = []
    trues: List[Any] = []
    total_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            by = by.to(device)
            out = model(bx)
            loss = circular_mae(out, by)
            total_loss += loss.item() * bx.size(0)
            preds.append(out.cpu())
            trues.append(by.cpu())
    pred_np = torch.cat(preds).numpy()
    true_np = torch.cat(trues).numpy()
    test_metrics = compute_metrics(pred_np, true_np)
    test_metrics["val_loss"] = total_loss / len(test_loader.dataset)

    # Convert to hours for viz
    pred_angles = np.arctan2(pred_np[:, 0], pred_np[:, 1])
    true_angles = np.arctan2(true_np[:, 0], true_np[:, 1])
    pred_hours = (pred_angles % (2 * math.pi)) / (2 * math.pi) * 24.0
    true_hours = (true_angles % (2 * math.pi)) / (2 * math.pi) * 24.0

    # Save model
    save_path = Path(save_path)
    save_model(model, save_path)

    return {
        "feature_cols": feature_cols,
        "history": history,
        "test_metrics": test_metrics,
        "model_path": str(save_path),
        "input_dim": len(feature_cols),
        "seq_len": seq_len,
        "test_pred_hours": pred_hours.tolist(),
        "test_true_hours": true_hours.tolist(),
    }
