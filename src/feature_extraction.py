"""Feature extraction utilities for circadian rhythm modeling.

This module derives interpretable physiological and behavioural features from
processed MMASH dataframes. Functions are designed for use both in notebooks
and scripted pipelines, and they favour transparency for downstream reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

from .utils.file_helpers import ensure_dir, write_csv_safe

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FeatureEngineeringConfig:
    """Configuration for temporal windowing and feature scaling."""

    window_minutes: int = 30
    step_minutes: int = 5
    hrv_frequency_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "ulf": (0.0, 0.003),
            "vlf": (0.003, 0.04),
            "lf": (0.04, 0.15),
            "hf": (0.15, 0.4),
        }
    )
    feature_columns: Optional[Sequence[str]] = None


@dataclass(slots=True)
class FeatureSet:
    """Container for features and scaler used during normalization."""

    features: pd.DataFrame
    targets: Optional[pd.DataFrame] = None
    scaler: Optional[StandardScaler] = None


def compute_hrv(rr_series: pd.Series) -> Dict[str, float]:
    """Compute time-domain HRV metrics: RMSSD and SDNN.

    Parameters
    ----------
    rr_series : pd.Series
        RR intervals in milliseconds.

    Returns
    -------
    Dict[str, float]
        Keys: rmssd, sdnn
    """

    s = pd.to_numeric(rr_series, errors="coerce").dropna()
    if len(s) < 2:
        return {"rmssd": float("nan"), "sdnn": float("nan")}
    diffs = s.diff().dropna()
    rmssd = float(np.sqrt(np.mean(diffs.values ** 2)))
    sdnn = float(np.std(s.values, ddof=1))
    return {"rmssd": rmssd, "sdnn": sdnn}


def compute_time_derived_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """Add circadian-friendly time-of-day encodings."""

    enriched = df.copy()
    if datetime_col not in enriched.columns:
        raise KeyError(f"Expected datetime column '{datetime_col}' not found.")
    time_index = pd.to_datetime(enriched[datetime_col], utc=True)
    seconds = time_index.dt.hour * 3600 + time_index.dt.minute * 60 + time_index.dt.second
    radians = 2 * np.pi * (seconds / 86400)
    enriched["time_sin"] = np.sin(radians)
    enriched["time_cos"] = np.cos(radians)
    enriched["day_of_week"] = time_index.dt.dayofweek
    return enriched


def compute_sleep_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling sleep metrics using the binary ``is_sleep`` flag."""

    enriched = df.copy()
    if "is_sleep" not in enriched.columns:
        LOGGER.warning("Dataframe lacks 'is_sleep'; sleep metrics will be zero.")
        enriched["is_sleep"] = False

    rolling_window = 12  # Assuming 5-minute cadence -> 1 hour
    enriched["sleep_fraction_1h"] = (
        enriched["is_sleep"].astype(int).rolling(rolling_window, min_periods=1).mean()
    )
    enriched["sleep_change"] = enriched["sleep_fraction_1h"].diff().fillna(0)
    return enriched


def compute_rrv_features(df: pd.DataFrame, rr_column: str = "rr_interval_ms") -> pd.DataFrame:
    """Compute HRV features from RR interval data."""

    enriched = df.copy()
    if rr_column not in enriched.columns:
        LOGGER.warning("RR interval column '%s' missing; HRV features will be NaN.", rr_column)
        enriched["rmssd"] = np.nan
        enriched["sdnn"] = np.nan
        return enriched

    rr = enriched[rr_column].astype(float)
    enriched["rmssd"] = np.sqrt(
        pd.Series(rr).diff().pipe(lambda s: (s**2).rolling(window=5, min_periods=1).mean())
    )
    enriched["sdnn"] = rr.rolling(window=12, min_periods=1).std()

    # Frequency-domain HRV using Welch's method
    fs = 4.0  # 4 Hz resampling after interpolation
    rr_seconds = rr / 1000.0
    if len(rr_seconds.dropna()) >= 32:
        fxx, pxx = welch(rr_seconds.dropna(), fs=fs, nperseg=min(256, len(rr_seconds.dropna())))
        for band, (low, high) in FeatureEngineeringConfig().hrv_frequency_bands.items():
            mask = (fxx >= low) & (fxx < high)
            enriched[f"hrv_{band}"] = np.trapz(pxx[mask], fxx[mask]) if np.any(mask) else np.nan
    else:
        for band in FeatureEngineeringConfig().hrv_frequency_bands:
            enriched[f"hrv_{band}"] = np.nan
    return enriched


def compute_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive vector magnitude and simple stats from accelerometry data."""

    enriched = df.copy()
    axes = [col for col in df.columns if col.startswith("acc_")]
    if len(axes) >= 3:
        enriched["acc_vm"] = np.sqrt((enriched[axes] ** 2).sum(axis=1))
    if "heart_rate" in enriched.columns:
        enriched["hr_zscore"] = (
            enriched["heart_rate"].rolling(window=60, min_periods=10).apply(zscore_last, raw=False)
        )
    return enriched


def zscore_last(series: pd.Series) -> float:
    """Return the z-score of the last element within a rolling window."""

    values = series.dropna()
    if len(values) < 2:
        return 0.0
    return (values.iloc[-1] - values.mean()) / values.std(ddof=0)


def make_sliding_windows(
    df: pd.DataFrame,
    config: FeatureEngineeringConfig,
    target_columns: Optional[Sequence[str]] = None,
    datetime_col: str = "datetime",
) -> FeatureSet:
    """Create sliding-window features suitable for sequence models."""

    if datetime_col not in df.columns:
        raise KeyError(f"Dataframe must include '{datetime_col}' for windowing.")

    minutes = (df[datetime_col].diff().dt.total_seconds() / 60.0).fillna(config.step_minutes)
    cadence = minutes.median()
    window_size = int(np.ceil(config.window_minutes / cadence))
    step_size = int(np.ceil(config.step_minutes / cadence))

    feature_cols = (
        list(config.feature_columns)
        if config.feature_columns is not None
        else [col for col in df.columns if col not in {datetime_col}]
    )
    matrix: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        matrix.append(window[feature_cols].to_numpy())
        if target_columns is not None:
            targets.append(window[target_columns].iloc[-1].to_numpy())

    feature_array = np.stack(matrix) if matrix else np.empty((0, window_size, len(feature_cols)))
    target_df = None
    if targets:
        target_array = np.stack(targets)
        target_df = pd.DataFrame(target_array, columns=target_columns)

    sequences = pd.DataFrame(
        {
            "sequence": list(feature_array),
            "start_time": df.loc[::step_size, datetime_col].iloc[: feature_array.shape[0]].to_list(),
        }
    )

    scaler: Optional[StandardScaler] = None
    if feature_array.size > 0:
        scaler = StandardScaler()
        flattened = feature_array.reshape(feature_array.shape[0], -1)
        scaler.fit(flattened)

    return FeatureSet(features=sequences, targets=target_df, scaler=scaler)


def normalize_features(
    dataframe: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Z-score normalize selected columns using an existing or new scaler."""

    normalized = dataframe.copy()
    cols = list(columns) if columns else normalized.select_dtypes(include=["number"]).columns.tolist()
    if not cols:
        raise ValueError("No numeric columns available for normalization.")

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(normalized[cols])

    normalized[cols] = scaler.transform(normalized[cols])
    return normalized, scaler


def attach_targets(
    df: pd.DataFrame,
    circadian_phase_column: str = "circadian_phase_radians",
) -> pd.DataFrame:
    """Append sin/cos encodings of circadian phase suitable for regression."""

    enriched = df.copy()
    if circadian_phase_column not in enriched.columns:
        raise KeyError(
            f"Dataframe lacks required circadian phase column '{circadian_phase_column}'."
        )
    radians = enriched[circadian_phase_column].astype(float)
    enriched["phase_sin"] = np.sin(radians)
    enriched["phase_cos"] = np.cos(radians)
    return enriched


def extract_circadian_features(
    all_users_path: str | Path,
    output_path: str | Path = "data/features/circadian_features.csv",
    log: bool = True,
) -> Path:
    """Generate ML-ready circadian features per user and persist to CSV.

    Steps per user_id:
    - Rolling stats over 30 min to 6 h for heart rate and activity VM
    - Time-domain HRV metrics from RR intervals (if present)
    - Sleep quality proxies from is_sleep flag (TST, WASO, efficiency via rolling)
    - Merge daily static values from questionnaire/saliva
    - Per-user z-score normalization of numeric columns
    """

    all_users_path = Path(all_users_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    df = pd.read_csv(all_users_path, parse_dates=["datetime"])
    if "user_id" not in df.columns:
        raise KeyError("Expected 'user_id' column in merged dataset.")

    # Pre-compute activity VM if possible
    axes = [c for c in df.columns if c.startswith("acc_")]
    if axes:
        df["acc_vm"] = np.sqrt((df[axes] ** 2).sum(axis=1))

    # Prepare containers
    frames: List[pd.DataFrame] = []
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values("datetime").copy()
        g.set_index("datetime", inplace=True)

        # Rolling windows
        windows = {"30min": "30min", "1h": "60min", "2h": "120min", "6h": "360min"}

        # Heart rate metrics
        if "heart_rate" in g.columns:
            for name, rule in windows.items():
                g[f"hr_mean_{name}"] = g["heart_rate"].rolling(rule, min_periods=5).mean()
                g[f"hr_var_{name}"] = g["heart_rate"].rolling(rule, min_periods=5).var()

        # RR intervals (time-domain HRV)
        rr_col = None
        for c in g.columns:
            cl = c.lower()
            if "rr" in cl and "ms" in cl:
                rr_col = c
                break
        if rr_col:
            # Compute HRV in rolling windows using apply with stride windows
            for name, rule in windows.items():
                g[f"rmssd_{name}"] = g[rr_col].rolling(rule, min_periods=10).apply(
                    lambda s: compute_hrv(pd.Series(s)).get("rmssd", np.nan), raw=False
                )
                g[f"sdnn_{name}"] = g[rr_col].rolling(rule, min_periods=10).apply(
                    lambda s: compute_hrv(pd.Series(s)).get("sdnn", np.nan), raw=False
                )

        # Activity features
        if "acc_vm" in g.columns:
            for name, rule in windows.items():
                g[f"vm_mean_{name}"] = g["acc_vm"].rolling(rule, min_periods=5).mean()
                g[f"vm_std_{name}"] = g["acc_vm"].rolling(rule, min_periods=5).std()

        # Sleep features (if is_sleep present): TST, WASO, efficiency via rolling 6h
        if "is_sleep" in g.columns:
            asleep = g["is_sleep"].astype(float)
            # Total Sleep Time proportion in window
            g["tst_prop_6h"] = asleep.rolling("360min", min_periods=5).mean()
            # Sleep transitions as proxy for fragmentation / WASO
            transitions = asleep.diff().abs().fillna(0.0)
            g["waso_events_6h"] = transitions.rolling("360min", min_periods=5).sum()
            # Efficiency proxy: sleep fraction variability
            g["sleep_eff_var_6h"] = (
                asleep.rolling("360min", min_periods=5).var()
            )

        # Static questionnaire/saliva: forward fill per day
        static_cols = [c for c in g.columns if c.startswith("questionnaire_") or c.startswith("melatonin") or c.startswith("cortisol") or c.startswith("user_")]
        if static_cols:
            g[static_cols] = g[static_cols].ffill()

        # Per-user normalization (z-score) on numeric columns
        g.reset_index(inplace=True)
        numeric_cols = g.select_dtypes(include=["number"]).columns
        # Do not normalize binary flags
        skip_cols = {"is_sleep"}
        norm_cols = [c for c in numeric_cols if c not in skip_cols]
        g[norm_cols] = (g[norm_cols] - g[norm_cols].mean()) / g[norm_cols].std(ddof=0)

        # Log missing ratio and feature count
        if log:
            missing_ratio = float(g.isna().mean().mean())
            LOGGER.info(
                "User %s | features=%d | missing_ratio=%.3f",
                uid,
                g.shape[1],
                missing_ratio,
            )
        frames.append(g)

    features = pd.concat(frames, axis=0, ignore_index=True)
    write_csv_safe(features, output_path)
    return output_path
