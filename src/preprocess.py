"""Utilities for loading, cleaning, and aligning multimodal MMASH data.

This module consolidates the various modality files that compose the MMASH
(Multi-Modal Mental and Physical Health) dataset and prepares a unified
chronological dataframe ready for feature extraction and modeling.

All functions are designed to be notebook-friendly while remaining fully
scriptable for reproducible pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .utils.file_helpers import (
    ensure_dir,
    find_case_insensitive_file,
    find_user_dirs,
    progress,
    write_csv_safe,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MMASHLoaderConfig:
    """Configuration describing the MMASH dataset layout.

    Attributes
    ----------
    root : Path
        Base directory containing the raw MMASH CSV files.
    timezone : str | None
        Optional timezone string (e.g. "Europe/Rome") for localized timestamps.
    file_map : Mapping[str, str]
        Mapping between modality keys and CSV filenames.
    participant_id_column : str
        Name of the column that identifies the participant across modalities.
    timestamp_columns : Mapping[str, str | List[str]]
        Mapping of modality to the timestamp column(s) to parse.
    """

    root: Path
    timezone: Optional[str] = "Europe/Rome"
    file_map: Mapping[str, str] = field(
        default_factory=lambda: {
            "sleep": "sleep.csv",
            "actigraphy": "actigraph.csv",
            "rr": "RR.csv",
            "questionnaire": "questionnaire.csv",
            "saliva": "saliva.csv",
        }
    )
    participant_id_column: str = "subject"
    timestamp_columns: Mapping[str, str | List[str]] = field(
        default_factory=lambda: {
            "sleep": "start_time",
            "actigraphy": "timestamp",
            "rr": "timestamp",
            "saliva": "collection_time",
        }
    )


def configure_logging(log_level: int = logging.INFO) -> None:
    """Configure a default logging handler for the package.

    Parameters
    ----------
    log_level : int, optional
        Logging level passed to :func:`logging.basicConfig`.
    """

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    LOGGER.debug("Logging configured with level %s", logging.getLevelName(log_level))


def log_dataframe_stats(name: str, df: pd.DataFrame) -> None:
    """Log basic dataframe stats for quick inspection."""

    if df is None or df.empty:
        LOGGER.info("%s: EMPTY", name)
        return
    missing = int(df.isna().sum().sum())
    LOGGER.info(
        "%s: %d rows x %d cols | missing=%d",
        name,
        len(df),
        df.shape[1],
        missing,
    )


def _read_csv(path: Path, parse_dates: Optional[Iterable[str]]) -> pd.DataFrame:
    """Read a CSV file with optional datetime parsing and logging."""

    LOGGER.info("Loading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")

    dataframe = pd.read_csv(path)
    if parse_dates:
        for column in parse_dates:
            if column in dataframe.columns:
                dataframe[column] = pd.to_datetime(
                    dataframe[column], errors="coerce", utc=True
                )
            else:
                LOGGER.warning(
                    "Column %s missing in %s; downstream alignment may fail.",
                    column,
                    path.name,
                )
    return dataframe


def _parse_timestamp_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce columns to pandas datetime (UTC) if present."""

    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
    return out


def load_modalities(config: MMASHLoaderConfig) -> Dict[str, pd.DataFrame]:
    """Load all requested modalities into memory."""

    loaded: Dict[str, pd.DataFrame] = {}
    for modality, filename in config.file_map.items():
        csv_path = config.root / filename
        parse_cols = config.timestamp_columns.get(modality)
        if parse_cols is None:
            parse_list: Optional[List[str]] = None
        elif isinstance(parse_cols, str):
            parse_list = [parse_cols]
        else:
            parse_list = list(parse_cols)

        loaded[modality] = _read_csv(csv_path, parse_list)
    return loaded


def clean_sleep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic cleaning on sleep metadata."""

    required_columns = {"start_time", "end_time", "duration", "sleep_quality"}
    missing = required_columns - set(df.columns)
    if missing:
        LOGGER.warning("Sleep dataframe missing columns: %s", sorted(missing))

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    if "duration" in cleaned.columns:
        cleaned["duration"] = cleaned["duration"].clip(lower=0)
    if "sleep_efficiency" in cleaned.columns:
        cleaned["sleep_efficiency"].fillna(cleaned["sleep_efficiency"].median(), inplace=True)
    return cleaned


def clean_actigraphy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and resample actigraphy data."""

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=[col for col in df.columns if "time" in col.lower()])
    for axis in ("x", "y", "z"):
        col = f"acc_{axis}"
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(float)
    if "heart_rate" in cleaned.columns:
        cleaned["heart_rate"] = cleaned["heart_rate"].astype(float).clip(lower=20, upper=220)
    return cleaned


def clean_rr_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean RR interval data by removing artifacts."""

    cleaned = df.copy()
    if "rr_interval_ms" in cleaned.columns:
        rr = cleaned["rr_interval_ms"].astype(float)
        mask = rr.between(300, 2000)
        cleaned = cleaned.loc[mask].copy()
        cleaned["rr_interval_ms"] = cleaned["rr_interval_ms"].interpolate(limit_direction="both")
    return cleaned


def clean_saliva_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean saliva hormone samples by removing negatives and standardising units."""

    cleaned = df.copy()
    hormone_columns = [col for col in cleaned.columns if "melatonin" in col or "cortisol" in col]
    for col in hormone_columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].clip(lower=0)
    return cleaned


def clean_questionnaire_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean questionnaire data by filling demographics and psychometrics."""

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=[col for col in cleaned.columns if "id" in col.lower()])
    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median())
    return cleaned


def harmonize_timezones(
    modalities: Mapping[str, pd.DataFrame],
    config: MMASHLoaderConfig,
) -> Dict[str, pd.DataFrame]:
    """Ensure all timestamp columns share the same timezone."""

    timezone = config.timezone
    harmonized: Dict[str, pd.DataFrame] = {}
    for modality, frame in modalities.items():
        frame_copy = frame.copy()
        parse_cols = config.timestamp_columns.get(modality)
        columns: List[str]
        if parse_cols is None:
            harmonized[modality] = frame_copy
            continue
        if isinstance(parse_cols, str):
            columns = [parse_cols]
        else:
            columns = list(parse_cols)
        for col in columns:
            if col not in frame_copy.columns:
                LOGGER.debug("Skipping missing timestamp column %s for %s", col, modality)
                continue
            if timezone:
                frame_copy[col] = frame_copy[col].dt.tz_convert(timezone)
            else:
                frame_copy[col] = frame_copy[col].dt.tz_localize(None)
        harmonized[modality] = frame_copy
    return harmonized


def resample_actigraphy(
    actigraphy: pd.DataFrame,
    timestamp_col: str = "timestamp",
    rule: str = "1min",
) -> pd.DataFrame:
    """Resample actigraphy at a fixed cadence for downstream modeling."""

    if timestamp_col not in actigraphy.columns:
        raise KeyError(f"Actigraphy data lacks timestamp column '{timestamp_col}'")

    indexed = actigraphy.set_index(timestamp_col)
    numeric_cols = indexed.select_dtypes(include=["number"]).columns
    resampled = indexed[numeric_cols].resample(rule).mean().interpolate()
    resampled.reset_index(inplace=True)
    return resampled


def merge_modalities(
    modalities: Mapping[str, pd.DataFrame],
    config: MMASHLoaderConfig,
    resample_rule: str = "1min",
) -> pd.DataFrame:
    """Merge modalities into a single time-aligned dataframe."""

    sleep = clean_sleep_dataframe(modalities.get("sleep", pd.DataFrame()))
    actigraphy_raw = clean_actigraphy_dataframe(modalities.get("actigraphy", pd.DataFrame()))
    rr = clean_rr_dataframe(modalities.get("rr", pd.DataFrame()))
    saliva = clean_saliva_dataframe(modalities.get("saliva", pd.DataFrame()))
    questionnaire = clean_questionnaire_dataframe(modalities.get("questionnaire", pd.DataFrame()))

    if actigraphy_raw.empty:
        raise ValueError("Actigraphy data is required to create the master timeline.")

    actigraphy = resample_actigraphy(actigraphy_raw, rule=resample_rule)

    merged = actigraphy.copy()
    merged.rename(columns={"timestamp": "datetime"}, inplace=True)

    if not rr.empty and "timestamp" in rr.columns:
        merged = pd.merge_asof(
            merged.sort_values("datetime"),
            rr.sort_values("timestamp"),
            left_on="datetime",
            right_on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(resample_rule),
        )
        merged.drop(columns=["timestamp"], inplace=True)

    if not saliva.empty and "collection_time" in saliva.columns:
        saliva_sorted = saliva.sort_values("collection_time")
        merged = pd.merge_asof(
            merged.sort_values("datetime"),
            saliva_sorted,
            left_on="datetime",
            right_on="collection_time",
            direction="nearest",
            tolerance=pd.Timedelta("1H"),
        )
        merged.drop(columns=["collection_time"], inplace=True)

    if not sleep.empty:
        merged = add_sleep_stage_annotations(merged, sleep)

    if not questionnaire.empty:
        merged = merged.assign(**{
            f"questionnaire_{col}": questionnaire[col].iloc[0]
            for col in questionnaire.columns
            if col != config.participant_id_column and not questionnaire.empty
        })

    merged.sort_values("datetime", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def add_sleep_stage_annotations(
    continuous_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate the continuous timeline with binary sleep flags."""

    annotated = continuous_df.copy()
    annotated["is_sleep"] = False
    if sleep_df.empty:
        return annotated

    if {"start_time", "end_time"}.issubset(sleep_df.columns):
        for _, row in sleep_df.iterrows():
            start = row["start_time"]
            end = row["end_time"]
            if pd.isna(start) or pd.isna(end):
                continue
            mask = (annotated["datetime"] >= start) & (annotated["datetime"] <= end)
            annotated.loc[mask, "is_sleep"] = True
    return annotated


def save_processed_dataset(
    dataframe: pd.DataFrame,
    output_path: Path,
    compression: Optional[str] = "gzip",
) -> Path:
    """Persist the processed dataframe to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving processed dataset to %s", output_path)
    dataframe.to_csv(output_path, index=False, compression=compression)
    return output_path


def process_mmash_dataset(
    config: MMASHLoaderConfig,
    resample_rule: str = "1min",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """End-to-end helper that loads, cleans, merges, and optionally saves data."""

    configure_logging()
    modalities = load_modalities(config)
    harmonized = harmonize_timezones(modalities, config)
    merged = merge_modalities(harmonized, config, resample_rule=resample_rule)
    if output_path:
        save_processed_dataset(merged, output_path)
    return merged


def load_processed_dataframe(path: Path) -> pd.DataFrame:
    """Convenience loader for already processed CSV files."""

    LOGGER.info("Loading processed dataframe from %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pd.read_csv(path, parse_dates=["datetime"])


# ------------------------
# User-level processing API
# ------------------------

USER_EXPECTED_FILES = {
    "sleep": "sleep.csv",
    "rr": "RR.csv",
    "actigraph": "Actigraph.csv",
    "activity": "Activity.csv",
    "questionnaire": "questionnaire.csv",
    "saliva": "saliva.csv",
    "user_info": "user_info.csv",
}


def inspect_user_data(user_id: str, frames: Mapping[str, pd.DataFrame]) -> None:
    """Print/log basic info for each modality for a user."""

    LOGGER.info("Inspecting user %s", user_id)
    for key, df in frames.items():
        log_dataframe_stats(f"{user_id}:{key}", df)


def _load_user_modalities(user_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all expected modality CSVs for a user directory (case-insensitive)."""

    loaded: Dict[str, pd.DataFrame] = {}
    for key, fname in USER_EXPECTED_FILES.items():
        path = find_case_insensitive_file(user_dir, fname)
        if path is None:
            LOGGER.warning("%s missing in %s", fname, user_dir)
            loaded[key] = pd.DataFrame()
            continue
        df = pd.read_csv(path)
        # Attempt to parse likely timestamp columns by name patterns
        ts_cols = [c for c in df.columns if "time" in c.lower() or c.lower() in {"timestamp", "datetime"}]
        df = _parse_timestamp_columns(df, ts_cols)
        loaded[key] = df
    return loaded


def _align_and_merge_user(
    user_id: str,
    frames: Mapping[str, pd.DataFrame],
    timezone: Optional[str] = "Europe/Rome",
) -> pd.DataFrame:
    """Unify timestamps (UTC), resample to 1-min, align via merge_asof, and clean missing values."""

    # Actigraph data as master timeline
    acti = frames.get("actigraph", pd.DataFrame()).copy()
    if acti.empty:
        # Fallback to Activity.csv if Actigraph.csv missing
        acti = frames.get("activity", pd.DataFrame()).copy()
    if acti.empty:
        raise ValueError(f"User {user_id}: No actigraphy/activity file found.")

    # Normalise timestamp column name
    ts_col = next((c for c in acti.columns if c.lower() in {"timestamp", "datetime"} or "time" in c.lower()), None)
    if ts_col is None:
        raise KeyError(f"User {user_id}: Could not locate timestamp column in actigraphy file.")
    acti = acti.rename(columns={ts_col: "timestamp"})
    if timezone:
        acti["timestamp"] = pd.to_datetime(acti["timestamp"], utc=True).dt.tz_convert(timezone)
    acti = clean_actigraphy_dataframe(acti)
    acti = resample_actigraphy(acti, timestamp_col="timestamp", rule="1min")
    acti.rename(columns={"timestamp": "datetime"}, inplace=True)

    merged = acti.copy()

    # Merge RR intervals
    rr = frames.get("rr", pd.DataFrame()).copy()
    if not rr.empty:
        rr_ts = next((c for c in rr.columns if c.lower() in {"timestamp", "datetime"} or "time" in c.lower()), None)
        if rr_ts:
            rr = rr.rename(columns={rr_ts: "timestamp"})
            if timezone:
                rr["timestamp"] = pd.to_datetime(rr["timestamp"], utc=True).dt.tz_convert(timezone)
            rr.sort_values("timestamp", inplace=True)
            merged = pd.merge_asof(
                merged.sort_values("datetime"),
                rr,
                left_on="datetime",
                right_on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("2min"),
            )
            merged.drop(columns=["timestamp"], inplace=True, errors="ignore")

    # Merge saliva (hormone samples)
    saliva = frames.get("saliva", pd.DataFrame()).copy()
    if not saliva.empty:
        sal_ts = next((c for c in saliva.columns if c.lower() in {"collection_time", "timestamp", "datetime"} or "time" in c.lower()), None)
        if sal_ts:
            saliva = saliva.rename(columns={sal_ts: "collection_time"})
            if timezone:
                saliva["collection_time"] = pd.to_datetime(saliva["collection_time"], utc=True).dt.tz_convert(timezone)
            saliva.sort_values("collection_time", inplace=True)
            merged = pd.merge_asof(
                merged.sort_values("datetime"),
                saliva,
                left_on="datetime",
                right_on="collection_time",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),
            )
            merged.drop(columns=["collection_time"], inplace=True, errors="ignore")

    # Sleep episodes -> annotate timeline
    sleep = frames.get("sleep", pd.DataFrame()).copy()
    if not sleep.empty:
        start_col = next((c for c in sleep.columns if "start" in c.lower() and "time" in c.lower()), None)
        end_col = next((c for c in sleep.columns if "end" in c.lower() and "time" in c.lower()), None)
        if start_col and end_col:
            sleep = sleep.rename(columns={start_col: "start_time", end_col: "end_time"})
            sleep["start_time"] = pd.to_datetime(sleep["start_time"], utc=True).dt.tz_convert(timezone)
            sleep["end_time"] = pd.to_datetime(sleep["end_time"], utc=True).dt.tz_convert(timezone)
            merged = add_sleep_stage_annotations(merged, sleep)

    # Questionnaire: drop rows with missing critical info and attach first row values
    q = frames.get("questionnaire", pd.DataFrame()).copy()
    if not q.empty:
        # Heuristic: drop rows missing >50% fields
        thresh = int(q.shape[1] * 0.5)
        q.dropna(thresh=thresh, inplace=True)
        if not q.empty:
            first = q.iloc[0]
            attach = {f"questionnaire_{c}": first[c] for c in q.columns}
            merged = merged.assign(**attach)

    # User info as static columns
    info = frames.get("user_info", pd.DataFrame()).copy()
    if not info.empty:
        first = info.iloc[0]
        attach = {f"user_{c}": first[c] for c in info.columns}
        merged = merged.assign(**attach)

    # Handle missing values: numeric interpolate, categorical ffill/mode
    numeric_cols = merged.select_dtypes(include=["number"]).columns
    merged[numeric_cols] = merged[numeric_cols].interpolate(limit_direction="both")
    categorical_cols = merged.select_dtypes(exclude=["number", "datetime64[ns, UTC]", "datetime64[ns]"]).columns
    merged[categorical_cols] = merged[categorical_cols].ffill()
    for c in categorical_cols:
        if merged[c].isna().any():
            mode = merged[c].mode(dropna=True)
            if not mode.empty:
                merged[c].fillna(mode.iloc[0], inplace=True)

    # Round datetime to minute for alignment
    merged["datetime"] = pd.to_datetime(merged["datetime"], utc=True).dt.floor("min")
    merged.sort_values("datetime", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged["user_id"] = user_id
    return merged


def process_all_users(
    raw_root: Path,
    processed_dir: Path,
    timezone: Optional[str] = "Europe/Rome",
    save_intermediate: bool = True,
) -> List[Path]:
    """Process all user folders and save user-level merged CSVs.

    Returns list of paths to processed user CSVs.
    """

    configure_logging()
    ensure_dir(processed_dir)
    outputs: List[Path] = []
    users = find_user_dirs(raw_root)
    LOGGER.info("Discovered %d user directories under %s", len(users), raw_root)
    for user_dir in progress(users, desc="Users"):
        user_id = user_dir.name.replace("user_", "").strip()
        frames = _load_user_modalities(user_dir)
        inspect_user_data(user_id, frames)
        merged = _align_and_merge_user(user_id, frames, timezone=timezone)
        out_path = processed_dir / f"User_{user_id}_merged.csv"
        if save_intermediate:
            write_csv_safe(merged, out_path)
        outputs.append(out_path)
    return outputs


def merge_all_users(processed_dir: Path, output_path: Optional[Path] = None) -> Path:
    """Combine user-level merged files into a single master CSV.

    Ensures consistent datetime type and presence of user_id.
    """

    ensure_dir(processed_dir)
    files = sorted(processed_dir.glob("User_*_merged.csv"))
    if not files:
        raise FileNotFoundError(f"No user-level files found in {processed_dir}")
    frames = []
    for fp in files:
        df = pd.read_csv(fp, parse_dates=["datetime"])
        if "user_id" not in df.columns:
            # Infer from filename
            uid = fp.stem.split("_")[1]
            df["user_id"] = uid
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("min")
        frames.append(df)
    master = pd.concat(frames, axis=0, ignore_index=True)
    master.sort_values(["user_id", "datetime"], inplace=True)
    out = output_path or (processed_dir / "all_users_merged.csv")
    write_csv_safe(master, out)
    return out
