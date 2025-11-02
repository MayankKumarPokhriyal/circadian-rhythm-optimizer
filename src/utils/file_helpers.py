"""Utility helpers for file management, safe I/O, and progress tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    """Create a directory and parents if missing."""

    path.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV with logging; returns empty frame if not found."""

    if not path.exists():
        LOGGER.warning("CSV missing: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        LOGGER.exception("Failed to read CSV: %s", path)
        raise


def write_csv_safe(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write CSV with logging, ensuring parent directory exists."""

    ensure_dir(path.parent)
    try:
        df.to_csv(path, index=False, **kwargs)
        LOGGER.info("Wrote CSV: %s (%d rows, %d cols)", path, len(df), df.shape[1])
    except Exception:
        LOGGER.exception("Failed to write CSV: %s", path)
        raise


def progress(iterable: Iterable, desc: str | None = None) -> Iterator:
    """Simple tqdm wrapper that plays nice in notebooks."""

    return tqdm(iterable, desc=desc, leave=False)


def find_user_dirs(raw_root: Path) -> list[Path]:
    """Return a list of candidate user directories under raw data.

    This searches recursively for folders matching common patterns (e.g., 'user_1').
    """

    candidates: list[Path] = []
    for p in raw_root.rglob("user_*"):
        if p.is_dir():
            candidates.append(p)
    return sorted(candidates)


def find_case_insensitive_file(folder: Path, filename: str) -> Optional[Path]:
    """Find a file by name in a folder, case-insensitively."""

    lower = filename.lower()
    for child in folder.iterdir():
        if child.is_file() and child.name.lower() == lower:
            return child
    return None
