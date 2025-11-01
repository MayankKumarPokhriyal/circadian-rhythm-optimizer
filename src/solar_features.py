"""Solar feature engineering leveraging pvlib to align with solar time."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import import_module
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SolarContext:
    """Metadata describing the location and time window for solar calculations."""

    latitude: float
    longitude: float
    elevation: float = 0.0
    timezone: Optional[str] = "Europe/Rome"


def compute_solar_events(
    context: SolarContext,
    dates: Iterable[datetime],
) -> pd.DataFrame:
    """Compute sunrise, sunset, and solar noon for each provided date."""

    pvlib = _get_pvlib()
    times = pd.DatetimeIndex(sorted({dt.replace(hour=0, minute=0, second=0, microsecond=0) for dt in dates}))
    observer = pvlib.location.Location(
        latitude=context.latitude,
        longitude=context.longitude,
        altitude=context.elevation,
        tz=context.timezone,
    )
    solpos = observer.get_solarposition(times)
    sun_rise_set = observer.get_sun_rise_set_transit(times)

    events = pd.DataFrame(
        {
            "date": times.tz_convert(context.timezone) if context.timezone else times,
            "sunrise": sun_rise_set["sunrise"],
            "sunset": sun_rise_set["sunset"],
            "solar_noon": sun_rise_set["transit"],
            "solar_elevation_noon": solpos.loc[:, "elevation"],
        }
    ).reset_index(drop=True)
    events["solar_midnight"] = events["solar_noon"] - timedelta(hours=12)
    events["day_length_hours"] = (
        (events["sunset"] - events["sunrise"]).dt.total_seconds() / 3600.0
    )
    return events


def _get_pvlib():
    """Lazy import pvlib to keep the dependency optional at import time."""

    try:
        return import_module("pvlib")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard for optional dependency
        raise ImportError(
            "pvlib is required for solar feature computation. Install it via "
            "'pip install pvlib'."
        ) from exc


def ideal_solar_schedule(events: pd.DataFrame) -> pd.DataFrame:
    """Generate ideal sleep window anchored to solar events."""

    schedule = events.copy()
    schedule["ideal_sleep_start"] = schedule["solar_midnight"] - timedelta(hours=4)
    schedule["ideal_sleep_end"] = schedule["solar_midnight"] + timedelta(hours=4)
    schedule["ideal_wake_time"] = schedule["ideal_sleep_end"]
    schedule["ideal_light_exposure_start"] = schedule["sunrise"] - timedelta(minutes=30)
    schedule["ideal_light_exposure_end"] = schedule["sunrise"] + timedelta(hours=2)
    return schedule


def attach_solar_features(
    df: pd.DataFrame,
    events: pd.DataFrame,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Align solar events with the main dataframe via nearest joins."""

    enriched = df.copy()
    if datetime_col not in enriched.columns:
        raise KeyError(f"Dataframe lacks '{datetime_col}' column for alignment.")

    events_sorted = events.sort_values("date")
    enriched = pd.merge_asof(
        enriched.sort_values(datetime_col),
        events_sorted,
        left_on=datetime_col,
        right_on="date",
        direction="nearest",
    )
    return enriched


def circadian_phase_difference(
    observed_phase_radians: np.ndarray,
    ideal_phase_radians: np.ndarray,
) -> np.ndarray:
    """Compute wrapped phase difference between observed and ideal rhythms."""

    delta = observed_phase_radians - ideal_phase_radians
    return np.arctan2(np.sin(delta), np.cos(delta))


def radians_to_clock_hours(radians: np.ndarray) -> np.ndarray:
    """Convert phase angles in radians to clock hours (0-24)."""

    hours = (radians % (2 * np.pi)) / (2 * np.pi) * 24.0
    return hours


def clock_hours_to_radians(hours: np.ndarray) -> np.ndarray:
    """Inverse conversion from hours (0-24) to phase radians."""

    radians = (hours % 24.0) / 24.0 * 2 * np.pi
    return radians


def solar_phase_reference(events: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    """Construct mapping from date to ideal phase angle (solar midnight)."""

    reference: Dict[pd.Timestamp, float] = {}
    for _, row in events.iterrows():
        midnight = row["solar_midnight"]
        if pd.isna(midnight):
            continue
        reference[pd.Timestamp(row["date"]).normalize()] = 0.0  # solar midnight as phase 0
    return reference
