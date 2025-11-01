"""Rule-based recommendation engine for circadian realignment plans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class UserContext:
    """Contextual information used to personalise recommendations."""

    participant_id: str
    latitude: float
    longitude: float
    chronotype: Optional[str] = None
    work_start_time: Optional[datetime] = None
    sleep_duration_target_hours: float = 8.0


@dataclass(slots=True)
class PhaseAlignmentPlan:
    """Container for daily recommendations."""

    day: datetime
    phase_shift_minutes: float
    sleep_start: datetime
    sleep_end: datetime
    light_exposure_start: datetime
    light_exposure_end: datetime
    notes: List[str]


def compute_phase_shift_minutes(delta_phase_radians: np.ndarray) -> float:
    """Convert mean phase difference to minutes."""

    wrapped = np.arctan2(np.sin(delta_phase_radians), np.cos(delta_phase_radians))
    return float(np.degrees(wrapped.mean()) * 4)  # 360 degrees -> 24h -> 1440 min, 1 degree -> 4 min


def classify_shift_direction(phase_shift_minutes: float) -> str:
    if abs(phase_shift_minutes) < 15:
        return "aligned"
    return "advance" if phase_shift_minutes > 0 else "delay"


def recommend_sleep_schedule(
    ideal_sleep_start: datetime,
    ideal_sleep_end: datetime,
    phase_shift_minutes: float,
    context: UserContext,
) -> Dict[str, datetime]:
    """Adjust sleep schedule based on phase shift direction."""

    direction = classify_shift_direction(phase_shift_minutes)
    shift = timedelta(minutes=phase_shift_minutes)
    if direction == "aligned":
        return {"start": ideal_sleep_start, "end": ideal_sleep_end}

    adjusted_start = ideal_sleep_start + shift * 0.5
    adjusted_end = ideal_sleep_end + shift * 0.5

    if context.work_start_time is not None:
        latest_wake = context.work_start_time - timedelta(hours=1)
        if adjusted_end > latest_wake:
            delta = adjusted_end - latest_wake
            adjusted_start -= delta
            adjusted_end -= delta
    return {"start": adjusted_start, "end": adjusted_end}


def recommend_light_exposure(
    ideal_start: datetime,
    ideal_end: datetime,
    phase_shift_minutes: float,
) -> Dict[str, datetime]:
    """Adjust light therapy window opposite to sleep shift direction."""

    direction = classify_shift_direction(phase_shift_minutes)
    shift = timedelta(minutes=abs(phase_shift_minutes) * 0.5)
    if direction == "aligned":
        return {"start": ideal_start, "end": ideal_end}
    if direction == "advance":
        return {"start": ideal_start + shift, "end": ideal_end + shift}
    return {"start": ideal_start - shift, "end": ideal_end - shift}


def generate_notes(phase_shift_minutes: float, context: UserContext) -> List[str]:
    direction = classify_shift_direction(phase_shift_minutes)
    notes = []
    if direction == "aligned":
        notes.append("Maintain current schedule; reinforce with consistent daylight exposure.")
    elif direction == "advance":
        notes.append("Aim for morning bright light and avoid blue light after sunset.")
    else:
        notes.append("Increase evening light exposure; keep mornings dim until alignment improves.")

    if context.chronotype:
        notes.append(f"Chronotype assessment: {context.chronotype}.")
    notes.append(f"Average phase shift required: {phase_shift_minutes:.1f} minutes ({direction}).")
    return notes


def create_weekly_realignment_plan(
    solar_schedule: Dict[str, datetime],
    phase_shift_minutes: float,
    context: UserContext,
) -> List[PhaseAlignmentPlan]:
    """Generate a 7-day progressive plan to realign circadian rhythm."""

    plans: List[PhaseAlignmentPlan] = []
    shift_per_day = phase_shift_minutes / 7
    for day_offset in range(7):
        day = solar_schedule["date"] + timedelta(days=day_offset)
        cumulative_shift = shift_per_day * (day_offset + 1)
        sleep = recommend_sleep_schedule(
            solar_schedule["ideal_sleep_start"],
            solar_schedule["ideal_sleep_end"],
            cumulative_shift,
            context,
        )
        light = recommend_light_exposure(
            solar_schedule["ideal_light_exposure_start"],
            solar_schedule["ideal_light_exposure_end"],
            cumulative_shift,
        )
        plans.append(
            PhaseAlignmentPlan(
                day=day,
                phase_shift_minutes=cumulative_shift,
                sleep_start=sleep["start"],
                sleep_end=sleep["end"],
                light_exposure_start=light["start"],
                light_exposure_end=light["end"],
                notes=generate_notes(cumulative_shift, context),
            )
        )
    return plans
