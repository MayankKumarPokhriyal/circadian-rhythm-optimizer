"""Visualization helpers for circadian rhythm analysis."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)

sns.set_style("whitegrid")


def plot_phase_alignment(
    timestamps: Iterable[pd.Timestamp],
    observed_phase: np.ndarray,
    ideal_phase: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot observed vs ideal circadian phase across time."""

    ax = ax or plt.gca()
    hours = np.array([(ts.hour + ts.minute / 60) for ts in timestamps])
    ax.plot(hours, np.rad2deg(observed_phase) % 360, label="Observed phase", marker="o")
    ax.plot(hours, np.rad2deg(ideal_phase) % 360, label="Ideal solar phase", linestyle="--")
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Phase angle (degrees)")
    ax.set_title("Circadian phase alignment")
    ax.legend()
    return ax


def scatter_light_vs_sleep_offset(
    light_exposure_minutes: np.ndarray,
    sleep_offset_minutes: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Visualize relationship between light exposure and sleep offset."""

    ax = ax or plt.gca()
    sns.regplot(x=light_exposure_minutes, y=sleep_offset_minutes, ax=ax)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Morning light exposure (min)")
    ax.set_ylabel("Sleep midpoint offset (min)")
    ax.set_title("Light exposure vs sleep alignment")
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[str],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot confusion matrix for chronotype classification."""

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    ax = ax or plt.gca()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Chronotype classification confusion matrix")
    return ax


def plot_weekly_plan(plan_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot weekly realignment plan using horizontal bars."""

    ax = ax or plt.gca()
    for idx, row in plan_df.iterrows():
        day_label = row["day"].strftime("%a")
        ax.barh(idx, (row["sleep_end"] - row["sleep_start"]).total_seconds() / 3600, left=row["sleep_start"].hour + row["sleep_start"].minute / 60, color="navy", alpha=0.6, label="Sleep" if idx == 0 else "")
        ax.barh(idx, (row["light_exposure_end"] - row["light_exposure_start"]).total_seconds() / 3600, left=row["light_exposure_start"].hour + row["light_exposure_start"].minute / 60, color="gold", alpha=0.7, label="Light" if idx == 0 else "")
        ax.text(24.5, idx, f"Δφ: {row['phase_shift_minutes']:.0f} min", va="center")
        ax.text(26.5, idx, " | ".join(row.get("notes", [])), va="center", fontsize=8)
    ax.set_yticks(range(len(plan_df)))
    ax.set_yticklabels(plan_df["day"].dt.strftime("%a %d %b"))
    ax.set_xlabel("Clock hour")
    ax.set_xlim(0, 30)
    ax.set_title("Weekly circadian realignment plan")
    ax.legend(loc="upper right")
    return ax
