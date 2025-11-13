"""Utility helpers shared across pipeline stages."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class Keypoint:
    x: float
    y: float


def compute_angle(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """Return the angle ABC in degrees."""
    ab = np.array([a.x - b.x, a.y - b.y])
    cb = np.array([c.x - b.x, c.y - b.y])
    ab_norm = np.linalg.norm(ab)
    cb_norm = np.linalg.norm(cb)
    if ab_norm == 0 or cb_norm == 0:
        return 0.0
    cos_angle = float(np.dot(ab, cb) / (ab_norm * cb_norm))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def vector_magnitude(dx: float, dy: float) -> float:
    return float(math.sqrt(dx * dx + dy * dy))


def chunk_events(df: pd.DataFrame, frames_per_event: int) -> pd.DataFrame:
    """Assign an event id to each frame based on order."""
    df = df.copy()
    df["event_id"] = df["frame_idx"] // frames_per_event
    return df


def rolling_speed(points: Iterable[Tuple[float, float]]) -> float:
    """Approximate speed from a sequence of (x, y) points."""
    points = list(points)
    if len(points) < 2:
        return 0.0
    distances = [
        vector_magnitude(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    ]
    return float(np.mean(distances))
