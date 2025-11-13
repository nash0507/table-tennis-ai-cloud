"""Feature engineering utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import Keypoint, compute_angle, rolling_speed, vector_magnitude


@dataclass
class StrokeFeatureConfig:
    landing_grid_size: int = 3
    contact_speed_scale: float = 1000.0


CONFIG = StrokeFeatureConfig()


def determine_stroke_hand(row: pd.Series) -> str:
    """Heuristic to detect forehand/backhand based on wrist and shoulder."""
    if row["wrist_x"] >= row["shoulder_x"]:
        return "forehand"
    return "backhand"


def assign_landing_zone(x: float, y: float) -> Tuple[int, int]:
    """Return the discrete landing zone indices within the 3x3 grid."""
    grid = CONFIG.landing_grid_size
    ix = int(np.clip(np.floor(x * grid), 0, grid - 1))
    iy = int(np.clip(np.floor(y * grid), 0, grid - 1))
    return ix, iy


def _aggregate_event(event: pd.DataFrame) -> Dict[str, float | int | str]:
    first = event.iloc[0]
    last = event.iloc[-1]
    wrist_points = list(zip(event["wrist_x"], event["wrist_y"]))
    ball_points = list(zip(event["ball_x"], event["ball_y"]))

    shoulder_angle = compute_angle(
        Keypoint(first["elbow_x"], first["elbow_y"]),
        Keypoint(first["shoulder_x"], first["shoulder_y"]),
        Keypoint(first["wrist_x"], first["wrist_y"]),
    )
    elbow_angle = compute_angle(
        Keypoint(first["shoulder_x"], first["shoulder_y"]),
        Keypoint(first["elbow_x"], first["elbow_y"]),
        Keypoint(first["wrist_x"], first["wrist_y"]),
    )
    wrist_angle = compute_angle(
        Keypoint(first["elbow_x"], first["elbow_y"]),
        Keypoint(first["wrist_x"], first["wrist_y"]),
        Keypoint(last["wrist_x"], last["wrist_y"]),
    )

    landing_x, landing_y = assign_landing_zone(last["ball_x"], last["ball_y"])
    approach_dx = last["wrist_x"] - first["wrist_x"]
    approach_dy = last["wrist_y"] - first["wrist_y"]

    return {
        "event_id": int(first["event_id"]),
        "video_id": first["video_id"],
        "rally_id": int(first["event_id"] // 2),
        "t_ms": float(last["timestamp_ms"]),
        "stroke_hand": determine_stroke_hand(last),
        "shoulder_angle": float(shoulder_angle),
        "elbow_angle": float(elbow_angle),
        "wrist_angle": float(wrist_angle),
        "body_rot": float(vector_magnitude(approach_dx, approach_dy)),
        "approach_vec": float(np.arctan2(approach_dy, approach_dx)),
        "serve_flag": int(first["event_id"] == 0),
        "landing_x": int(landing_x),
        "landing_y": int(landing_y),
        "contact_speed_proxy": float(rolling_speed(wrist_points) * CONFIG.contact_speed_scale),
        "ball_speed_proxy": float(rolling_speed(ball_points) * CONFIG.contact_speed_scale),
    }


def keypoints_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert extracted keypoints into per-stroke features."""
    if df.empty:
        raise ValueError("No keypoints provided")
    grouped = df.groupby("event_id")
    features = [_aggregate_event(event) for _, event in grouped]
    features_df = pd.DataFrame(features)
    return features_df


def merge_labels(features: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    """Attach optional quality labels to the feature dataframe."""
    if labels is None or labels.empty:
        features = features.copy()
        features["stroke_good"] = (features["contact_speed_proxy"] > features["contact_speed_proxy"].median()).astype(int)
        return features
    merged = features.merge(labels, on=["video_id", "t_ms"], how="left")
    merged["stroke_good"].fillna(merged["stroke_good"].median() or 0, inplace=True)
    merged["stroke_good"] = merged["stroke_good"].astype(int)
    return merged
