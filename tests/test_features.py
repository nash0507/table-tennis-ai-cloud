from __future__ import annotations

import pandas as pd

from backend.pipeline import features


def _make_keypoints() -> pd.DataFrame:
    rows = []
    for event_id in range(2):
        for frame in range(4):
            rows.append(
                {
                    "video_id": "vid",
                    "frame_idx": event_id * 4 + frame,
                    "event_id": event_id,
                    "timestamp_ms": frame * 16,
                    "shoulder_x": 0.4 + 0.1 * event_id,
                    "shoulder_y": 0.5,
                    "elbow_x": 0.45 + 0.1 * event_id,
                    "elbow_y": 0.55,
                    "wrist_x": 0.6 + 0.1 * event_id,
                    "wrist_y": 0.65,
                    "ball_x": 0.2 + 0.5 * event_id,
                    "ball_y": 0.3 + 0.4 * event_id,
                }
            )
    return pd.DataFrame(rows)


def test_stroke_hand_and_features():
    df = _make_keypoints()
    feats = features.keypoints_to_features(df)
    assert set(feats["stroke_hand"]) == {"forehand"}
    assert (feats["landing_x"] <= 2).all()
    assert (feats["landing_y"] <= 2).all()


def test_assign_landing_zone_bins():
    assert features.assign_landing_zone(0.05, 0.95) == (0, 2)
    assert features.assign_landing_zone(0.9, 0.1) == (2, 0)
