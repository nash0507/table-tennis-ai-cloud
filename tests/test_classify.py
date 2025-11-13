from __future__ import annotations

import numpy as np
import pandas as pd

from backend.pipeline import classify


def _feature_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for event_id in range(8):
        rows.append(
            {
                "event_id": event_id,
                "shoulder_angle": rng.uniform(20, 160),
                "elbow_angle": rng.uniform(30, 170),
                "wrist_angle": rng.uniform(10, 150),
                "body_rot": rng.uniform(0.1, 1.5),
                "approach_vec": rng.uniform(-3.14, 3.14),
                "serve_flag": int(event_id == 0),
                "landing_x": rng.integers(0, 3),
                "landing_y": rng.integers(0, 3),
                "contact_speed_proxy": rng.uniform(50, 300),
                "ball_speed_proxy": rng.uniform(40, 280),
                "stroke_hand": "forehand" if event_id % 2 == 0 else "backhand",
                "stroke_good": int(event_id % 3 != 0),
            }
        )
    return pd.DataFrame(rows)


def test_classifier_training_and_prediction():
    features_df = _feature_frame()
    classifier = classify.train_classifier(features_df)
    preds = classifier.predict(features_df)
    assert set(preds["stroke_hand_pred"]).issubset({"forehand", "backhand"})
    fi = classifier.feature_importances()
    assert isinstance(fi, dict)
    assert all(name in fi for name in classify.FEATURE_COLUMNS)
