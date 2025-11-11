from __future__ import annotations

import pandas as pd

from backend.pipeline import metrics


def _sample_data():
    features_df = pd.DataFrame(
        [
            {
                "event_id": 0,
                "serve_flag": 1,
                "landing_x": 1,
                "landing_y": 2,
                "contact_speed_proxy": 200,
                "ball_speed_proxy": 180,
            },
            {
                "event_id": 1,
                "serve_flag": 0,
                "landing_x": 0,
                "landing_y": 1,
                "contact_speed_proxy": 150,
                "ball_speed_proxy": 160,
            },
        ]
    )
    preds_df = pd.DataFrame(
        [
            {
                "event_id": 0,
                "stroke_hand_pred": "forehand",
                "stroke_good_pred": 1,
                "confidence": 0.8,
            },
            {
                "event_id": 1,
                "stroke_hand_pred": "backhand",
                "stroke_good_pred": 0,
                "confidence": 0.6,
            },
        ]
    )
    return features_df, preds_df


def test_metrics_and_recommendations():
    features_df, preds_df = _sample_data()
    result = metrics.aggregate_metrics(features_df, preds_df)
    assert result["summary"]["total_events"] == 2
    assert len(result["recommendations"]) >= 3
    assert all(isinstance(rec, str) and rec for rec in result["recommendations"])
