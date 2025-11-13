"""Aggregation and report generation utilities."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _assign_player(event_id: int) -> str:
    return "A" if event_id % 2 == 0 else "B"


def _player_stats(df: pd.DataFrame) -> Dict[str, float]:
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "forehand_ratio": 0.0,
            "backhand_ratio": 0.0,
            "good_rate": 0.0,
            "serve_success": 0.0,
            "avg_contact_speed": 0.0,
            "avg_ball_speed": 0.0,
        }
    forehand = (df["stroke_hand_pred"] == "forehand").sum()
    good = df["stroke_good_pred"].sum()
    serves = df[df["serve_flag"] == 1]
    serve_success = serves["stroke_good_pred"].mean() if not serves.empty else 0.0
    return {
        "total": total,
        "forehand_ratio": float(forehand / total),
        "backhand_ratio": float(1 - forehand / total),
        "good_rate": float(good / total),
        "serve_success": float(serve_success or 0.0),
        "avg_contact_speed": float(df["contact_speed_proxy"].mean()),
        "avg_ball_speed": float(df["ball_speed_proxy"].mean()),
    }


def compute_heatmap(df: pd.DataFrame) -> List[List[float]]:
    grid = np.zeros((3, 3), dtype=float)
    for _, row in df.iterrows():
        grid[int(row["landing_y"]), int(row["landing_x"])] += 1 - row["stroke_good_pred"]
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid.tolist()


def build_radar(player_a: Dict[str, float], player_b: Dict[str, float]) -> List[Dict[str, float]]:
    metrics = [
        "forehand_ratio",
        "good_rate",
        "serve_success",
        "avg_contact_speed",
        "avg_ball_speed",
    ]
    radar = []
    for metric in metrics:
        a_val = float(player_a.get(metric, 0.0))
        b_val = float(player_b.get(metric, 0.0))
        radar.append(
            {
                "metric": metric,
                "player_a": a_val,
                "player_b": b_val,
                "delta": a_val - b_val,
            }
        )
    return radar


def generate_recommendations(radar: List[Dict[str, float]], heatmap: List[List[float]]) -> List[str]:
    recs: List[str] = []
    for entry in radar:
        delta = entry["delta"]
        metric = entry["metric"]
        if metric == "good_rate" and abs(delta) > 0.1:
            better = "A" if delta > 0 else "B"
            recs.append(
                f"Player {better} converts strokes {abs(delta):.0%} better; focus on shot selection for the opponent."
            )
        if metric == "serve_success" and abs(delta) > 0.15:
            weaker = "A" if delta < 0 else "B"
            recs.append(f"Player {weaker} should drill serve consistency to close the gap.")
    heatmap_arr = np.array(heatmap)
    hot_y, hot_x = np.unravel_index(np.argmax(heatmap_arr), heatmap_arr.shape)
    if heatmap_arr[hot_y, hot_x] > 0.5:
        recs.append(
            f"Reduce errors targeting cell ({hot_x + 1}, {hot_y + 1}); consider adjusting positioning during rallies."
        )
    if not recs:
        recs.append("Maintain current training focus; both players show balanced performance.")
    while len(recs) < 3:
        recs.append("Introduce multi-ball drills emphasising backhand recovery speed.")
    return recs[:5]


def aggregate_metrics(features: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, object]:
    merged = features.merge(predictions, on="event_id")
    merged["player"] = merged["event_id"].apply(_assign_player)
    player_a = _player_stats(merged[merged["player"] == "A"])
    player_b = _player_stats(merged[merged["player"] == "B"])
    heatmap = compute_heatmap(merged)
    radar = build_radar(player_a, player_b)
    recommendations = generate_recommendations(radar, heatmap)
    summary = {
        "total_events": int(len(merged)),
        "player_a_events": int(player_a["total"]),
        "player_b_events": int(player_b["total"]),
    }
    return {
        "summary": summary,
        "player_a": player_a,
        "player_b": player_b,
        "heatmap": heatmap,
        "radar": radar,
        "recommendations": recommendations,
        "merged": merged,
    }
