"""Pydantic schemas for API requests and responses."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class VideoUploadResponse(BaseModel):
    video_id: str


class AnalyzeResponse(BaseModel):
    video_id: str
    events: int
    report_path: str


class StrokePrediction(BaseModel):
    event_id: int
    stroke_hand: str
    stroke_good: int
    confidence: float


class RadarEntry(BaseModel):
    metric: str
    player_a: float
    player_b: float
    delta: float


class Report(BaseModel):
    video_id: str
    summary: dict
    stroke_predictions: List[StrokePrediction]
    aggregates: dict
    radar: List[RadarEntry]
    heatmap: List[List[float]]
    feature_importances: List[dict]
    recommendations: List[str]
    raw_events_path: Optional[str]
