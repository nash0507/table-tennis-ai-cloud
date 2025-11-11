"""Application configuration values for paths and thresholds."""
from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    """Runtime configuration for the backend application."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("data/reports")
    models_dir: Path = Path("backend/models")
    features_prefix: str = "features_"
    labels_file: str = "labels.csv"
    features_suffix: str = ".csv"
    report_suffix: str = ".json"
    frames_per_event: int = 8
    min_events_per_video: int = 6

    class Config:
        arbitrary_types_allowed = True


settings = Settings()
