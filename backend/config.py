"""Application configuration values for paths and thresholds."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Runtime configuration for the backend application."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("data/reports")
    features_prefix: str = "features_"
    labels_file: str = "labels.csv"
    features_suffix: str = ".csv"
    report_suffix: str = ".json"
    frames_per_event: int = 8
    min_events_per_video: int = 6
    model_repository_uri: str = Field(
        default_factory=lambda: str(Path("data/model_store").absolute())
    )

    class Config:
        env_prefix = "TTAI_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()
