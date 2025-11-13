"""Simple persistence helpers for the MVP pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import settings
from .dataset_store import DATASET_STORE


def ensure_directories() -> None:
    """Ensure that all data directories exist."""
    for directory in [
        settings.raw_dir,
        settings.interim_dir,
        settings.processed_dir,
        settings.reports_dir,
    ]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_features(video_id: str, features: pd.DataFrame) -> Path:
    """Persist the generated feature dataframe to disk."""
    ensure_directories()
    file_path = (
        settings.processed_dir
        / f"{settings.features_prefix}{video_id}{settings.features_suffix}"
    )
    features.to_csv(file_path, index=False)
    return file_path


def load_features(video_id: str) -> Optional[pd.DataFrame]:
    """Load features for a video if they have been generated."""
    file_path = (
        settings.processed_dir
        / f"{settings.features_prefix}{video_id}{settings.features_suffix}"
    )
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


def load_labels() -> Optional[pd.DataFrame]:
    """Load accumulated human labels if available."""
    labels_file = settings.processed_dir / settings.labels_file
    if labels_file.exists():
        return pd.read_csv(labels_file)
    return None


def save_report(video_id: str, report: dict) -> Path:
    """Persist the final analysis report as JSON."""
    ensure_directories()
    file_path = settings.reports_dir / f"{video_id}{settings.report_suffix}"
    file_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def load_report(video_id: str) -> Optional[dict]:
    """Load a previously generated analysis report."""
    file_path = settings.reports_dir / f"{video_id}{settings.report_suffix}"
    if file_path.exists():
        return json.loads(file_path.read_text(encoding="utf-8"))
    return None


def load_all_features() -> pd.DataFrame:
    """Return all stored feature tables for incremental training."""
    ensure_directories()
    pattern = f"{settings.features_prefix}*{settings.features_suffix}"
    frames = []
    for file_path in sorted(settings.processed_dir.glob(pattern)):
        try:
            frames.append(pd.read_csv(file_path))
        except pd.errors.EmptyDataError:
            continue
    try:
        frames.extend(list(DATASET_STORE.iter_feature_tables()))
    except Exception:
        pass
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if {"video_id", "event_id"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["video_id", "event_id"], keep="last")
    return combined
