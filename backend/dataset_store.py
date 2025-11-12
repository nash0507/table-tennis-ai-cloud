"""IBM Cloud Object Storage dataset repository helpers using fsspec/s3fs."""

from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd
from fsspec.core import url_to_fs

from .config import settings


class DatasetRepository:
    """Persist raw videos and derived features to a remote dataset store."""

    def __init__(self, base_uri: str, storage_options: Optional[dict] = None):
        self.base_uri = base_uri.rstrip("/")
        storage_options = storage_options or {}
        self.fs, self.base_path = url_to_fs(self.base_uri, **storage_options)
        if hasattr(self.fs, "makedirs") and self.base_path:
            try:
                self.fs.makedirs(self.base_path, exist_ok=True)
            except TypeError:
                self.fs.makedirs(self.base_path, True)

    def _full_path(self, relative: str) -> str:
        if not self.base_path:
            return relative
        if self.base_path.endswith("/"):
            return f"{self.base_path}{relative}"
        return f"{self.base_path}/{relative}"

    def _ensure_parent(self, path: str) -> None:
        if not hasattr(self.fs, "makedirs"):
            return
        parent = path.rsplit("/", 1)[0]
        if parent:
            try:
                self.fs.makedirs(parent, exist_ok=True)
            except TypeError:
                self.fs.makedirs(parent, True)

    def save_video(self, video_id: str, data: bytes) -> None:
        path = self._full_path(f"{settings.dataset_raw_prefix}{video_id}.mp4")
        self._ensure_parent(path)
        with self.fs.open(path, "wb") as handle:
            handle.write(data)

    def save_features(self, video_id: str, frame: pd.DataFrame) -> None:
        path = self._full_path(
            f"{settings.dataset_features_prefix}{settings.features_prefix}{video_id}{settings.features_suffix}"
        )
        self._ensure_parent(path)
        with self.fs.open(path, "wb") as handle:
            frame.to_csv(handle, index=False)

    def load_features(self, video_id: str) -> Optional[pd.DataFrame]:
        path = self._full_path(
            f"{settings.dataset_features_prefix}{settings.features_prefix}{video_id}{settings.features_suffix}"
        )
        if not self.fs.exists(path):
            return None
        with self.fs.open(path, "rb") as handle:
            return pd.read_csv(handle)

    def iter_feature_tables(self) -> Iterable[pd.DataFrame]:
        prefix = self._full_path(settings.dataset_features_prefix)
        try:
            entries: List[str] = self.fs.glob(f"{prefix}*{settings.features_suffix}")
        except (FileNotFoundError, NotImplementedError):
            entries = []
        for entry in sorted(entries):
            try:
                with self.fs.open(entry, "rb") as handle:
                    yield pd.read_csv(handle)
            except pd.errors.EmptyDataError:
                continue


DATASET_STORE = DatasetRepository(
    settings.dataset_repository_uri, settings.dataset_storage_options
)
