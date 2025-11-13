"""Model repository abstraction backed by fsspec for cloud storage."""
from __future__ import annotations

from typing import Any, Optional

import joblib
from fsspec.core import url_to_fs

from .config import settings


class ModelRepository:
    """Persist and retrieve models from a configurable URI."""

    def __init__(self, base_uri: str):
        self.base_uri = base_uri.rstrip("/")
        self.fs, self.base_path = url_to_fs(self.base_uri)
        if hasattr(self.fs, "makedirs") and self.base_path:
            try:
                self.fs.makedirs(self.base_path, exist_ok=True)
            except TypeError:
                self.fs.makedirs(self.base_path, True)

    def _full_path(self, name: str) -> str:
        if not self.base_path:
            return name
        if self.base_path.endswith("/"):
            return f"{self.base_path}{name}"
        return f"{self.base_path}/{name}"

    def exists(self, name: str) -> bool:
        return bool(self.fs.exists(self._full_path(name)))

    def save(self, name: str, model: Any) -> None:
        path = self._full_path(name)
        with self.fs.open(path, "wb") as handle:
            joblib.dump(model, handle)

    def load(self, name: str) -> Optional[Any]:
        path = self._full_path(name)
        if not self.fs.exists(path):
            return None
        with self.fs.open(path, "rb") as handle:
            try:
                return joblib.load(handle)
            except Exception:
                return None


MODEL_STORE = ModelRepository(settings.model_repository_uri)

