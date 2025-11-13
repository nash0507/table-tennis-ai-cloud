"""Application configuration values for paths and thresholds."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseSettings, Field, validator


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
    dataset_repository_uri: str = Field(
        default_factory=lambda: str(Path("data/ibm_cos_dataset").absolute())
    )
    dataset_storage_options: dict = Field(default_factory=dict)
    ibm_cos_access_key_id: Optional[str] = None
    ibm_cos_secret_access_key: Optional[str] = None
    ibm_cos_endpoint_url: Optional[str] = None
    ibm_cos_region: Optional[str] = None
    ibm_cos_signature_version: str = "s3v4"
    dataset_raw_prefix: str = "videos/"
    dataset_features_prefix: str = "features/"

    @validator("dataset_storage_options", pre=True, always=True)
    def _inject_ibm_defaults(
        cls, value: Optional[Dict[str, Any]], values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge IBM Cloud Object Storage credentials into storage options."""

        options: Dict[str, Any] = dict(value or {})

        key = values.get("ibm_cos_access_key_id")
        secret = values.get("ibm_cos_secret_access_key")
        if key and secret:
            options.setdefault("key", key)
            options.setdefault("secret", secret)

        endpoint = values.get("ibm_cos_endpoint_url")
        region = values.get("ibm_cos_region")
        client_kwargs: Dict[str, Any] = dict(options.get("client_kwargs") or {})
        if endpoint and "endpoint_url" not in client_kwargs:
            client_kwargs["endpoint_url"] = endpoint
        if region and "region_name" not in client_kwargs:
            client_kwargs["region_name"] = region
        if client_kwargs:
            options["client_kwargs"] = client_kwargs

        signature = values.get("ibm_cos_signature_version")
        if signature:
            config_kwargs: Dict[str, Any] = dict(options.get("config_kwargs") or {})
            config_kwargs.setdefault("signature_version", signature)
            options["config_kwargs"] = config_kwargs

        return options

    class Config:
        env_prefix = "TTAI_"
        env_file = ".env"
        case_sensitive = False


settings = Settings()
