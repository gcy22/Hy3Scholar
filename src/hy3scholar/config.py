from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secrets are read from the environment only."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HY3_API_KEY", "TENCENT_TOKENHUB_API_KEY"),
    )
    base_url: str = Field(
        default="https://tokenhub.tencentmaas.com/v1",
        validation_alias="HY3_BASE_URL",
    )
    model: str = Field(default="hy3", validation_alias="HY3_MODEL")
    reasoning_effort: str = Field(
        default="high", validation_alias="HY3_REASONING_EFFORT"
    )
    temperature: float = Field(default=0.2, validation_alias="HY3_TEMPERATURE")
    top_p: float = Field(default=1.0, validation_alias="HY3_TOP_P")
    timeout_seconds: float = Field(
        default=180.0, validation_alias="HY3_TIMEOUT_SECONDS"
    )
    max_retries: int = Field(default=3, validation_alias="HY3_MAX_RETRIES")
    storage_dir: Path = Field(
        default=Path("data"), validation_alias="HY3SCHOLAR_STORAGE_DIR"
    )
    max_parallel_evaluators: int = Field(
        default=3, validation_alias="HY3SCHOLAR_MAX_PARALLEL_EVALUATORS"
    )
    uncertainty_threshold: float = Field(
        default=0.72, validation_alias="HY3SCHOLAR_UNCERTAINTY_THRESHOLD"
    )
    chunk_chars: int = 2200
    chunk_overlap_chars: int = 240
    default_top_k: int = 12
    max_context_chars: int = 52_000
    max_claims: int = 32
    max_key_points: int = 24
    web_search_source: str = Field(
        default="lite", validation_alias="HY3_WEB_SEARCH_SOURCE"
    )
    literature_timeout_seconds: float = Field(
        default=30.0, validation_alias="HY3SCHOLAR_LITERATURE_TIMEOUT_SECONDS"
    )
    literature_max_retries: int = Field(
        default=2, validation_alias="HY3SCHOLAR_LITERATURE_MAX_RETRIES"
    )
    openalex_api_key: str | None = Field(
        default=None, validation_alias="OPENALEX_API_KEY"
    )
    openalex_mailto: str | None = Field(
        default=None, validation_alias="OPENALEX_MAILTO"
    )
    crossref_mailto: str | None = Field(
        default=None, validation_alias="CROSSREF_MAILTO"
    )
    max_downloads: int = Field(
        default=10, ge=1, le=20, validation_alias="HY3SCHOLAR_MAX_DOWNLOADS"
    )
    max_pdf_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
        validation_alias="HY3SCHOLAR_MAX_PDF_BYTES",
    )
    arxiv_min_interval_seconds: float = Field(
        default=3.0,
        ge=0,
        validation_alias="HY3SCHOLAR_ARXIV_MIN_INTERVAL_SECONDS",
    )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "未检测到 Hy3 API Key。请设置 HY3_API_KEY 或 "
                "TENCENT_TOKENHUB_API_KEY；不要把密钥写入代码或提交到 Git。"
            )
        return self.api_key
