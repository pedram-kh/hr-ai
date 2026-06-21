"""Application configuration.

Sprint 1 adds the document-extraction (`/extract`) config: S3 access (shared
MinIO bucket with hr-backend) and the internal service token. The embedding /
RAG placeholders remain unused this sprint.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Read-only DSN against the Postgres owned/migrated by hr-backend.
    # hr-ai never runs migrations; it only proves the read path this sprint.
    database_url: str = "postgresql://hr:hr_secret@localhost:5432/hr_platform"

    # --- Document extraction (Sprint 1, ADR-0010) ---
    # Shared S3-compatible object storage (MinIO in dev). hr-ai READS the
    # original and WRITES page images — object storage only, never the DB.
    aws_endpoint: str = "http://localhost:9000"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    aws_region: str = "us-east-1"
    aws_bucket: str = "hr-documents"
    aws_use_path_style: bool = True

    # Image render resolution for page rasterization.
    extract_image_dpi: int = 150

    # Shared secret guarding internal endpoints (hr-backend ↔ hr-ai).
    internal_token: str = "dev-internal-token"

    # --- Placeholders (unused this sprint) ---
    embed_model: str = "BGE-M3"
    embed_dim: int = 1024
    anthropic_api_key: str = ""


settings = Settings()
