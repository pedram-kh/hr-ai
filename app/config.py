"""Application configuration.

Sprint 0: config placeholders only. EMBED_MODEL / EMBED_DIM / ANTHROPIC_API_KEY
are declared so the shape is visible, but they are NOT exercised this sprint —
no embeddings, retrieval, routing, or answering logic is built yet.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Read-only DSN against the Postgres owned/migrated by hr-backend.
    # hr-ai never runs migrations; this sprint it only proves the read path.
    database_url: str = "postgresql://hr:hr_secret@localhost:5432/hr_platform"

    # --- Placeholders (unused in Sprint 0) ---
    embed_model: str = "BGE-M3"
    embed_dim: int = 1024
    anthropic_api_key: str = ""


settings = Settings()
