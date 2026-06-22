"""Application configuration.

Sprint 1 added document extraction (`/extract`, ADR-0010). Sprint 2a adds:
- BGE-M3 embeddings (self-hosted, in-process; ADR-0006) — `vector(1024)`.
- The column-aware chunking pipeline (ADR-0013) settings.
- The salary `.xlsx` parser (extract-and-return; ADR-0010/0014).
- A WRITE path to `document_chunks` ONLY, via a dedicated, scoped Postgres role
  (`hr_ai`) created by an hr-backend migration (ADR-0007 enforced at the DB).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DSN against the Postgres owned/migrated by hr-backend. In Sprint 2a this is
    # the *scoped* `hr_ai` role: SELECT on registry/scope tables + INSERT/UPDATE/
    # DELETE on document_chunks ONLY (no other write, no DDL) — ADR-0007 enforced
    # at the database, not by convention. (Inside the dev container the host DB is
    # reached at host.docker.internal:55432.)
    database_url: str = "postgresql://hr_ai:hr_ai_secret@localhost:5432/hr_platform"

    # --- Document extraction (Sprint 1, ADR-0010) ---
    aws_endpoint: str = "http://localhost:9000"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    aws_region: str = "us-east-1"
    aws_bucket: str = "hr-documents"
    aws_use_path_style: bool = True
    extract_image_dpi: int = 150

    # Shared secret guarding internal endpoints (hr-backend ↔ hr-ai).
    internal_token: str = "dev-internal-token"

    # --- Embeddings (Sprint 2a, ADR-0006) ---
    # BGE-M3, multilingual, self-hostable, 1024-dim. Self-hosted in-process via
    # sentence-transformers (CPU acceptable — embedding is a background admin path).
    embed_model: str = "BGE-M3"
    embed_model_hf: str = "BAAI/bge-m3"
    embed_dim: int = 1024

    # --- Chunking (Sprint 2a, ADR-0013) ---
    # Article-aware chunking with a size cap; target ~400, hard cap ~512 tokens.
    chunk_token_target: int = 400
    chunk_token_cap: int = 512
    # De-spacing (geometry-first; tuned on the real Gipuzkoa file at the eyes-on
    # gate, plan §9 Q9): a glyph gap counts as a REAL space only when it exceeds
    # this fraction of the line's median glyph advance; smaller gaps are the
    # justification artifact and get merged.
    chunk_space_gap_ratio: float = 0.30
    # DEMOTED (Sprint 2a Correction-01): full width alone is NO LONGER furniture —
    # that bare rule silently ate full-width body prose (preámbulo paragraphs,
    # whole Navarra bodies). Furniture is now REPETITION at a margin band (below).
    # Retained only as a soft "wide block" reference for future tuning; not a
    # standalone stripper. A non-repeating full-width body block is kept as prose.
    chunk_furniture_width_ratio: float = 0.70
    # A header/footer line is furniture only when it REPEATS (same normalized text
    # in the top/bottom 12% y-band on at least this fraction of pages) — the
    # reliable signal (the bilingual BOG footer recurs on every page). This is the
    # catch-1 correctness win and the primary furniture rule.
    chunk_repeat_furniture_min_page_fraction: float = 0.30

    # --- Answer synthesis (Sprint 2b-1, ADR-0015) ---
    # The answer model is EXTERNAL and PLUGGABLE (quality-dominant trade, unlike
    # the self-hosted embedding model — ADR-0006). These are NON-SECRET settings:
    # which provider, which model, which endpoint. The API key is NOT here — it is
    # owned by hr-backend (encrypted at rest) and passed per synthesis call.
    # ANSWER_MODEL / ANSWER_ENDPOINT MUST point at an EU-available model/endpoint
    # (GDPR is deploy-time — deploy.md §1: EU endpoint, signed DPA, zero-retention).
    answer_provider: str = "claude"
    answer_model: str = "claude-sonnet-4-5"
    answer_endpoint: str = "https://api.anthropic.com"

    # DEPRECATED placeholder kept for /health/config back-compat; the real key is
    # never stored in hr-ai (ADR-0015) — it arrives per call from hr-backend.
    anthropic_api_key: str = ""


settings = Settings()
