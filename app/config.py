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
    # "" not "minioadmin": local dev's .env ALWAYS sets both explicitly (MinIO's
    # static root user), so this default is never actually exercised there.
    # Found live on staging: pydantic-settings falls back to this class default
    # when the env var is absent (NOT to "" — that only happens if the var is
    # explicitly set to an empty string), so the old "minioadmin" default made
    # app/storage.py's `if settings.aws_access_key_id and ...:` fallback check
    # always see two non-empty strings and pass them to boto3 as real (bogus)
    # credentials — staging's deliberately-unset AWS_ACCESS_KEY_ID/SECRET
    # (ADR-0009: instance-profile credential chain) never actually took effect,
    # and every S3 call failed with InvalidAccessKeyId ("minioadmin" isn't a
    # real AWS key). "" is falsy, so the fallback now works as designed.
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
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
    # Sprint 8, Step 5 (plan.md §1.3, build-authorization additions): the cap
    # on `/embed-batch`'s `texts` list. The nightly question-cluster job
    # (hr-backend) batches its OWN caller-side loop at this size — hr-ai's
    # `embed_texts()` already internally batches at `batch_size=16`
    # (embeddings.py), this is a REQUEST-size cap, not a model batch size, to
    # keep one HTTP call bounded regardless of how many distinct questions a
    # given night's corpus has.
    embed_batch_max_texts: int = 256

    # --- Chunking (Sprint 2a ADR-0013; Sprint 2c ADR-0017) ---
    # ARTICLE-BOUNDARY chunking (Sprint 2c): one chunk per article, NO
    # cross-article packing (the buried-grant fix). `target` governs the
    # preamble / anchor-less paragraph fallback; `cap` is the size at which a
    # single oversized article is sub-split on a sub-clause/paragraph boundary.
    # Raised from 2a's 400/512 to keep a typical WHOLE article in one chunk
    # (the §1.4/build-time BGE-M3 distribution shows the vast majority of
    # articles fit under 512; only the disciplinary/classification/salud
    # long-tail exceeds 800 and is sub-split). Locked after the real-tokenizer
    # recompute — see sprint-02c-rechunk/review.md.
    chunk_token_target: int = 512
    chunk_token_cap: int = 800
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

    # --- Question router (Sprint 2b-2, ADR-0016) ---
    # The router is a SMALL/FAST model classification call (salary | prose |
    # off_domain) reusing the SAME pluggable provider + hr-backend-owned key path
    # (ADR-0015). The model is NON-SECRET config and MAY differ from the answer
    # model (smaller is fine). The endpoint defaults to ANSWER_ENDPOINT; if it
    # differs it MUST still be EU (deploy.md §1). The per-claim grounding check
    # (§5) deliberately uses ANSWER_MODEL, not this — entailment is subtle.
    router_model: str = "claude-haiku-4-5"
    router_endpoint: str = ""  # empty → falls back to answer_endpoint

    # --- OCR fallback (Sprint 7e, ADR-0026) ---
    # Deliberately its OWN config value, never aliased to answer_model: the
    # engine eval (sprint-07e/eval/, review.md §1) scored THREE Claude models
    # against human-corrected gold and the winner is NOT claude-sonnet-4-5 (the
    # current answer_model default) — claude-opus-5 measurably beats both
    # claude-sonnet-4-5 and claude-sonnet-5 on header survival, table-cell
    # accuracy, and eu-language WER, and its cost is trivial at the actual
    # backfill's page volume (≤105 pages) even though it's the most expensive
    # of the three per page. If answer_model is ever changed for unrelated
    # (chat-quality) reasons, ocr_model must NOT silently follow it — the OCR
    # decision is its own eval, its own ADR, its own model. Same EU-endpoint
    # constraint as answer_model/router_model applies (deploy.md §1); the
    # endpoint defaults to answer_endpoint unless overridden.
    ocr_model: str = "claude-opus-5"
    ocr_endpoint: str = ""  # empty → falls back to answer_endpoint

    # DEPRECATED placeholder kept for /health/config back-compat; the real key is
    # never stored in hr-ai (ADR-0015) — it arrives per call from hr-backend.
    anthropic_api_key: str = ""


settings = Settings()
