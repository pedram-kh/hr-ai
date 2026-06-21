# hr-ai

Python + FastAPI service for the HR platform's RAG and reasoning pipeline. See
`AGENTS.md` and the canonical specs in `hr-docs`.

> **Sprint 1: document extraction.** This service now does **document
> processing** (`/extract`, ADR-0010) in addition to the (still-unbuilt)
> RAG/reasoning pipeline — no embeddings, vector search, routing, answer
> synthesis, or LLM tagging yet. It **never writes the database** and **never
> migrates**: it reads originals from S3 and writes only page images to S3.

## Requirements

- Python 3.11+ (async)
- The infra from `hr-backend/docker-compose.yml` running (Postgres on
  `localhost:5432`, MinIO/S3 on `localhost:9000`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"
cp .env.example .env
#   Set the S3 (MinIO) credentials/bucket and INTERNAL_TOKEN to match hr-backend.

uvicorn app.main:app --reload --port 8001
```

## Endpoints

- `GET /health` → `{ "status": "ok", "service": "hr-ai" }` (liveness)
- `GET /health/db` → read-only DB connectivity check (`SELECT 1` on a
  `read_only` session); `200` on success, `503` on failure
- `GET /health/config` → echoes non-secret config placeholders (`EMBED_MODEL`,
  `EMBED_DIM`, whether an Anthropic key is set) — none are exercised this sprint
- `POST /extract` (**internal**, guarded by `X-Internal-Token`) — body
  `{ "storage_key": "...", "document_uuid": "..." }`. Reads the original PDF from
  S3, extracts per-page text (PyMuPDF), renders each page to a JPEG **written
  back to S3**, and returns
  `{ "page_count": N, "pages": [{ "page_number", "text", "image_key" }] }`.
  `hr-backend` persists the rows. PDF-only this sprint; image-only (scanned)
  pages yield empty `text` (no OCR) — the page image is still produced.

## Lint

```bash
ruff check .
```
