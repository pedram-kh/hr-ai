# hr-ai

Python + FastAPI service for the HR platform's RAG and reasoning pipeline. See
`AGENTS.md` and the canonical specs in `hr-docs`.

> **Sprint 0: scaffold only.** No AI logic is built yet — no embeddings, vector
> search, routing, answer synthesis, or LLM tagging — and `hr-backend` does not
> call this service this sprint. This repo **never runs migrations**; it only
> proves a read-only connection to the Postgres owned by `hr-backend`.

## Requirements

- Python 3.11+ (async)
- The infra from `hr-backend/docker-compose.yml` running (Postgres on `localhost:5432`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"
cp .env.example .env

uvicorn app.main:app --reload --port 8001
```

## Endpoints

- `GET /health` → `{ "status": "ok", "service": "hr-ai" }` (liveness)
- `GET /health/db` → read-only DB connectivity check (`SELECT 1` on a
  `read_only` session); `200` on success, `503` on failure
- `GET /health/config` → echoes non-secret config placeholders (`EMBED_MODEL`,
  `EMBED_DIM`, whether an Anthropic key is set) — none are exercised this sprint

## Lint

```bash
ruff check .
```
