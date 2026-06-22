# hr-ai

Python + FastAPI service for the HR platform's RAG and reasoning pipeline. See
`AGENTS.md` and the canonical specs in `hr-docs`.

> **Sprint 2a: the retrieval substrate.** On top of Sprint 1 extraction
> (`/extract`, ADR-0010), this service now embeds prose into vectors, parses
> salary `.xlsx`, and serves vector retrieval:
> - `POST /embed` — re-extract a PDF **column-aware** (Euskara left / Spanish
>   right via PyMuPDF block bboxes, after stripping page furniture), normalize
>   the BOG intra-word spacing artifact, **article-chunk**, embed with **BGE-M3
>   (1024-dim, self-hosted in-process, CPU)**, and **write `document_chunks`**.
> - `POST /extract-salary` — parse a salary `.xlsx` and **return** structured
>   rows (extract-and-return; hr-ai writes no salary rows).
> - `POST /retrieve` — scope-prefilter then **exact** similarity ranking
>   (full recall).
>
> `document_chunks` is the **only** table hr-ai writes, via a dedicated, scoped
> `hr_ai` Postgres role (ADR-0007 enforced at the DB). It still **never
> migrates**. No router / answer LLM / guardrail yet (that is 2b).

## Requirements

- Python 3.11+ (async)
- The infra from `hr-backend/docker-compose.yml` running (Postgres + MinIO/S3).
  In the dev container the host DB/S3 are reached at `host.docker.internal`
  (`:55432` Postgres, `:9900` MinIO).
- Embeddings pull `sentence-transformers` + `torch` (CPU) and download the
  `BAAI/bge-m3` weights (~2.3 GB) on first use.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"
cp .env.example .env
#   DATABASE_URL must point at the scoped `hr_ai` role (created by an hr-backend
#   migration); set the S3 (MinIO) credentials/bucket + INTERNAL_TOKEN to match
#   hr-backend.

uvicorn app.main:app --reload --port 8001
```

## Endpoints

- `GET /health` → `{ "status": "ok", "service": "hr-ai" }` (liveness)
- `GET /health/db` → DB connectivity check; `200`/`503`
- `GET /health/config` → echoes non-secret config (`EMBED_MODEL`, `EMBED_DIM`, …)
- `POST /extract` (**internal**) — body `{ storage_key, document_uuid }`. PDF →
  per-page text + page-image S3 keys (Sprint 1). `hr-backend` persists the rows.
- `POST /embed` (**internal**) — body
  `{ document_id, document_uuid, storage_key, scope }` where `scope` carries the
  hr-backend-resolved `{ convenio_id, territory_id, sector_id, validity_start,
  validity_end, retrieval_status, authority_level }`. Re-extracts column-aware,
  de-spaces, article-chunks, embeds, and **writes `document_chunks`** (idempotent
  per document). Returns `{ chunks_written, language_streams, stats }` where
  `stats` includes `furniture_blocks_stripped`, `repeating_furniture_lines`,
  `pages_not_cleanly_split` (for the eyes-on gate).
- `POST /extract-salary` (**internal**) — body `{ storage_key, document_uuid }`.
  Parses the `.xlsx` (skips junk sheets, finds the header row, maps cryptic
  columns per format, multi-year → many tables) and **returns**
  `{ tables:[{ sheet, year, rows:[{ job_category_name, group_code, gross_annual,
  base_salary_monthly, num_payments, hourly_rate, extra_pay, night_plus,
  raw_values }] }], warnings }`. hr-ai writes nothing; hr-backend writes the rows.
- `POST /retrieve` (**internal**) — body `{ query, convenio_id?,
  include_national_law, retrieval_status[], as_of_date?, k }`. Embeds the query,
  scope-prefilters `document_chunks`, ranks by an **exact flat scan** (full
  recall — the ANN layer never drops an eligible chunk), returns
  `{ chunks:[{ …, score }], eligible_total }`.

## Sanity test (BGE-M3 / 1024 go-no-go)

```bash
docker exec hr_ai python scripts/sanity_test.py
```

Embeds real ES + EU chunks (Gipuzkoa eu+es + the Estatuto es) and reports
same-language self-retrieval accuracy + extraction stats. Run **before** any
bulk embed (`chunks:embed` on the hr-backend side).

## Lint

```bash
ruff check .
```
