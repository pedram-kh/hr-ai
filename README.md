# hr-ai

Python + FastAPI service for the HR platform's RAG and reasoning pipeline. See
`AGENTS.md` and the canonical specs in `hr-docs`.

> **Sprint 2a: the retrieval substrate.** On top of Sprint 1 extraction
> (`/extract`, ADR-0010), this service now embeds prose into vectors, parses
> salary `.xlsx`, and serves vector retrieval:
> - `POST /embed` — re-extract a PDF **column-aware** (Euskara left / Spanish
>   right via PyMuPDF block bboxes, after stripping page furniture), normalize
>   the BOG intra-word spacing artifact, **article-boundary chunk** (one chunk per
>   article, ADR-0017), embed with **BGE-M3 (1024-dim, self-hosted in-process,
>   CPU)**, and **write `document_chunks`**.
> - `POST /extract-salary` — parse a salary `.xlsx` and **return** structured
>   rows (extract-and-return; hr-ai writes no salary rows).
> - `POST /retrieve` — scope-prefilter then **exact** similarity ranking
>   (full recall).
>
> `document_chunks` is the **only** table hr-ai writes, via a dedicated, scoped
> `hr_ai` Postgres role (ADR-0007 enforced at the DB). It still **never
> migrates**.
>
> **Sprint 2b-1: answer synthesis (ADR-0015).** Adds `POST /synthesise` — compose
> a **cited** answer grounded *only* in the eligible chunks `hr-backend` passes,
> honouring the **convenio-over-`national_law` precedence rule** (encoded in the
> prompt; `authority_used` returned for the audit trace). The provider is
> **pluggable** (`app/providers/`, default Claude); the API key arrives in the
> request **body per call** and is **never stored, logged, or persisted** here —
> `hr-backend` owns it (ADR-0015) and owns the answer-or-escalate decision.
>
> **Sprint 2b-2: the router + per-claim grounding (ADR-0016).** Adds two more
> provider calls on the **same** key path (key in the body, per call, never
> persisted):
> - `POST /route` — a **small/fast** model (`ROUTER_MODEL`) classifies the
>   question `salary` \| `prose` \| `off_domain` and decomposes a **compound**
>   question into `subqueries`. Sees the question only (never the chunks);
>   `hr-backend`'s guardrail baseline fires first, so sensitive/other-employee
>   never reach it. Fail-safe by contract: a parse/transport failure returns a
>   low-confidence prose result.
> - `POST /ground` — the **per-claim entailment** grounding check using the
>   **capable answer model** (entailment is subtle — never the cheap router
>   model). **Table-aware**: digit-presence in a tabular chunk is not entailment.
> Synthesis also now **renumbers the `[Fuente N]` markers** to the cited subset
> (1..M) so they map 1:1 to the displayed sources. Salary-in-chat is SQL in
> `hr-backend` (not here — ADR-0006).
>
> **Sprint 2c: article-boundary chunking (ADR-0017).** A **substrate** change in
> `app/chunking/chunker.py` only — the answer loop is untouched. Each detected
> article is now its **own chunk** and **cross-article packing is removed** (the
> durable fix for the buried-grant artifact behind 2b-2 Correction-03). Only an
> article over `chunk_token_cap` (**800** tok; preamble/fallback target **512**)
> is sub-split on a sub-clause/paragraph/sentence boundary (never mid-sentence),
> carrying its `Artículo N.º <título>` header onto each sub-chunk. The
> now-load-bearing header detector runs with **three precision guards** —
> **line-anchored, case-aware, monotonic-number** — so an inline `…del artículo
> 22…` cannot spawn a chunk (variants: `Artículo N` · `Art. N.º` ·
> `Artículo N.—/-` · Salamanca `ART N.-` · `N. artikulua` · `Disposición …` ·
> defensive spelled-out). It **composes with — never replaces** — the 2a
> extraction front-end (`extract_columns.py`: de-spacing, furniture stripping,
> two-column positive-evidence detection, language gate, language tagging — all
> unchanged). Re-chunk is the existing idempotent `chunks:embed`.

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
- `GET /health/config` → echoes non-secret config (`EMBED_MODEL`, `EMBED_DIM`,
  `ANSWER_PROVIDER`, `ANSWER_MODEL`, `ANSWER_ENDPOINT`, `ROUTER_MODEL`,
  `ROUTER_ENDPOINT` — the answer key is **not** here; it arrives per call from
  `hr-backend`)
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
- `POST /synthesise` (**internal**, ADR-0015) — body `{ question,
  chunks:[{ chunk_id, document_id, page_from, page_to, content, score,
  authority_level }], provider_api_key, provider_config:{ provider, model,
  endpoint } }`. Composes a cited answer grounded only in `chunks`, applying the
  convenio-over-baseline precedence rule. Returns `{ answer,
  citations:[{ chunk_id, document_id, page_from, page_to, authority_level }],
  grounding_signal:{ grounded, citation_count, top_chunk_score }, confidence,
  authority_used:[…], trace_fragment }`. On a provider failure: `200` with
  `{ error:"provider_error", detail }` (the key is never echoed) so `hr-backend`
  escalates cleanly. **The key is used for this one call only — never persisted.**
  In-text `[Fuente N]` markers are renumbered to the cited subset (1..M).
- `POST /route` (**internal**, ADR-0016) — body `{ question, provider_api_key,
  provider_config:{ provider, model, endpoint } }` with the **router** model.
  Returns `{ label:"salary"|"prose"|"off_domain", confidence, subqueries:[…],
  reason, trace_fragment }`. On a provider failure: `200` with
  `{ error:"provider_error", detail }` so `hr-backend` fails safe to prose.
- `POST /ground` (**internal**, the grounding gate) — body `{ question, answer,
  chunks:[{ chunk_id, content, authority_level, is_tabular }], provider_api_key,
  provider_config }` with the **answer** model. Returns `{ grounded,
  claims:[{ claim, grounded, supporting_source }], ungrounded:[…],
  trace_fragment }`. Table-aware per-claim entailment; on a provider failure:
  `200` with `{ error:"provider_error", … }` so `hr-backend` escalates (not
  grounded). **The key is used for this one call only — never persisted.**

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
