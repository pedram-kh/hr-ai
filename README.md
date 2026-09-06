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
>
> **Sprint 7a: the LLM tagging tier (ADR-0020).** Adds `POST /propose-tags` — read
> a document's page text + the **closed candidate vocabulary** hr-backend passes
> (full territory/sector/document_type lists + a convenio shortlist) and
> **return** proposed facets + per-facet confidence + `raw_unmatched_values`
> (variant hints). Same key-in-the-body, never-persisted, no-DB posture as
> `/synthesise`·`/route`·`/ground`; **hr-ai writes nothing and never migrates**
> (ADR-0007). It is a pure proposer — hr-backend persists the result as inert
> `ai_agent` provenance (document stays `under_review`, FK scope columns
> untouched). Document-level facet tagging only (multi-scope fact segmentation is
> Sprint 7b).
>
> **Sprint 7b-1: the docx/xlsx reader (ADR-0021).** Adds `POST /read-structured` —
> read a **non-salary** `.docx` (python-docx) or `.xlsx` (openpyxl) and **return**
> per-section/per-sheet content for the manual reference-fact path. The format
> extension to ADR-0010 (PDF-only). A content-extraction utility only — no scope,
> no segmentation (that is 7b-2). hr-ai writes nothing, never migrates; hr-backend
> stores it as display `document_pages`, never embedded.
>
> **Sprint 7b-2: the segmentation agent (ADR-0022).** Adds `POST /segment-facts` —
> read a `reference_source`'s `document_pages` text + a **convenio-centric** closed
> candidate vocabulary (each convenio carries its derived territory/sector/job
> categories/aliases) + the approved topics, and **return** an array of per-scope
> proposed facts, each with `convenio_id` (+ optional `job_category_id`/`topic_id`),
> `value`/`raw_values`, `confidence`, structured `uncertainty`, and a mandatory
> `source_excerpt`. The core cognition is the **header-carry contract** (each value
> line inherits the most-recent TERRITORY+SECTOR, reset on each header — re-derived
> from text, since the `/read-structured` section split is style-dependent and
> unreliable across the fixtures). **Closed-set-validates every returned id**
> (convenio / per-convenio job category / topic) before returning — a hallucinated
> id can never reach hr-backend (ADR-0011). Same key-in-the-body, never-persisted,
> writes-nothing, never-migrates posture; on provider/parse failure returns
> `{ "facts": [], "error": "provider_error" }` (200) so the doc just stays
> unsegmented in the human queue. hr-backend's `ReferenceFactProposalService`
> persists the result as inert `ai_agent`/`needs_review` facts.

## Requirements

- Python 3.11+ (async)
- The infra from `hr-backend/docker-compose.yml` running (Postgres + MinIO/S3).
  In the dev container the host DB/S3 are reached at `host.docker.internal`
  (`:55432` Postgres, `:9900` MinIO).
- Embeddings pull `sentence-transformers` + `torch` (CPU) and download the
  `BAAI/bge-m3` weights (~2.3 GB) on first use.
- **S3 credentials (ADR-0009 — config only):** locally, `AWS_ACCESS_KEY_ID`/
  `AWS_SECRET_ACCESS_KEY` are always set (MinIO's static `minioadmin`/
  `minioadmin`). On staging/production, leave them **unset** — `app/storage.py`
  then omits explicit credentials from the `boto3` client and falls back to
  the default credential chain, which resolves the EC2 instance profile via
  IMDS automatically (no long-lived key on the box — that's what the instance
  profile exists to avoid). `AWS_ENDPOINT`/`AWS_REGION` are still set explicitly
  either way (real S3's regional endpoint instead of MinIO's).

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
- `POST /read-structured` (**internal**, Sprint 7b-1, ADR-0021) — body
  `{ storage_key, document_uuid, format }` (`format` ∈ `docx` | `xlsx`). Reads a
  **non-salary** `.docx` (python-docx) or `.xlsx` (openpyxl) and **returns**
  `{ format, pages:[{ page_number, label, text, locator }] }` — one row per docx
  section / xlsx sheet. A **content-extraction utility only**: it does NOT decide
  scope or segment into facts (that is 7b-2). hr-ai writes nothing, never
  migrates (ADR-0007); hr-backend stores the content as display `document_pages`,
  **never** `document_chunks` (queried-not-embedded, ADR-0006). A salary `.xlsx`
  never reaches here — it is routed to `/extract-salary` by its document_type tag.
- `POST /segment-facts` (**internal**, Sprint 7b-2, ADR-0022) — body
  `{ document_id, document_uuid, source_format, pages:[…],
  candidate_vocabulary:{ convenios:[{ id, numero, name, aliases, territory,
  sector, job_categories }], topics:[{ id, name }] }, provider_api_key,
  provider_config }`. Segments a `reference_source`'s text into **per-scope** facts
  via the **header-carry contract** and binds each to a real `convenio_id`;
  **returns** `{ facts:[{ convenio_id, job_category_id?, topic_id?, value,
  raw_values, validity_start?, validity_end?, confidence, uncertainty?,
  source_locator, source_excerpt }], trace_fragment }`. **Closed-set-validates
  every id** before returning. Key-in-the-body, never-persisted, writes-nothing,
  never-migrates; on failure → `{ facts: [], error: "provider_error" }` (200).
- `POST /retrieve` (**internal**) — body `{ query, convenio_id?,
  include_national_law, retrieval_status[], as_of_date?, k }`. Embeds the query,
  scope-prefilters `document_chunks`, ranks by an **exact flat scan** (full
  recall — the ANN layer never drops an eligible chunk), returns
  `{ chunks:[{ …, score }], eligible_total }`.
- `POST /synthesise` (**internal**, ADR-0015) — body `{ question,
  chunks:[{ chunk_id, document_id, page_from, page_to, content, score,
  authority_level, source_type? }], provider_api_key, provider_config:{ provider,
  model, endpoint } }`. Composes a cited answer grounded only in `chunks`, applying
  the convenio-over-baseline precedence rule. Returns `{ answer,
  citations:[{ chunk_id, document_id, page_from, page_to, authority_level,
  source_type }], grounding_signal:{ grounded, citation_count, top_chunk_score },
  confidence, authority_used:[…], trace_fragment }`. On a provider failure: `200`
  with `{ error:"provider_error", detail }` (the key is never echoed) so
  `hr-backend` escalates cleanly. **The key is used for this one call only — never
  persisted.** In-text `[Fuente N]` markers are renumbered to the cited subset
  (1..M). **7c composition (ADR-0023, additive):** `chunk_id` is now `int | None`
  and each source carries `source_type` (default `"chunk"`); a `structured_reference`
  reference fact is passed as one more typed source (`source_type:"reference_fact"`,
  `chunk_id:null`) ranked **below** `official_convenio` and above `national_law`
  (`_AUTHORITY_RANK`) — a fact never outranks a convenio. The citation dedup key is
  null-safe (a fact keys on `(source_type, document_id)`). A prose-only turn sends no
  fact source — the request is byte-for-byte identical to pre-7c.
- `POST /route` (**internal**, ADR-0016) — body `{ question, provider_api_key,
  provider_config:{ provider, model, endpoint } }` with the **router** model.
  Returns `{ label:"salary"|"prose"|"off_domain", confidence, subqueries:[…],
  reason, trace_fragment }`. On a provider failure: `200` with
  `{ error:"provider_error", detail }` so `hr-backend` fails safe to prose.
- `POST /ground` (**internal**, the grounding gate) — body `{ question, answer,
  chunks:[{ chunk_id, content, authority_level, is_tabular, source_type? }],
  provider_api_key, provider_config }` with the **answer** model. (`chunk_id` is
  `int | None` and `source_type` defaults `"chunk"` — 7c, additive: a composed
  answer's fact claim is entailed against the fact source, `chunk_id:null`.) Returns `{ grounded,
  claims:[{ claim, grounded, supporting_source }], ungrounded:[…],
  trace_fragment }`. Table-aware per-claim entailment; on a provider failure:
  `200` with `{ error:"provider_error", … }` so `hr-backend` escalates (not
  grounded). **The key is used for this one call only — never persisted.**
- `POST /propose-tags` (**internal**, ADR-0020) — body `{ document_id, page_text,
  candidate_vocabulary:{ territory:[{id,name}], sector:[…], document_type:[…],
  convenio:[…] }, provider_api_key, provider_config }` with the **answer** model.
  Returns `{ facets:[{ facet, value_id, value, confidence }], topics:[…],
  raw_unmatched_values:[{ facet, value, variant_of?, similarity? }],
  overall_confidence, trace_fragment }` — the model binds to **real ids** from the
  closed vocabulary (never free text); anything it cannot bind becomes a
  `raw_unmatched_value`. hr-ai writes nothing; hr-backend persists it as an inert
  proposal. **The key is used for this one call only — never persisted.**

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
