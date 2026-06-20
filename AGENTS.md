# hr-ai — agent instructions

Python FastAPI service for the HR platform's RAG and reasoning: embeddings, vector search, query routing, answer synthesis, and the LLM tagging tier. It is called by `hr-backend`.

## Read before building
Canonical specs in the `hr-docs` repo (beside this one in the workspace):
- `architecture.md`, `data-model.md`, `decisions/`, `glossary.md`.
Read the relevant doc before coding. For the current task, read the active sprint spec in `hr-docs/sprints/`.

## Non-negotiable rules
- **Never run migrations.** `hr-backend` owns the schema. This service READS registry/scope tables and READS & WRITES `document_chunks` only.
- **Pre-filter before similarity:** vector search over `document_chunks` MUST filter by the denormalized scope columns (convenio_id, province_id, sector_id, validity, retrieval_status, authority_level) BEFORE ranking. Only `retrieval_status = active` is cited as current; `historical` answers time-scoped questions but is never cited as current.
- **Salary questions are SQL lookups**, not vector search — route them to structured queries over `salary_tables` / `salary_table_rows`.
- **Language is never a filter.** Search across all languages; language is metadata only.
- **The tagging agent proposes, never creates** controlled-vocabulary values. Low-confidence or registry-conflicting tags go to a review queue; conflict detection outranks confidence.
- **Every answer returns a structured trace** (profile, scope filters, router decision, retrieved chunks + scores, guardrail result, confidence).
- **The backend contract:** receive `{query, scope_filters}`, return `{answer, citations, confidence, trace}`. Keep in sync with `hr-backend/AGENTS.md`.

## Stack & conventions
- Python 3.11+, FastAPI, async. PostgreSQL via an async driver; pgvector.
- Embeddings: BGE-M3, 1024-dim. Confirm retrieval quality on real ES + EU convenios before relying on it (the first real AI task).
- LLM calls: Anthropic API.
- Type hints required; `ruff` for lint/format.

## Workflow
For any sprint: read the spec, write `plan.md`, STOP for review before building. When given a correction, apply it AND record it in the named doc as instructed.
