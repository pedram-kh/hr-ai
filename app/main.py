"""hr-ai — FastAPI service.

Sprint 1: document extraction (`/extract`, ADR-0010). Sprint 2a adds the
retrieval substrate:
- `/embed`   — re-extract column-aware, de-space, article-chunk, embed (BGE-M3/
               1024) and WRITE `document_chunks` directly (the one table hr-ai
               may write). hr-backend passes the resolved scope (ADR-0007/0013).
- `/extract-salary` — parse a salary `.xlsx` and RETURN structured rows
               (extract-and-return; hr-backend writes the salary tables).
- `/read-structured` — Sprint 7b-1 (ADR-0021): read a NON-salary `.docx`/`.xlsx`
               and RETURN structured per-section/per-sheet content. A
               content-extraction utility (no scope, no segmentation — that is
               7b-2). hr-backend persists it as display `document_pages`, never
               `document_chunks` (queried-not-embedded, ADR-0006).
- `/segment-facts` — Sprint 7b-2 (ADR-0022): read a multi-scope reference_source
               and SEGMENT it into per-scope facts BOUND to the closed convenio
               vocabulary (header-carry: scope resets on each TERRITORY/SECTOR
               header). A strict, inert proposer — RETURNS facts, writes nothing;
               hr-backend persists each as `ai_agent`/`needs_review` (ADR-0007).
- `/retrieve` — scope-prefilter (WHERE) then EXACT similarity ranking; full
               recall (catch 2). No router (that is 2b-2).
- `/compare-scope` — Sprint 7d (ADR-0024): the read-only semantic COMPARISON
               primitive. Embeds N probe texts and ranks a scope's chunks
               against each, with the `authority_level` band applied IN THE SQL
               so a threshold decision on the top score is k-independent (the
               publish fence is a safety gate; a post-top-k authority filter
               could hide the one overlapping passage — a fail-open). No LLM,
               no write, no migration.
- `/synthesise` — Sprint 2b-1 (ADR-0015): compose a CITED answer grounded ONLY
               in the eligible chunks hr-backend passes, honouring the
               convenio-over-baseline precedence rule. The provider is pluggable
               (default Claude). The API key arrives PER CALL from hr-backend and
               is NEVER stored, logged, or persisted here. hr-backend owns the
               answer-or-escalate decision; this endpoint only synthesises.

hr-ai still NEVER migrates and writes NO table other than `document_chunks`.
"""

from datetime import date

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .db import check_db_connection
from .extract import extract_pdf

app = FastAPI(title="hr-ai", version="0.2.0")


def require_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    """Guard internal service-to-service endpoints with a shared secret."""
    if x_internal_token != settings.internal_token:
        raise HTTPException(status_code=401, detail="Invalid internal token")


class ExtractRequest(BaseModel):
    storage_key: str
    document_uuid: str


class Scope(BaseModel):
    convenio_id: int | None = None
    territory_id: int | None = None
    sector_id: int | None = None
    validity_start: date | None = None
    validity_end: date | None = None
    retrieval_status: str | None = None
    authority_level: str | None = None


class EmbedRequest(BaseModel):
    document_id: int
    document_uuid: str
    storage_key: str
    scope: Scope


class SalaryExtractRequest(BaseModel):
    storage_key: str
    document_uuid: str


class ReadStructuredRequest(BaseModel):
    """Non-salary docx/xlsx content read (Sprint 7b-1, ADR-0021). Reads-and-
    returns ONLY — never writes the DB, never migrates (ADR-0007). A
    content-extraction utility: it does NOT decide scope or segment into facts
    (that is 7b-2). `format` ∈ docx | xlsx; a salary xlsx never reaches here (it
    is routed to /extract-salary by its document_type tag — Invariant 2)."""

    storage_key: str
    document_uuid: str
    format: str  # "docx" | "xlsx"


class RetrieveRequest(BaseModel):
    query: str
    convenio_id: int | None = None
    include_national_law: bool = True
    retrieval_status: list[str] = ["active"]
    as_of_date: date | None = None
    k: int = 8


class SandboxRetrieveRequest(BaseModel):
    """Single-document sandbox retrieval (Sprint 3). Read-only; ranks ONE
    document's chunks by similarity. ADDITIVE — the employee /retrieve and the
    whole answer loop are untouched (they never pass a document_id)."""

    query: str
    document_id: int
    k: int = 8


class CompareScopeRequest(BaseModel):
    """Sprint 7d (ADR-0024) — the semantic COMPARISON primitive. Read-only.

    Ranks a SCOPE's chunks (a convenio + an authority band) against N probe
    texts. `texts` are embedded here with the same BGE-M3 model the corpus was
    embedded with; nothing is written and nothing is persisted.

    Why this exists instead of reusing /retrieve: /retrieve has no
    `authority_level` filter, so the caller would have to filter AFTER the
    top-k — and an overlapping official-convenio passage could be crowded out
    of the top-k by other same-convenio chunks, making a SAFETY GATE report "no
    conflict" when there is one. Here `authority_levels` is applied in the SQL
    WHERE, so a threshold decision on `max_score` is k-independent.

    `document_ids` (optional) overrides `texts`: the probes are read from those
    documents' own chunk texts (the document↔document comparison used by the
    §8.5 reverse re-check and the succession proposal), so hr-backend never has
    to ship chunk text it already stored back over the wire.
    """

    texts: list[str] = []
    document_ids: list[int] = []  # probe side: use these documents' chunk texts
    probe_limit: int = 40  # cap on probes taken from document_ids
    convenio_id: int | None = None
    authority_levels: list[str] = ["official_convenio"]
    # Candidate side. `candidate_document_ids` (non-empty) pins the candidates to an
    # exact document list — hr-backend passes the `documents` registry's truth,
    # because the chunk table's denormalized scope copy is only refreshed on
    # re-embed. `retrieval_status: []` then means "don't filter on that stale copy".
    candidate_document_ids: list[int] = []
    retrieval_status: list[str] = ["active"]
    as_of_date: date | None = None
    exclude_document_ids: list[int] = []
    k: int = 10


class SynthesisChunk(BaseModel):
    # chunk_id is nullable + source_type discriminates the source kind (Sprint 7c
    # Q7, ADR-0023): a vector chunk carries its chunk_id; a structured reference
    # fact is `source_type="reference_fact"` with chunk_id=None (it is not a
    # chunk — ADR-0006). Purely additive: a prose-only turn sends no fact source
    # and every field defaults exactly as before (byte-for-byte identical request).
    chunk_id: int | None = None
    source_type: str = "chunk"  # 'chunk' | 'reference_fact'
    document_id: int
    page_from: int | None = None
    page_to: int | None = None
    content: str
    score: float = 0.0
    authority_level: str | None = None


class ProviderConfigBody(BaseModel):
    provider: str = "claude"
    model: str
    endpoint: str | None = None


class SynthesiseRequest(BaseModel):
    question: str
    chunks: list[SynthesisChunk]
    # The decrypted answer-model key, owned by hr-backend, passed in the BODY
    # (never a header) per call. Used for this one request only; never persisted.
    provider_api_key: str
    provider_config: ProviderConfigBody


class RouteRequest(BaseModel):
    """Router classification (Sprint 2b-2, ADR-0016). Sees the QUESTION only —
    never the chunks (the same privacy posture as /synthesise). The provider_config
    carries the SMALL/FAST router model. The key is hr-backend-owned, per call."""

    question: str
    provider_api_key: str
    provider_config: ProviderConfigBody


class GroundChunkBody(BaseModel):
    # Nullable chunk_id + source_type (Sprint 7c Q7, ADR-0023) — a reference fact
    # is entailed against its own quoted value with chunk_id=None. Additive: a
    # prose-only /ground call is byte-for-byte identical to today.
    chunk_id: int | None = None
    source_type: str = "chunk"  # 'chunk' | 'reference_fact'
    content: str
    authority_level: str | None = None
    is_tabular: bool = False


class GroundRequest(BaseModel):
    """Per-claim entailment grounding (Sprint 2b-2 §5). Uses the CAPABLE answer
    model (entailment is subtle). `chunks` are the CITED chunks only."""

    question: str
    answer: str
    chunks: list[GroundChunkBody]
    provider_api_key: str
    provider_config: ProviderConfigBody


class VocabularyCandidateBody(BaseModel):
    id: int
    name: str
    aliases: list[str] = []
    code: str | None = None


class ProposeTagsRequest(BaseModel):
    """Document-level facet tagging proposal (Sprint 7a, ADR-0011/0020).

    hr-backend passes the document's already-extracted `page_text` (no re-read)
    and the CLOSED candidate vocabulary so the model binds to real ids — it never
    invents a value. hr-ai READS and PROPOSES; it writes NOTHING (ADR-0007). The
    proposal is INERT: hr-backend persists it as `ai_agent` provenance and keeps
    the document `under_review` (the embedding gate) until a human verifies.
    Document-level facet tagging only — never multi-scope fact segmentation (7b).
    """

    document_id: int
    page_text: str
    candidate_vocabulary: dict[str, list[VocabularyCandidateBody]] = {}
    provider_api_key: str
    provider_config: ProviderConfigBody


class JobCategoryCandidateBody(BaseModel):
    id: int
    name: str
    group_code: str | None = None


class ConvenioCandidateBody(BaseModel):
    """One convenio in the segmentation candidate vocabulary (Sprint 7b-2).
    Convenio-centric: scope rides the convenio, so each carries its derived
    territory + sector (+ sparse job categories) so the model binds to a real id
    whose (territory, sector) matches the carried header."""

    id: int
    name: str
    numero: str | None = None
    aliases: list[str] = []
    territory_name: str = ""
    territory_aliases: list[str] = []
    sector_name: str = ""
    sector_aliases: list[str] = []
    job_categories: list[JobCategoryCandidateBody] = []


class SegmentFactsRequest(BaseModel):
    """Reference-source fact segmentation (Sprint 7b-2, ADR-0022).

    hr-backend passes the FULL concatenated /read-structured content (Q9 — never
    chunked, so header-carry sees the whole sequence) + the CLOSED candidate
    convenios (convenio-centric) + the approved topics. hr-ai READS and SEGMENTS
    into per-scope facts bound to real ids; it writes NOTHING (ADR-0007). The
    proposal is INERT: hr-backend persists each fact as `ai_agent`/`needs_review`
    (not answerable until a human verifies). No salary, no new vocabulary, no
    validity/authority (hr-backend owns those)."""

    document_id: int
    document_uuid: str = ""
    source_format: str = "docx"  # docx | xlsx — frames the prompt (prose vs grid)
    pages_text: str
    candidate_convenios: list[ConvenioCandidateBody] = []
    candidate_topics: list[VocabularyCandidateBody] = []
    provider_api_key: str
    provider_config: ProviderConfigBody


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "hr-ai"}


@app.get("/health/db")
async def health_db() -> JSONResponse:
    """Read-only DB connectivity check (no writes, no migrations)."""
    try:
        result = await check_db_connection()
        return JSONResponse({"status": "ok", **result})
    except Exception as exc:  # noqa: BLE001 - surface any connection failure
        return JSONResponse(
            {"status": "error", "connected": False, "detail": str(exc)},
            status_code=503,
        )


@app.get("/health/config")
async def health_config() -> dict[str, object]:
    """Echo non-secret config placeholders so the contract shape is visible."""
    return {
        "embed_model": settings.embed_model,
        "embed_dim": settings.embed_dim,
        # The answer model is external + pluggable (ADR-0015). These are NON-SECRET.
        # The key is NOT held here — it arrives per call from hr-backend.
        "answer_provider": settings.answer_provider,
        "answer_model": settings.answer_model,
        "answer_endpoint": settings.answer_endpoint,
        # The router (ADR-0016) — small/fast model, same key path. NON-SECRET.
        "router_model": settings.router_model,
        "router_endpoint": settings.router_endpoint or settings.answer_endpoint,
    }


@app.post("/extract", dependencies=[Depends(require_internal_token)])
def extract(req: ExtractRequest) -> JSONResponse:
    """PDF → per-page text + page-image S3 keys (ADR-0010).

    Reads the original from S3, writes page images to S3, returns page data.
    Never writes the database.
    """
    try:
        result = extract_pdf(req.storage_key, req.document_uuid)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - surface extraction/storage failure
        return JSONResponse(
            {"status": "error", "detail": str(exc)},
            status_code=502,
        )


@app.post("/embed", dependencies=[Depends(require_internal_token)])
async def embed(req: EmbedRequest) -> JSONResponse:
    """Re-extract column-aware → de-space → article-chunk → embed (BGE-M3/1024)
    → WRITE document_chunks (ADR-0013). hr-backend passes the resolved scope;
    the denormalized scope columns are copied verbatim. Idempotent re-embed.
    """
    from .pipeline import embed_document

    try:
        scope = req.scope.model_dump()
        scope["validity_start"] = req.scope.validity_start
        scope["validity_end"] = req.scope.validity_end
        result = await embed_document(req.document_id, req.storage_key, scope)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - surface embed/storage/db failure
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)


@app.post("/extract-salary", dependencies=[Depends(require_internal_token)])
def extract_salary(req: SalaryExtractRequest) -> JSONResponse:
    """Parse a salary .xlsx and RETURN structured rows (ADR-0010/0014). hr-ai
    writes NO salary rows — hr-backend writes salary_tables/_rows/categories.
    """
    from .salary import parse_salary_xlsx
    from .storage import get_object_bytes

    try:
        xlsx_bytes = get_object_bytes(req.storage_key)
        result = parse_salary_xlsx(xlsx_bytes)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - surface parse/storage failure
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)


@app.post("/read-structured", dependencies=[Depends(require_internal_token)])
def read_structured_endpoint(req: ReadStructuredRequest) -> JSONResponse:
    """Read a non-salary .docx/.xlsx → structured per-section/per-sheet content
    (Sprint 7b-1, ADR-0021). hr-ai READS and RETURNS — it writes NO DB rows and
    never migrates (ADR-0007). hr-backend persists the content as display
    `document_pages` (never `document_chunks` — queried-not-embedded, ADR-0006)
    and the human reads it to create reference facts by hand (7b-1). It does NOT
    segment or assign scope (that is the 7b-2 AI). A salary .xlsx is never sent
    here — it is routed to /extract-salary by its document_type tag (Invariant 2).
    """
    from .read_structured import read_structured

    try:
        result = read_structured(req.storage_key, req.document_uuid, req.format)
        return JSONResponse(result)
    except Exception as exc:  # noqa: BLE001 - surface read/storage failure
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)


@app.post("/retrieve", dependencies=[Depends(require_internal_token)])
async def retrieve_endpoint(req: RetrieveRequest) -> JSONResponse:
    """Scope-prefilter (WHERE on denormalized scope columns) THEN exact
    similarity ranking over document_chunks (data-model §11). Full recall — the
    ANN layer never drops an eligible chunk (catch 2). No router/answer LLM (2b).
    """
    from .chunks_db import count_eligible, retrieve
    from .embeddings import embed_query

    try:
        qvec = embed_query(req.query)
        chunks = await retrieve(
            qvec,
            req.convenio_id,
            req.include_national_law,
            req.retrieval_status,
            req.as_of_date,
            req.k,
        )
        eligible_total = await count_eligible(
            req.convenio_id, req.include_national_law, req.retrieval_status, req.as_of_date
        )
        for c in chunks:
            c["score"] = round(1.0 - float(c.pop("distance")), 6)
        return JSONResponse({"chunks": chunks, "eligible_total": eligible_total})
    except Exception as exc:  # noqa: BLE001 - surface retrieval failure
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)


@app.post("/sandbox-retrieve", dependencies=[Depends(require_internal_token)])
async def sandbox_retrieve(req: SandboxRetrieveRequest) -> JSONResponse:
    """Sandbox retrieval over ONE document's chunks (Sprint 3 Knowledge Center).

    Read-only and ADDITIVE: distinct from /retrieve so the employee answer loop's
    scope-prefilter primitive is provably untouched. hr-ai writes nothing here.
    """
    from .chunks_db import retrieve_by_document
    from .embeddings import embed_query

    try:
        qvec = embed_query(req.query)
        chunks = await retrieve_by_document(qvec, req.document_id, req.k)
        for c in chunks:
            c["score"] = round(1.0 - float(c.pop("distance")), 6)
        return JSONResponse({"chunks": chunks})
    except Exception as exc:  # noqa: BLE001 - surface retrieval failure
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)


@app.post("/compare-scope", dependencies=[Depends(require_internal_token)])
async def compare_scope_endpoint(req: CompareScopeRequest) -> JSONResponse:
    """Sprint 7d (ADR-0024) — the read-only semantic COMPARISON primitive.

    Embeds N probe texts and ranks the scope's chunks against each, with the
    `authority_level` band applied IN THE SQL so the top score is k-independent
    (see chunks_db.compare_scope). SELECT only: hr-ai writes nothing here and
    still never migrates (ADR-0007). No LLM — this is embed-and-rank, the same
    machinery the answer loop uses, pointed at a comparison job.

    `max_score` is the single number a safety gate reads: the best similarity
    any probe found against any eligible chunk. `eligible_total` is reported so
    the caller can tell "nothing close" apart from "nothing to compare against".
    """
    from .chunks_db import chunk_texts_for_document, compare_scope, count_eligible_in_scope
    from .embeddings import embed_texts

    try:
        # Probe side: explicit texts, or the chunk texts of the given documents.
        probes: list[dict] = [{"text": t, "source": None} for t in req.texts if t and t.strip()]
        for doc_id in req.document_ids:
            for row in await chunk_texts_for_document(doc_id, req.probe_limit):
                probes.append({
                    "text": row["content"],
                    "source": {
                        "document_id": doc_id,
                        "chunk_id": row["id"],
                        "chunk_index": row["chunk_index"],
                        "page_from": row["page_from"],
                    },
                })
        probes = probes[: req.probe_limit]

        if not probes:
            return JSONResponse({
                "matches": [], "max_score": None, "eligible_total": 0, "probe_count": 0,
            })

        vecs = embed_texts([p["text"] for p in probes])
        ranked = await compare_scope(
            vecs,
            req.convenio_id,
            req.authority_levels,
            req.retrieval_status,
            req.as_of_date,
            req.exclude_document_ids,
            req.candidate_document_ids,
            req.k,
        )
        eligible_total = await count_eligible_in_scope(
            req.convenio_id,
            req.authority_levels,
            req.retrieval_status,
            req.as_of_date,
            req.exclude_document_ids,
            req.candidate_document_ids,
        )

        matches = []
        max_score: float | None = None
        for idx, (probe, chunks) in enumerate(zip(probes, ranked, strict=True)):
            for c in chunks:
                c["score"] = round(1.0 - float(c.pop("distance")), 6)
                if max_score is None or c["score"] > max_score:
                    max_score = c["score"]
            matches.append({
                "probe_index": idx,
                "probe_source": probe["source"],
                "probe_excerpt": probe["text"][:400],
                "chunks": chunks,
            })

        return JSONResponse({
            "matches": matches,
            "max_score": max_score,
            "eligible_total": eligible_total,
            "probe_count": len(probes),
        })
    except Exception as exc:  # noqa: BLE001 - surface comparison failure to the caller
        # A non-2xx is what makes hr-backend take its fail-toward-caution branch
        # (never a silent "no conflict") — see SemanticFenceService.
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)


@app.post("/synthesise", dependencies=[Depends(require_internal_token)])
def synthesise(req: SynthesiseRequest) -> JSONResponse:
    """Compose a CITED answer grounded ONLY in the provided chunks (ADR-0015).

    The provider is pluggable (default Claude). The decrypted key arrives in the
    body per call and is NEVER stored, logged, or persisted. The precedence rule
    (convenio over national-law baseline) is encoded in the prompt; `authority_used`
    is computed deterministically from the cited chunks for the audit trail.

    On a provider failure this returns 200 with `{ "error": "provider_error", ... }`
    (the key is never echoed) so hr-backend can escalate (low_confidence) cleanly.
    """
    from .providers import ChunkInput, ProviderConfig, get_provider

    try:
        provider = get_provider(req.provider_config.provider)
        chunks = [
            ChunkInput(
                chunk_id=c.chunk_id,
                source_type=c.source_type,
                document_id=c.document_id,
                page_from=c.page_from,
                page_to=c.page_to,
                content=c.content,
                score=c.score,
                authority_level=c.authority_level,
            )
            for c in req.chunks
        ]
        config = ProviderConfig(
            provider=req.provider_config.provider,
            model=req.provider_config.model,
            endpoint=req.provider_config.endpoint,
        )
        result = provider.synthesise(req.question, chunks, req.provider_api_key, config)
        return JSONResponse(
            {
                "answer": result.answer,
                "citations": result.citations,
                "grounding_signal": result.grounding_signal,
                "confidence": result.confidence,
                "authority_used": result.authority_used,
                "trace_fragment": result.trace_fragment,
            }
        )
    except Exception as exc:  # noqa: BLE001 - provider/parse failure → escalation
        # NEVER include the request body (it carries the key). Only the message.
        return JSONResponse({"error": "provider_error", "detail": str(exc)}, status_code=200)


@app.post("/route", dependencies=[Depends(require_internal_token)])
def route(req: RouteRequest) -> JSONResponse:
    """Classify the question salary | prose | off_domain and decompose a compound
    question into subqueries (ADR-0016). Small/fast model. Runs AFTER hr-backend's
    hardcoded guardrail baseline (sensitive / other-employee never reach here).

    On a provider failure this returns 200 with `{ "error": "provider_error", ... }`
    so hr-backend stays FAIL-SAFE (the safe prose+floor path), never a misroute.
    """
    from .providers import ProviderConfig, get_provider

    try:
        provider = get_provider(req.provider_config.provider)
        config = ProviderConfig(
            provider=req.provider_config.provider,
            model=req.provider_config.model,
            endpoint=req.provider_config.endpoint,
        )
        result = provider.classify(req.question, req.provider_api_key, config)
        return JSONResponse(
            {
                "label": result.label,
                "confidence": result.confidence,
                "subqueries": result.subqueries,
                "reason": result.reason,
                "trace_fragment": result.trace_fragment,
            }
        )
    except Exception as exc:  # noqa: BLE001 - never echo the body (it carries the key)
        return JSONResponse({"error": "provider_error", "detail": str(exc)}, status_code=200)


@app.post("/ground", dependencies=[Depends(require_internal_token)])
def ground(req: GroundRequest) -> JSONResponse:
    """Per-claim entailment of a prose answer against its CITED chunks (Sprint
    2b-2 §5) — the REAL grounding gate. Uses the CAPABLE answer model (entailment
    is subtle; never the cheap router model). Table-aware (digit-presence is not
    entailment — Q5's lesson). Salary answers are SQL-grounded and skip this.

    On a provider failure this returns 200 with `{ "error": "provider_error", ... }`
    so hr-backend escalates (low_confidence) — never surfaces an unverified answer.
    """
    from .providers import GroundChunk, ProviderConfig, get_provider

    try:
        provider = get_provider(req.provider_config.provider)
        chunks = [
            GroundChunk(
                chunk_id=c.chunk_id,
                source_type=c.source_type,
                content=c.content,
                authority_level=c.authority_level,
                is_tabular=c.is_tabular,
            )
            for c in req.chunks
        ]
        config = ProviderConfig(
            provider=req.provider_config.provider,
            model=req.provider_config.model,
            endpoint=req.provider_config.endpoint,
        )
        result = provider.ground(req.question, req.answer, chunks, req.provider_api_key, config)
        return JSONResponse(
            {
                "grounded": result.grounded,
                "claims": result.claims,
                "ungrounded": result.ungrounded,
                "trace_fragment": result.trace_fragment,
            }
        )
    except Exception as exc:  # noqa: BLE001 - never echo the body (it carries the key)
        return JSONResponse({"error": "provider_error", "detail": str(exc)}, status_code=200)


@app.post("/propose-tags", dependencies=[Depends(require_internal_token)])
def propose_tags(req: ProposeTagsRequest) -> JSONResponse:
    """Read a document's text and PROPOSE document-level facets + confidence,
    bound to the CLOSED candidate vocabulary hr-backend passes (Sprint 7a,
    ADR-0011/0020). hr-ai READS and PROPOSES — it writes NOTHING and never
    migrates (ADR-0007). The proposal is INERT: hr-backend persists it as
    `ai_agent` provenance and keeps the document `under_review` (the embedding
    gate) until a human verifies. The AI never invents vocabulary (unresolvable
    values become raw_unmatched_values) and does DOCUMENT-LEVEL facet tagging
    only — never multi-scope fact segmentation (that is 7b).

    On a provider failure this returns 200 with `{ "error": "provider_error" }`
    (the key is never echoed) so hr-backend leaves the doc in the human queue —
    a tagging failure never blocks ingest or surfaces an answerable doc.
    """
    from .providers import ProviderConfig, VocabularyCandidate, get_provider

    try:
        provider = get_provider(req.provider_config.provider)
        config = ProviderConfig(
            provider=req.provider_config.provider,
            model=req.provider_config.model,
            endpoint=req.provider_config.endpoint,
        )
        candidates = {
            key: [
                VocabularyCandidate(id=c.id, name=c.name, aliases=c.aliases, code=c.code)
                for c in items
            ]
            for key, items in req.candidate_vocabulary.items()
        }
        result = provider.propose_tags(req.page_text, candidates, req.provider_api_key, config)
        return JSONResponse(
            {
                "facets": result.facets,
                "topics": result.topics,
                "raw_unmatched_values": result.raw_unmatched_values,
                "overall_confidence": result.overall_confidence,
                "trace_fragment": result.trace_fragment,
            }
        )
    except Exception as exc:  # noqa: BLE001 - never echo the body (it carries the key)
        return JSONResponse({"error": "provider_error", "detail": str(exc)}, status_code=200)


@app.post("/segment-facts", dependencies=[Depends(require_internal_token)])
def segment_facts(req: SegmentFactsRequest) -> JSONResponse:
    """Read a multi-scope reference_source's text and SEGMENT it into per-scope
    facts, each BOUND to a real convenio in the closed candidate list (Sprint
    7b-2, ADR-0022). hr-ai READS and SEGMENTS — it writes NOTHING and never
    migrates (ADR-0007). Each fact is INERT: hr-backend persists it as
    `ai_agent`/`needs_review` (not answerable until a human verifies) and forces
    the authority floor + the source validity. The AI never invents vocabulary
    (closed-set id validation), never writes a salary row, and flags uncertainty
    rather than guessing scope.

    The load-bearing prompt detail is HEADER-CARRY: scope resets on each
    TERRITORY/SECTOR header; the model re-derives the hierarchy from the text.

    On a provider failure this returns 200 with `{ "facts": [], "error": ... }`
    (the key is never echoed) so hr-backend leaves the source unsegmented in the
    human queue — a segmentation failure never blocks ingest or surfaces an
    answerable fact.
    """
    from .providers import (
        ConvenioCandidate,
        JobCategoryCandidate,
        ProviderConfig,
        VocabularyCandidate,
        get_provider,
    )

    try:
        provider = get_provider(req.provider_config.provider)
        config = ProviderConfig(
            provider=req.provider_config.provider,
            model=req.provider_config.model,
            endpoint=req.provider_config.endpoint,
        )
        convenios = [
            ConvenioCandidate(
                id=c.id,
                name=c.name,
                numero=c.numero,
                aliases=c.aliases,
                territory_name=c.territory_name,
                territory_aliases=c.territory_aliases,
                sector_name=c.sector_name,
                sector_aliases=c.sector_aliases,
                job_categories=[
                    JobCategoryCandidate(id=jc.id, name=jc.name, group_code=jc.group_code)
                    for jc in c.job_categories
                ],
            )
            for c in req.candidate_convenios
        ]
        topics = [
            VocabularyCandidate(id=t.id, name=t.name, aliases=t.aliases, code=t.code)
            for t in req.candidate_topics
        ]
        result = provider.segment_facts(
            req.pages_text, convenios, topics, req.provider_api_key, config
        )
        return JSONResponse({"facts": result.facts, "trace_fragment": result.trace_fragment})
    except Exception as exc:  # noqa: BLE001 - never echo the body (it carries the key)
        return JSONResponse({"facts": [], "error": "provider_error", "detail": str(exc)}, status_code=200)
