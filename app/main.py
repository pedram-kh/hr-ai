"""hr-ai — FastAPI service.

Sprint 1: document extraction (`/extract`, ADR-0010). Sprint 2a adds the
retrieval substrate:
- `/embed`   — re-extract column-aware, de-space, article-chunk, embed (BGE-M3/
               1024) and WRITE `document_chunks` directly (the one table hr-ai
               may write). hr-backend passes the resolved scope (ADR-0007/0013).
- `/extract-salary` — parse a salary `.xlsx` and RETURN structured rows
               (extract-and-return; hr-backend writes the salary tables).
- `/retrieve` — scope-prefilter (WHERE) then EXACT similarity ranking; full
               recall (catch 2). No router (that is 2b-2).
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


class RetrieveRequest(BaseModel):
    query: str
    convenio_id: int | None = None
    include_national_law: bool = True
    retrieval_status: list[str] = ["active"]
    as_of_date: date | None = None
    k: int = 8


class SynthesisChunk(BaseModel):
    chunk_id: int
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
