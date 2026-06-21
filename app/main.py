"""hr-ai — FastAPI service.

Sprint 1 adds document extraction (`/extract`, ADR-0010): hr-backend uploads a
PDF original to S3, then calls this endpoint; hr-ai reads it, produces per-page
text + page images (written to S3), and returns the data. hr-backend writes the
documents/document_pages rows. hr-ai still NEVER writes the database and NEVER
migrates — no embeddings, vector search, routing, answering, or LLM tagging is
built yet.
"""

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import settings
from .db import check_db_connection
from .extract import extract_pdf

app = FastAPI(title="hr-ai", version="0.1.0")


def require_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    """Guard internal service-to-service endpoints with a shared secret."""
    if x_internal_token != settings.internal_token:
        raise HTTPException(status_code=401, detail="Invalid internal token")


class ExtractRequest(BaseModel):
    storage_key: str
    document_uuid: str


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
        "anthropic_api_key_configured": bool(settings.anthropic_api_key),
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
