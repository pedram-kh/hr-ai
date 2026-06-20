"""hr-ai — FastAPI service (Sprint 0: scaffold only).

NO AI LOGIC IS BUILT THIS SPRINT. There is no embedding, vector search, query
routing, answer synthesis, or LLM tagging tier here, and hr-backend does not
call this service yet. The BGE-M3 retrieval / dimension-lock test
(data-model §12.1) is the first AI task of a later sprint.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config import settings
from .db import check_db_connection

app = FastAPI(title="hr-ai", version="0.0.1")


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
