"""Read-only database connectivity check.

This proves the read path only. hr-backend owns the schema and all migrations;
hr-ai never writes DDL and (in later sprints) only reads registry/scope tables
and reads & writes `document_chunks`. We explicitly set the session to
read-only to make that intent concrete.
"""

import asyncpg

from .config import settings


async def check_db_connection() -> dict[str, object]:
    conn = await asyncpg.connect(dsn=settings.database_url)
    try:
        # Force a read-only session — no writes, no DDL.
        await conn.execute("SET default_transaction_read_only = on")
        value = await conn.fetchval("SELECT 1")
        return {"connected": True, "select_1": value}
    finally:
        await conn.close()
