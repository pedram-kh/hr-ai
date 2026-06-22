"""Write/read access to `document_chunks` — the ONE table hr-ai may write.

Connects as the dedicated, scoped `hr_ai` Postgres role (SELECT on registry/
scope tables + INSERT/UPDATE/DELETE on document_chunks only; no other write, no
DDL) created by an hr-backend migration — ADR-0007 enforced at the database.

Retrieval is FULL-RECALL by construction (catch 2): the scope prefilter is a
plain WHERE on the denormalized scope columns, and we force an exact (flat) scan
for the similarity ordering so the ANN/HNSW layer can never silently drop an
eligible chunk. A legal-weight answer must never under-return eligible chunks;
correctness is never traded for ANN speed. At this corpus size the exact scan is
trivially fast (HNSW is not yet warranted — noted in review.md).
"""

from __future__ import annotations

import asyncpg

from .config import settings


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


async def _connect() -> asyncpg.Connection:
    return await asyncpg.connect(dsn=settings.database_url)


async def replace_document_chunks(
    document_id: int,
    scope: dict,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Idempotent re-embed: DELETE this document's chunks, then INSERT the fresh
    set in one transaction (plan §3.5). Returns rows written."""
    conn = await _connect()
    try:
        async with conn.transaction():
            await conn.execute("DELETE FROM document_chunks WHERE document_id = $1", document_id)
            written = 0
            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                await conn.execute(
                    """
                    INSERT INTO document_chunks
                        (document_id, chunk_index, page_from, page_to, content,
                         token_count, embedding,
                         convenio_id, territory_id, sector_id,
                         validity_start, validity_end, retrieval_status, authority_level,
                         created_at, updated_at)
                    VALUES
                        ($1, $2, $3, $4, $5,
                         $6, $7::vector,
                         $8, $9, $10,
                         $11, $12, $13, $14,
                         now(), now())
                    """,
                    document_id,
                    idx,
                    chunk.get("page_from"),
                    chunk.get("page_to"),
                    chunk["content"],
                    chunk.get("token_count", 0),
                    _vec_literal(emb),
                    scope.get("convenio_id"),
                    scope.get("territory_id"),
                    scope.get("sector_id"),
                    scope.get("validity_start"),
                    scope.get("validity_end"),
                    scope.get("retrieval_status"),
                    scope.get("authority_level"),
                )
                written += 1
            return written
    finally:
        await conn.close()


async def retrieve(
    query_vec: list[float],
    convenio_id: int | None,
    include_national_law: bool,
    statuses: list[str],
    as_of_date,
    k: int,
) -> list[dict]:
    """Scope-prefilter (WHERE) THEN exact similarity ranking (data-model §11).

    Forces a flat scan (`enable_indexscan/bitmapscan = off`) so the eligible set
    is filtered exactly and the top-k is exact — full recall guaranteed.
    """
    conn = await _connect()
    try:
        async with conn.transaction():
            # Force exact (flat) scan — never let HNSW post-filtering drop an
            # eligible chunk. Correctness over ANN speed (catch 2).
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL enable_bitmapscan = off")
            rows = await conn.fetch(
                """
                SELECT id, document_id, chunk_index, page_from, page_to, content,
                       retrieval_status, authority_level, convenio_id,
                       (embedding <=> $1::vector) AS distance
                FROM document_chunks
                WHERE ( ($2::bigint IS NOT NULL AND convenio_id = $2)
                        OR ($3::boolean AND authority_level = 'national_law') )
                  AND retrieval_status = ANY($4::varchar[])
                  AND ($5::date IS NULL OR validity_start IS NULL OR validity_start <= $5::date)
                  AND ($5::date IS NULL OR validity_end IS NULL OR validity_end >= $5::date)
                ORDER BY embedding <=> $1::vector
                LIMIT $6
                """,
                _vec_literal(query_vec),
                convenio_id,
                include_national_law,
                statuses,
                as_of_date,
                k,
            )
            return [dict(r) for r in rows]
    finally:
        await conn.close()


async def count_eligible(
    convenio_id: int | None,
    include_national_law: bool,
    statuses: list[str],
    as_of_date,
) -> int:
    """Exact count of the eligible set (used by the harness's full-recall
    assertion: every eligible chunk must be reachable)."""
    conn = await _connect()
    try:
        return await conn.fetchval(
            """
            SELECT count(*) FROM document_chunks
            WHERE ( ($1::bigint IS NOT NULL AND convenio_id = $1)
                    OR ($2::boolean AND authority_level = 'national_law') )
              AND retrieval_status = ANY($3::varchar[])
              AND ($4::date IS NULL OR validity_start IS NULL OR validity_start <= $4::date)
              AND ($4::date IS NULL OR validity_end IS NULL OR validity_end >= $4::date)
            """,
            convenio_id,
            include_national_law,
            statuses,
            as_of_date,
        )
    finally:
        await conn.close()
