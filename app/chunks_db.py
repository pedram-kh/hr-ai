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


async def retrieve_by_document(
    query_vec: list[float],
    document_id: int,
    k: int,
) -> list[dict]:
    """Sandbox retrieval (Sprint 3): rank ONE document's chunks by similarity to
    the query. Read-only, no scope filter beyond `document_id` — the Knowledge
    Center "test a question against this document" panel scopes to a single doc.

    This is ADDITIVE and DISTINCT from `retrieve()`: the employee answer loop
    never passes a document_id and its primitive is untouched. Same exact (flat)
    scan posture for determinism. hr-ai stays read-only (SELECT only here).
    """
    conn = await _connect()
    try:
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL enable_bitmapscan = off")
            rows = await conn.fetch(
                """
                SELECT id, document_id, chunk_index, page_from, page_to, content,
                       retrieval_status, authority_level, convenio_id,
                       (embedding <=> $1::vector) AS distance
                FROM document_chunks
                WHERE document_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                _vec_literal(query_vec),
                document_id,
                k,
            )
            return [dict(r) for r in rows]
    finally:
        await conn.close()


async def compare_scope(
    probe_vecs: list[list[float]],
    convenio_id: int | None,
    authority_levels: list[str],
    statuses: list[str],
    as_of_date,
    exclude_document_ids: list[int],
    candidate_document_ids: list[int],
    k: int,
) -> list[list[dict]]:
    """Sprint 7d (ADR-0024): rank a scope's chunks against N probe vectors.

    READ-ONLY (SELECT only) and ADDITIVE — distinct from `retrieve()` so the
    employee answer loop's primitive is provably untouched, and distinct from
    `retrieve_by_document()` because the candidate set here is a SCOPE (a
    convenio + an authority band), not one document.

    THE LOAD-BEARING DIFFERENCE from `retrieve()`: `authority_level` is filtered
    **in the SQL**, before the ORDER BY. That makes a threshold decision on the
    top score **k-independent** — the best eligible chunk is rank 1 of an
    exactly-filtered, exactly-ordered set, so no choice of `k` can hide it. The
    publish fence is a safety gate; filtering authority client-side AFTER a
    top-k would let unrelated same-convenio chunks (e.g. other published
    rulings) crowd out the one overlapping official-convenio passage and make
    the fence silently report "no conflict" — a fail-open. Hence this endpoint.

    `candidate_document_ids` (when non-empty) pins the candidate set to an EXACT
    list of documents — also in the SQL, preserving k-independence. hr-backend
    uses it to make the candidate set exactly what the `documents` registry (the
    system of record) says is eligible, because the denormalized scope columns on
    `document_chunks` are only refreshed on re-embed and can be stale after a
    lifecycle edit. `statuses = []` and `as_of_date = None` then mean "do not
    filter on the chunk's (possibly stale) copy of the scope" — deliberate, not
    an oversight.

    `national_law` stays scope-global exactly as in `retrieve()` (the Estatuto
    baseline is not convenio-scoped); it is only reachable when the caller asks
    for it in `authority_levels`.

    Returns one ranked list per probe, in probe order. Same forced flat scan for
    exactness/determinism.
    """
    if not probe_vecs:
        return []

    conn = await _connect()
    try:
        async with conn.transaction():
            # Force exact (flat) scan — identical posture to retrieve(): the
            # scope+authority prefilter is applied exactly, and the top-k is exact.
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL enable_bitmapscan = off")
            results: list[list[dict]] = []
            for vec in probe_vecs:
                rows = await conn.fetch(
                    """
                    SELECT id, document_id, chunk_index, page_from, page_to, content,
                           retrieval_status, authority_level, convenio_id,
                           (embedding <=> $1::vector) AS distance
                    FROM document_chunks
                    WHERE authority_level = ANY($2::varchar[])
                      AND ( $3::bigint IS NULL
                            OR convenio_id = $3
                            OR authority_level = 'national_law' )
                      AND (cardinality($4::varchar[]) = 0 OR retrieval_status = ANY($4::varchar[]))
                      AND ($5::date IS NULL OR validity_start IS NULL OR validity_start <= $5::date)
                      AND ($5::date IS NULL OR validity_end IS NULL OR validity_end >= $5::date)
                      AND NOT (document_id = ANY($6::bigint[]))
                      AND (cardinality($7::bigint[]) = 0 OR document_id = ANY($7::bigint[]))
                    ORDER BY embedding <=> $1::vector
                    LIMIT $8
                    """,
                    _vec_literal(vec),
                    authority_levels,
                    convenio_id,
                    statuses,
                    as_of_date,
                    exclude_document_ids,
                    candidate_document_ids,
                    k,
                )
                results.append([dict(r) for r in rows])
            return results
    finally:
        await conn.close()


async def count_eligible_in_scope(
    convenio_id: int | None,
    authority_levels: list[str],
    statuses: list[str],
    as_of_date,
    exclude_document_ids: list[int],
    candidate_document_ids: list[int],
) -> int:
    """Exact count of the /compare-scope eligible set — the SAME WHERE as
    `compare_scope` minus the vector. Lets hr-backend distinguish "compared
    against N chunks and found nothing close" from "there was nothing to compare
    against at all" (the second is not evidence of no conflict)."""
    conn = await _connect()
    try:
        return await conn.fetchval(
            """
            SELECT count(*) FROM document_chunks
            WHERE authority_level = ANY($1::varchar[])
              AND ( $2::bigint IS NULL
                    OR convenio_id = $2
                    OR authority_level = 'national_law' )
              AND (cardinality($3::varchar[]) = 0 OR retrieval_status = ANY($3::varchar[]))
              AND ($4::date IS NULL OR validity_start IS NULL OR validity_start <= $4::date)
              AND ($4::date IS NULL OR validity_end IS NULL OR validity_end >= $4::date)
              AND NOT (document_id = ANY($5::bigint[]))
              AND (cardinality($6::bigint[]) = 0 OR document_id = ANY($6::bigint[]))
            """,
            authority_levels,
            convenio_id,
            statuses,
            as_of_date,
            exclude_document_ids,
            candidate_document_ids,
        )
    finally:
        await conn.close()


async def chunk_texts_for_document(document_id: int, limit: int) -> list[dict]:
    """Sprint 7d: the probe side of a document↔document comparison (the §8.5
    reverse re-check and the succession proposal use a document's own chunk
    texts as probes). READ-ONLY; ordered by chunk_index so the probe order is
    deterministic and reproducible in an audit."""
    conn = await _connect()
    try:
        rows = await conn.fetch(
            """
            SELECT id, chunk_index, page_from, page_to, content
            FROM document_chunks
            WHERE document_id = $1
            ORDER BY chunk_index
            LIMIT $2
            """,
            document_id,
            limit,
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
