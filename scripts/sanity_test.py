"""BGE-M3 / vector(1024) retrieval sanity test — FIRST hr-ai task, go/no-go.

Embeds a small mixed set of REAL chunks — Euskara + Spanish from the Gipuzkoa
bilingual convenio and Spanish from the Estatuto — and verifies that
same-language queries retrieve the right chunk. Confirms BGE-M3/1024 retrieves
sensibly on OUR content before any bulk embed (ADR-0006). If it fails badly:
STOP and raise in review.md — do NOT swap models or re-dimension.

Reads the originals from S3 (by the storage_path recorded at ingest), so it also
smoke-tests the column-aware extractor + de-spacer on real input. No DB writes.

Run (inside the hr-ai container):
    docker exec hr_ai python scripts/sanity_test.py
"""

from __future__ import annotations

import asyncio
import sys

import asyncpg

sys.path.insert(0, "/app")

from app.config import settings  # noqa: E402
from app.embeddings import embed_texts  # noqa: E402
from app.pipeline import build_chunks  # noqa: E402
from app.storage import get_object_bytes  # noqa: E402


async def _find(con, like: str) -> tuple[int, str, str] | None:
    row = await con.fetchrow(
        "SELECT id, source_filename, storage_path FROM documents "
        "WHERE source_filename ILIKE $1 ORDER BY id LIMIT 1",
        like,
    )
    return (row["id"], row["source_filename"], row["storage_path"]) if row else None


def _cos_rank(query_vec, corpus_vecs):
    import math

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    sims = [dot(query_vec, v) for v in corpus_vecs]  # vecs are unit-normalized
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return order, sims


async def main() -> int:
    con = await asyncpg.connect(dsn=settings.database_url)
    try:
        gip = await _find(con, "%Limpieza Gipuzkoa%")
        est = await _find(con, "%ESTATUTO%")
    finally:
        await con.close()

    if gip is None or est is None:
        print("FAIL: could not find the Gipuzkoa convenio and/or the Estatuto in documents.")
        print(f"  gipuzkoa={gip} estatuto={est}")
        return 2

    print(f"Gipuzkoa: {gip[1]}  ({gip[2]})")
    print(f"Estatuto: {est[1]}  ({est[2]})")

    gip_chunks, gip_stats = build_chunks(get_object_bytes(gip[2]))
    est_chunks, est_stats = build_chunks(get_object_bytes(est[2]))

    print("\n--- extraction stats (eyes-on gate a) ---")
    print(f"Gipuzkoa: {gip_stats}")
    print(f"Estatuto: {est_stats}")

    gip_es = [c for c in gip_chunks if c["language"] == "es"]
    gip_eu = [c for c in gip_chunks if c["language"] == "eu"]
    print(f"\nGipuzkoa chunks: es={len(gip_es)} eu={len(gip_eu)}; Estatuto chunks: es={len(est_chunks)}")

    # Sample a small mixed corpus.
    sample = (gip_es[:6] + gip_eu[:6] + est_chunks[:6])
    if len(sample) < 6:
        print("FAIL: too few chunks produced to test.")
        return 2
    labels = (
        [f"GIP-es#{c['chunk_index'] if 'chunk_index' in c else i}" for i, c in enumerate(gip_es[:6])]
        + [f"GIP-eu#{i}" for i in range(len(gip_eu[:6]))]
        + [f"EST-es#{i}" for i in range(len(est_chunks[:6]))]
    )
    texts = [c["content"] for c in sample]

    print("\n--- sample chunk boundaries (eyeball: clean lang separation, de-spacing, sensible splits) ---")
    for lbl, c in list(zip(labels, sample, strict=True))[:6]:
        preview = " ".join(c["content"].split())[:160]
        print(f"  {lbl} (p{c['page_from']}-{c['page_to']}, {c['token_count']}tok): {preview}…")

    corpus_vecs = embed_texts(texts)

    # Self-retrieval: query = the chunk's leading text; the chunk must rank #1.
    print("\n--- same-language self-retrieval (query = chunk's own lead text) ---")
    queries = [" ".join(t.split())[:120] for t in texts]
    qvecs = embed_texts(queries)
    hits = 0
    for i, qv in enumerate(qvecs):
        order, sims = _cos_rank(qv, corpus_vecs)
        ok = order[0] == i
        hits += ok
        if i < 6 or not ok:
            print(f"  {labels[i]}: top={labels[order[0]]} sim={sims[order[0]]:.3f} {'OK' if ok else 'MISS'}")
    acc = hits / len(qvecs)
    print(f"self-retrieval accuracy: {hits}/{len(qvecs)} = {acc:.0%}")

    # Cross-language alignment (allowed, never filtered): an eu chunk's nearest
    # es neighbour should be semantically related, but a same-language query
    # still wins (above).
    print("\n--- cross-language alignment (no language filter; eu↔es space is shared) ---")
    if gip_eu and gip_es:
        eu_vec = embed_texts([gip_eu[0]["content"]])[0]
        es_vecs = embed_texts([c["content"] for c in gip_es[:6]])
        order, sims = _cos_rank(eu_vec, es_vecs)
        print(f"  GIP-eu#0 nearest es chunk sim={sims[order[0]]:.3f} (>0 ⇒ multilingual space works)")

    # De-spacing A/B: a re-spaced (artifact-like) query should retrieve its chunk
    # worse than the clean de-spaced query — normalization helps, not hurts.
    print("\n--- de-spacing A/B (normalized vs artifact-like) ---")
    base = " ".join(texts[0].split())[:120]
    artifact = "".join(ch + (" " if (i % 3 == 2 and ch != " ") else "") for i, ch in enumerate(base))
    ab = embed_texts([base, artifact])
    order_clean, sims_clean = _cos_rank(ab[0], corpus_vecs)
    order_art, sims_art = _cos_rank(ab[1], corpus_vecs)
    print(f"  clean query  → top={labels[order_clean[0]]} sim={sims_clean[order_clean[0]]:.3f}")
    print(f"  artifact qry → top={labels[order_art[0]]} sim={sims_art[order_art[0]]:.3f}")

    # Verdict.
    passed = acc >= 0.85
    print("\n==================== VERDICT ====================")
    print(f"BGE-M3/1024 sanity: {'PASS' if passed else 'FAIL'} (self-retrieval {acc:.0%}, threshold 85%)")
    print("=================================================")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
