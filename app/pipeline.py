"""Embed pipeline (ADR-0013): S3 original → column-aware streams → de-spaced
article chunks → BGE-M3 embeddings → write `document_chunks` directly.

hr-ai WRITES document_chunks (its one permitted table); hr-backend resolves and
PASSES each document's scope (ADR-0007), which is copied verbatim into the
denormalized scope columns. Idempotent: re-embedding replaces the document's
chunks cleanly.
"""

from __future__ import annotations

import asyncio

from .chunking.chunker import chunk_document
from .chunking.extract_columns import extract_language_streams
from .chunks_db import replace_document_chunks
from .embeddings import count_tokens, embed_texts
from .storage import get_object_bytes


def build_chunks(pdf_bytes: bytes) -> tuple[list[dict], dict]:
    """Synchronous CPU work: extract columns → chunk. Returns (chunks, stats)."""
    extracted = extract_language_streams(pdf_bytes)
    chunks = chunk_document(extracted["streams"], count_tokens)
    return chunks, extracted["stats"]


async def embed_document(
    document_id: int,
    storage_key: str,
    scope: dict,
) -> dict:
    pdf_bytes = await asyncio.to_thread(get_object_bytes, storage_key)
    chunks, stats = await asyncio.to_thread(build_chunks, pdf_bytes)

    embeddings: list[list[float]] = []
    if chunks:
        embeddings = await asyncio.to_thread(embed_texts, [c["content"] for c in chunks])

    written = await replace_document_chunks(document_id, scope, chunks, embeddings)

    return {
        "document_id": document_id,
        "chunks_written": written,
        "language_streams": {
            "es": sum(1 for c in chunks if c.get("language") == "es"),
            "eu": sum(1 for c in chunks if c.get("language") == "eu"),
        },
        "stats": stats,
    }
