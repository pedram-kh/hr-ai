"""BGE-M3 embeddings (ADR-0006), self-hosted in-process.

Multilingual (Spanish + Euskara), open-weight, 1024-dim. Loaded lazily as a
process singleton so importing this module stays cheap; the (large) model is
fetched/loaded on first use. CPU is acceptable — embedding runs on the
background admin path (ingestion), never the latency-critical employee path.

Language is NEVER a filter (ADR-0006): the same model embeds `es` and `eu` into
one shared space, so cross-language retrieval is *possible* (not filtered out)
while same-language semantic matches dominate.
"""

from __future__ import annotations

import threading

from .config import settings

_model = None
_tokenizer = None
_lock = threading.Lock()


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        # Imported lazily so the service (and /health, /extract) does not pay the
        # torch import cost unless embeddings are actually used.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.embed_model_hf, device="cpu")
        _tokenizer = model.tokenizer
        _model = model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts → list of 1024-dim unit vectors (cosine-ready)."""
    if not texts:
        return []
    _load()
    vectors = _model.encode(
        texts,
        normalize_embeddings=True,  # unit vectors → cosine == dot product
        batch_size=16,
        show_progress_bar=False,
    )
    return [[float(x) for x in row] for row in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def count_tokens(text: str) -> int:
    """Token count using BGE-M3's own (XLM-RoBERTa) tokenizer — the unit the
    chunk size cap is expressed in."""
    _load()
    return len(_tokenizer.encode(text, add_special_tokens=False))


def embed_dim() -> int:
    return settings.embed_dim
