"""OCR fallback orchestration for text-less pages (Sprint 7e, ADR-0026).

Mirrors `extract.py`/`salary.py`/`read_structured.py`'s existing shape: a
top-level module `main.py`'s endpoint delegates to, which itself calls the
pluggable LLM provider (`app/providers/`) for the actual vision call. hr-ai
READS the already-rendered page image from S3 and RETURNS a result; the only
WRITE here is the S3 sidecar (hr-ai's existing S3-write privilege, same as the
page image itself — ADR-0010) — hr-ai still never touches the database.

Reused by exactly one endpoint, `POST /ocr-page` (main.py), called once per
page by hr-backend — either from a queued job (ingest-time fallback) or a
synchronous CLI loop (the Step-3 backfill command); see review.md §2.1 for why
that split exists and why this one function serves both callers identically.
"""

from __future__ import annotations

import json
import re

import fitz  # PyMuPDF

from .chunking.extract_columns import _es_ratio
from .providers import OcrPageResult, ProviderConfig, get_provider
from .storage import get_object_bytes, put_object_bytes

# A real transcription is overwhelmingly letters/digits/common punctuation/
# whitespace; OCR noise glyphs (accented-Latin lookalikes, control-ish
# characters, stray symbols) are what a "garbage ratio" is measuring. Spanish
# and Basque both use these accented letters routinely, so they are NOT garbage.
_CLEAN_CHAR = re.compile(
    r"[a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑàèìòùçÇ\s.,;:()\-–—'\"«»%€/ºª¿?¡!]"
)

# `_es_ratio`'s own documented floor ("Spanish prose runs ~0.12-0.20") — used to
# scale signal 1 to a 0..1 score without inventing a second threshold.
_ES_RATIO_FLOOR = 0.12
# A real transcribed body page is never anywhere near this sparse relative to
# its own rendered pixel area (empirically well above the 5 eval fixtures'
# floor, incl. the low-quality/marginalia fixture); a near-blank OCR result is
# a real, distinct low-quality signal worth surfacing, not a false alarm.
_CHARS_PER_PIXEL2_FLOOR = 0.00006


def ocr_sidecar_key(document_uuid: str, page_number: int) -> str:
    """The S3 key `extract_language_streams` probes for a zero-native-block page
    (review.md §2.3, Option B). Written once, here, at OCR time."""
    return f"documents/{document_uuid}/ocr/{page_number:04d}.json"


def render_page_text(
    article_headers: list[str], columns: list[dict], table_rows: list[list]
) -> str:
    """Plain reading-order text for `document_pages.text` (review.md §2.2).

    Headers MUST be included here, not just in the sidecar: under the pinned
    table contract a table page's title lives ONLY in `article_headers` — if
    this function dropped them, a table page's title would vanish from
    `document_pages.text` entirely, breaking both the viewer and the 7a
    tagger's `page_text` read (`TagProposalService::propose()` concatenates
    `document_pages.text` verbatim).
    """
    parts = list(article_headers)
    parts.extend(c.get("text", "") for c in columns if c.get("text"))
    if table_rows:
        parts.append("\n".join(" | ".join(str(cell) for cell in row) for row in table_rows))
    return "\n\n".join(p for p in parts if p and p.strip())


def _quality_score(text: str, pixel_area: float) -> tuple[float, dict]:
    """Deterministic per-page quality score (§2.5) — no second LLM call. Three
    signals, averaged: function-word density (reused `_es_ratio`, unmodified),
    a garbage-character ratio, and text-length-vs-page-area."""
    stripped = text.strip()
    if not stripped:
        return 0.0, {"reason": "empty transcription"}

    density = _es_ratio(stripped)
    density_score = min(1.0, density / _ES_RATIO_FLOOR)

    clean = len(_CLEAN_CHAR.findall(stripped))
    garbage_ratio = 1.0 - (clean / len(stripped))
    garbage_score = max(0.0, 1.0 - min(1.0, garbage_ratio / 0.05))

    chars_per_area = len(stripped) / pixel_area if pixel_area > 0 else 0.0
    density_area_score = min(1.0, chars_per_area / _CHARS_PER_PIXEL2_FLOOR)

    scores = {
        "function_word_density": round(density_score, 3),
        "garbage_char_ratio": round(garbage_score, 3),
        "text_length_vs_page_area": round(density_area_score, 3),
    }
    quality = round(sum(scores.values()) / len(scores), 3)
    weakest = min(scores, key=lambda k: scores[k])
    return quality, {**scores, "weakest_signal": weakest}


def ocr_page(
    document_uuid: str,
    page_number: int,
    image_key: str,
    provider_api_key: str,
    provider_config: ProviderConfig,
) -> dict:
    """OCR one page (reusing its already-rendered image), write the S3
    sidecar, and return the envelope `main.py`'s `/ocr-page` hands back to
    hr-backend. Never re-renders the page (the image at `image_key` was
    already produced by `extract_pdf` at ingest — ADR-0010)."""
    image_bytes = get_object_bytes(image_key)
    pixmap = fitz.Pixmap(image_bytes)
    pixel_area = float(pixmap.width * pixmap.height)

    provider = get_provider(provider_config.provider)
    result: OcrPageResult = provider.ocr_page(image_bytes, provider_api_key, provider_config)

    if result.layout == "parse_error":
        return {
            "error": "provider_error",
            "detail": "ocr transcription unparseable",
            "trace_fragment": result.trace_fragment,
        }

    text = render_page_text(result.article_headers, result.columns, result.table_rows)
    quality, quality_notes = _quality_score(text, pixel_area)

    sidecar = {
        "layout": result.layout,
        "columns": result.columns,
        "table_rows": result.table_rows,
        "article_headers": result.article_headers,
    }
    put_object_bytes(
        ocr_sidecar_key(document_uuid, page_number),
        json.dumps(sidecar, ensure_ascii=False).encode("utf-8"),
        "application/json",
    )

    return {
        "text": text,
        "layout": result.layout,
        "bilingual": result.bilingual,
        "quality": quality,
        "quality_notes": quality_notes,
        "cost_usd": result.trace_fragment.get("cost_usd"),
        "sec_per_page": result.trace_fragment.get("sec_per_page"),
        "engine": provider_config.model,
    }
