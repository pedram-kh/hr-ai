"""Sprint 7e (ADR-0026) — the sidecar-only-when-zero-blocks proof, self-contained.

hr-ai has no pytest suite (this repo's established pattern — see sanity_test.py,
rechunk_survey.py — is a standalone, directly-run verification script, not a
unit-test framework), so this follows the same convention rather than
introducing one for a single new invariant.

Proves three things about `extract_language_streams(pdf_bytes, document_uuid)`
(review.md §2.3, Option B), with a synthetic in-memory PDF (no real S3, no DB —
`app.storage.get_object_bytes` is monkeypatched to serve a fake sidecar):

  1. A page WITH native text blocks ignores an OCR sidecar even if one exists
     at that exact key (the sidecar probe never runs for it — a page whose
     furniture-stripping happened to remove every block is a DIFFERENT case
     from a scanned page and must never be re-probed here).
  2. A page with ZERO native blocks and NO sidecar contributes nothing (never
     a crash, never a KeyError) — the exact prior (pre-7e) behavior.
  3. A page with ZERO native blocks and a sidecar PRESENT gets its columns
     (es/eu-tagged) and its table_rows appended into the right stream, in the
     pinned-contract shape (headers excluded from the stream — §2.2/§2.3).

Run (inside the hr-ai container, or any env with the repo's deps installed):
    python3 scripts/ocr_sidecar_test.py
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import fitz  # PyMuPDF

# Runnable both inside the hr-ai container (repo root mounted at /app) and
# directly from a checkout (repo root = this script's grandparent directory).
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _one_page_pdf(text: str | None) -> bytes:
    """A single-page PDF with real text (`text` given) or none (scanned page,
    `text=None` — zero native blocks, same as a real scan's rawdict)."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4-ish, in points
    if text:
        page.insert_text((72, 72), text)
    return doc.tobytes()


def main() -> int:
    from app.chunking.extract_columns import extract_language_streams

    failures: list[str] = []

    # --- (1) native blocks present → sidecar ignored, even if it "exists" ---
    pdf_with_text = _one_page_pdf("Artículo 1. Este texto viene de la capa nativa del PDF.")
    sidecar_that_should_be_ignored = json.dumps(
        {"columns": [{"order": 0, "language": "es", "text": "TEXTO DEL SIDECAR — NO DEBE APARECER"}]}
    ).encode("utf-8")
    with patch("app.storage.get_object_bytes", return_value=sidecar_that_should_be_ignored) as mock_get:
        result = extract_language_streams(pdf_with_text, document_uuid="doc-with-native-text")
    flat_es = " ".join(t for _, t in result["streams"]["es"])
    if "NO DEBE APARECER" in flat_es:
        failures.append("(1) FAILED: sidecar text leaked into a page that had native blocks")
    if "capa nativa" not in flat_es:
        failures.append("(1) FAILED: native text itself went missing")
    if mock_get.called:
        failures.append("(1) FAILED: get_object_bytes (the sidecar probe) was called for a page with native blocks")
    print("(1) native-blocks-present ignores sidecar:", "FAIL" if any("(1)" in f for f in failures) else "OK")

    # --- (2) zero native blocks, no sidecar → contributes nothing, no crash ---
    pdf_scanned = _one_page_pdf(None)
    with patch("app.storage.get_object_bytes", side_effect=Exception("NoSuchKey")):
        result = extract_language_streams(pdf_scanned, document_uuid="doc-scanned-no-sidecar")
    if result["streams"]["es"] or result["streams"]["eu"]:
        failures.append("(2) FAILED: a zero-block page with no sidecar produced stream units")
    if result["stats"]["ocr_sidecar_pages_used"]:
        failures.append("(2) FAILED: ocr_sidecar_pages_used non-empty with no sidecar present")
    print("(2) zero-blocks + no sidecar → nothing, no crash:", "FAIL" if any("(2)" in f for f in failures) else "OK")

    # --- (3) zero native blocks, sidecar present → appended, pinned-contract shape ---
    sidecar = json.dumps(
        {
            "layout": "table",
            "columns": [{"order": 0, "language": "es", "text": "Plus Festivo: 3,18 €/h."}],
            "table_rows": [["Grupo", "Salario"], ["1", "21000"]],
            "article_headers": ["ANEXO I: TABLA SALARIAL"],
        }
    ).encode("utf-8")
    with patch("app.storage.get_object_bytes", return_value=sidecar):
        result = extract_language_streams(pdf_scanned, document_uuid="doc-scanned-with-sidecar")
    flat_es = " ".join(t for _, t in result["streams"]["es"])
    if "Plus Festivo" not in flat_es:
        failures.append("(3) FAILED: sidecar column text missing from the es stream")
    if "Grupo | Salario" not in flat_es or "1 | 21000" not in flat_es:
        failures.append("(3) FAILED: sidecar table_rows missing/malformed in the es stream")
    if "ANEXO I" in flat_es:
        failures.append("(3) FAILED: article_headers leaked into the stream (they must not — §2.3)")
    if 1 not in result["stats"]["ocr_sidecar_pages_used"]:
        failures.append("(3) FAILED: ocr_sidecar_pages_used did not record page 1")
    print("(3) zero-blocks + sidecar present → appended, headers excluded:", "FAIL" if any("(3)" in f for f in failures) else "OK")

    print()
    if failures:
        print(f"VERDICT: FAIL ({len(failures)} failure(s))")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("VERDICT: PASS — all 3 invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
