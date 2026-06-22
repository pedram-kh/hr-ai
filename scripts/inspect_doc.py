"""Ad-hoc inspector (Sprint 2a eyes-on): for a given storage key, print the
pipeline's eu-tagged chunks in full and the raw PyMuPDF text of named pages, so a
reviewer can confirm eu chunks aren't mis-split Spanish and that
`pages_not_cleanly_split` are genuine non-prose. Usage:

    python scripts/inspect_doc.py <storage_key> <page,page,...>
"""

from __future__ import annotations

import sys

import fitz

sys.path.insert(0, "/app")

from app.pipeline import build_chunks  # noqa: E402
from app.storage import get_object_bytes  # noqa: E402


def main() -> int:
    storage_key = sys.argv[1]
    pages = [int(p) for p in sys.argv[2].split(",")] if len(sys.argv) > 2 else []

    pdf_bytes = get_object_bytes(storage_key)
    chunks, stats = build_chunks(pdf_bytes)

    es = [c for c in chunks if c.get("language") == "es"]
    eu = [c for c in chunks if c.get("language") == "eu"]
    print(f"== {storage_key}")
    print(f"chunks: es={len(es)} eu={len(eu)}; stats={stats}\n")

    print("================ eu-tagged chunks (FULL text) ================")
    for i, c in enumerate(eu):
        print(f"--- eu #{i} p{c.get('page_from')}-{c.get('page_to')} "
              f"tokens={c.get('token_count')} ---")
        print(c["content"])
        print()

    print("================ raw PyMuPDF text of named pages ================")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for p in pages:
            page = doc.load_page(p - 1)
            pw = float(page.rect.width)
            txt = page.get_text().strip()
            print(f"\n--- page {p} (width={pw:.0f}pt) — raw text ---")
            print(txt if txt else "(no extractable text — likely image/scanned)")
    finally:
        doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
