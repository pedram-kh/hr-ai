"""Predict the exact post-re-embed chunk count for ONE document, on the real
ingest path.

The blast-radius harness calls `extract_language_streams(pdf)` without a
document uuid, which skips the Sprint 7e OCR sidecar probe. For most documents
that is identical, but for any document with OCR'd pages it under-counts — doc
75 reads 212 chunks there against 235 rows in the staging DB. This script closes
that gap by going through `pipeline.build_chunks`, uuid included, which is
exactly what `/embed` runs.

Read-only: extracts, chunks and counts. Writes nothing, embeds nothing.

    python rechunk_predict_doc75.py --baseline /tmp/baseline_chunker.py \
        --candidate /tmp/candidate_chunker.py <uuid> <storage_key>
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from collections import Counter

from app.chunking.extract_columns import extract_language_streams
from app.embeddings import count_tokens
from app.storage import get_object_bytes

_ART = re.compile(r"(?:ART[ÍI]CULO|Art[íi]culo|ART?\.?)\s+(\d{1,3})", re.IGNORECASE)


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarise(chunks: list[dict]) -> tuple[int, set[int], Counter]:
    arts: set[int] = set()
    per_stream: Counter = Counter()
    for c in chunks:
        per_stream[c.get("language") or c.get("stream") or "?"] += 1
        m = _ART.match(c["content"].lstrip())
        if m:
            arts.add(int(m.group(1)))
    return len(chunks), arts, per_stream


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("uuid")
    ap.add_argument("key")
    args = ap.parse_args()

    pdf = get_object_bytes(args.key)
    extracted = extract_language_streams(pdf, args.uuid)
    print(f"extract stats: {extracted['stats']}")

    for label, path in (("baseline (main)", args.baseline), ("candidate (10a)", args.candidate)):
        mod = load(path, f"chunker_{label[:4]}")
        chunks = mod.chunk_document(extracted["streams"], count_tokens)
        total, arts, per_stream = summarise(chunks)
        print(f"\n{label}: {total} chunks  {dict(per_stream)}")
        print(f"  distinct articles: {len(arts)}")
        missing = [n for n in range(1, 93) if n not in arts]
        print(f"  articles 1-92 without an own chunk: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
