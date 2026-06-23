"""Sprint 2c regression + cap-recompute harness (NO DB writes).

For each `id:storage_key` arg, run the real pipeline (extract_columns → the new
article-boundary chunker) with the REAL BGE-M3 tokenizer and print:
  - new chunk counts (es/eu) + extraction stats (blocks/kept/furniture/flags),
  - the per-article token distribution (BEFORE the size-fallback sub-split), so
    the cap can be locked on real data,
  - a sample eu chunk (confirm genuine Basque, not mis-split Spanish).

Usage:
    python scripts/rechunk_survey.py 95:documents/<uuid>/original.pdf [more...]
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/app")

from app.chunking.chunker import _find_anchors, chunk_stream  # noqa: E402
from app.chunking.extract_columns import extract_language_streams  # noqa: E402
from app.embeddings import count_tokens  # noqa: E402
from app.storage import get_object_bytes  # noqa: E402


def _article_token_lengths(units: list[tuple[int, str]]) -> list[int]:
    """Token length of each article SEGMENT (pre sub-split) in one stream."""
    sep = "\n\n"
    text = sep.join(t for _, t in units)
    if not text.strip():
        return []
    anchors = _find_anchors(text)
    if not anchors:
        return []
    lens: list[int] = []
    bounds = anchors + [len(text)]
    for i in range(len(anchors)):
        seg = text[bounds[i] : bounds[i + 1]].strip()
        if seg:
            lens.append(count_tokens(seg))
    return lens


def main() -> int:
    corpus_art_lens: list[int] = []
    print("id      es  eu  | blocks kept furn | not_clean | new_total")
    for arg in sys.argv[1:]:
        label, key = arg.split(":", 1)
        pdf_bytes = get_object_bytes(key)
        extracted = extract_language_streams(pdf_bytes)
        streams = extracted["streams"]
        st = extracted["stats"]

        es_chunks = chunk_stream(streams.get("es", []), count_tokens)
        eu_chunks = chunk_stream(streams.get("eu", []), count_tokens)
        total = len(es_chunks) + len(eu_chunks)

        art_lens = _article_token_lengths(streams.get("es", [])) + _article_token_lengths(
            streams.get("eu", [])
        )
        corpus_art_lens += art_lens

        print(
            f"[{label:>3}] {len(es_chunks):3d} {len(eu_chunks):3d} | "
            f"{st.get('blocks_total'):6d} {st.get('column_blocks_kept'):4d} "
            f"{st.get('furniture_blocks_stripped'):4d} | "
            f"{str(st.get('pages_not_cleanly_split'))[:18]:18s} | {total}"
        )
        if art_lens:
            sl = sorted(art_lens)
            print(
                f"        articles={len(art_lens)} med={sl[len(sl)//2]} "
                f"max={sl[-1]} >512={sum(t>512 for t in art_lens)} "
                f">800={sum(t>800 for t in art_lens)} >1024={sum(t>1024 for t in art_lens)}"
            )
        for i, c in enumerate(eu_chunks[:1]):
            snip = c["content"][:90].replace("\n", " ")
            print(f"        eu#{i} p{c['page_from']}-{c['page_to']} tok={c['token_count']}: {snip}")

    if corpus_art_lens:
        sl = sorted(corpus_art_lens)
        n = len(sl)
        print(f"\n=== CORPUS article-length distribution (real BGE-M3 tokenizer, {n} articles) ===")
        print(f"  median={sl[n//2]} p90={sl[int(n*.9)]} p95={sl[int(n*.95)]} p99={sl[int(n*.99)]} max={sl[-1]}")
        for thr in (400, 512, 640, 800, 1024, 1200):
            c = sum(t > thr for t in sl)
            print(f"  > {thr:4d} tok: {c:3d} ({100*c/n:.1f}%)")
        band = sum(800 < t <= 1024 for t in sl)
        print(f"  800–1024 band: {band} ({100*band/n:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
