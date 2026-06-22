"""Batch chunk-stats inspector for the Correction-01 regression guard. Loads the
model once and, for each "id:storage_key" arg, prints chunk counts + stats and a
short sample of any eu-tagged chunk (so we can confirm eu is real Basque, not
mis-split Spanish). No DB writes. Usage:

    python scripts/regress_chunks.py 46:documents/<uuid>/original.pdf 2:documents/<uuid>/original.pdf
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/app")

from app.pipeline import build_chunks  # noqa: E402
from app.storage import get_object_bytes  # noqa: E402


def main() -> int:
    for arg in sys.argv[1:]:
        label, key = arg.split(":", 1)
        pdf_bytes = get_object_bytes(key)
        chunks, stats = build_chunks(pdf_bytes)
        es = [c for c in chunks if c.get("language") == "es"]
        eu = [c for c in chunks if c.get("language") == "eu"]
        print(f"\n===== doc {label} =====")
        print(f"chunks: total={len(chunks)} es={len(es)} eu={len(eu)}")
        print(f"  page_count={stats['page_count']} two_col={stats['two_column_pages']} "
              f"single={stats['single_column_pages']} tabular={stats.get('tabular_pages')} "
              f"blocks_total={stats.get('blocks_total')} kept={stats.get('column_blocks_kept')} "
              f"furniture={stats['furniture_blocks_stripped']}")
        print(f"  pages_not_cleanly_split={stats['pages_not_cleanly_split']}")
        print(f"  repeating_furniture_lines={stats['repeating_furniture_lines']}")
        for i, c in enumerate(eu[:3]):
            snippet = c["content"][:140].replace("\n", " ")
            print(f"  eu#{i} p{c['page_from']}-{c['page_to']}: {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
