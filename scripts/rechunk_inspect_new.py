"""Show the chunks the Sprint 10a detector ADDS to one document (CP-1 evidence).

`rechunk_blast_radius_remote.py` says how many chunks a document gained; this
says what they actually are. A gain is only good news if the new chunk starts at
a real article header — the 2c §3.3 risk is precisely that a relaxed guard
"gains" chunks by splitting mid-sentence on a cross-reference.

Read-only: no DB, no S3 write, no embedding. Runs inside the hr-ai container.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

sys.path.insert(0, "/app")

from app.chunking.extract_columns import extract_language_streams  # noqa: E402
from app.storage import get_object_bytes  # noqa: E402


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokens(s: str) -> int:
    return len(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("key")
    args = ap.parse_args()

    base = load(args.baseline, "baseline_chunker")
    cand = load(args.candidate, "candidate_chunker")

    streams = extract_language_streams(get_object_bytes(args.key))
    s = streams["streams"] if "streams" in streams else streams

    for lang, units in s.items():
        if not units:
            continue
        oc = base.chunk_stream(units, tokens)
        nc = cand.chunk_stream(units, tokens)
        old_heads = {c["content"].lstrip()[:60] for c in oc}
        added = [c for c in nc if c["content"].lstrip()[:60] not in old_heads]
        removed = [c for c in oc if c["content"].lstrip()[:60] not in {x["content"].lstrip()[:60] for x in nc}]

        print(f"\n===== stream '{lang}': {len(oc)} -> {len(nc)} chunks "
              f"({len(added)} new starts, {len(removed)} gone) =====")
        print(f"\n--- NEW chunk starts (first {args.limit}) ---")
        for c in added[: args.limit]:
            print(f"  p{c['page_from']}-{c['page_to']} [{len(c['content'])}c] "
                  f"{c['content'].lstrip()[:110]!r}")
        print(f"\n--- chunk starts that DISAPPEARED (first {args.limit}) ---")
        for c in removed[: args.limit]:
            print(f"  p{c['page_from']}-{c['page_to']} [{len(c['content'])}c] "
                  f"{c['content'].lstrip()[:110]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
