"""Blast-radius comparison for the Sprint 10a chunker change (build step 3, CP-1).

F1 (TOC guard + index-line strip) and F2 (sentence-initial headers) are changes
to the SHARED detector, so they apply to every prose document, not just the
Estatuto. Sprint 2c held the Estatuto out precisely because the convenio-vs-
baseline balance is the most sensitive axis in the system; the mirror-image risk
here is that a fix aimed at the Estatuto silently re-chunks the convenios.

This script chunks each given PDF with the CURRENT working-tree chunker and with
the chunker as committed on `main` (fetched via `git show`), and prints a
per-document diff: chunk counts, distinct articles detected, and any article
that gained or lost its own chunk. No DB writes, no S3 writes, no embedding.

Usage (from hr-ai/):
    python scripts/rechunk_blast_radius.py /tmp/hr10a/doc75-estatuto-julio2025.pdf \
                                           /tmp/hr10a/conv-13.pdf ...
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chunking import chunker as new_chunker  # noqa: E402
from app.chunking.extract_columns import extract_language_streams  # noqa: E402

BASELINE_REF = os.environ.get("BASELINE_REF", "main")


def load_baseline_chunker():
    """Import `app/chunking/chunker.py` as it exists on BASELINE_REF."""
    src = subprocess.check_output(
        ["git", "show", f"{BASELINE_REF}:app/chunking/chunker.py"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        text=True,
    )
    # The module does a relative `from ..config import settings`; rewrite to the
    # absolute package path so it can be loaded standalone.
    src = src.replace("from ..config import settings", "from app.config import settings")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    spec = importlib.util.spec_from_file_location("baseline_chunker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokens(s: str) -> int:
    return len(s.split())


_TOC = re.compile(r"\.{6,}\s*\d")


def articles(chunks: list[dict]) -> dict[int, bool]:
    """article number → is this chunk REAL article text (True) or an index
    decoy (False)?

    The distinction is the whole point of the comparison. On `main` the
    Estatuto's index produced chunks that open with `Artículo N.` and contain
    nothing but dot leaders — a naive "which article numbers head a chunk?" diff
    scores REMOVING one of those as a lost article, which is backwards. A chunk
    is a decoy when the majority of its lines are index entries."""
    out: dict[int, bool] = {}
    for c in chunks:
        body = c["content"].lstrip()
        m = re.match(r"Art[íi]culo\s+(\d{1,3})", body)
        if not m:
            continue
        lines = [ln for ln in c["content"].splitlines() if ln.strip()]
        is_real = not lines or (sum(1 for ln in lines if _TOC.search(ln)) / len(lines)) < 0.5
        n = int(m.group(1))
        out[n] = out.get(n, False) or is_real
    return out


def main() -> int:
    baseline = load_baseline_chunker()
    paths = sys.argv[1:]
    if not paths:
        print("usage: rechunk_blast_radius.py <pdf> [<pdf>...]")
        return 2

    print(f"baseline = {BASELINE_REF}\n")
    header = f"{'document':<34} {'old':>5} {'new':>5} {'Δ':>5} {'oldArt':>7} {'newArt':>7}"
    print(header)
    print("-" * len(header))

    regressions: list[str] = []
    for path in paths:
        name = os.path.basename(path)
        pdf = open(path, "rb").read()
        streams = extract_language_streams(pdf)
        s = streams["streams"] if "streams" in streams else streams
        old_n = new_n = 0
        old_map: dict[int, bool] = {}
        new_map: dict[int, bool] = {}
        for lang, units in s.items():
            if not units:
                continue
            oc = baseline.chunk_stream(units, tokens)
            nc = new_chunker.chunk_stream(units, tokens)
            old_n += len(oc)
            new_n += len(nc)
            for k, v in articles(oc).items():
                old_map[k] = old_map.get(k, False) or v
            for k, v in articles(nc).items():
                new_map[k] = new_map.get(k, False) or v

        old_real = {k for k, v in old_map.items() if v}
        new_real = {k for k, v in new_map.items() if v}
        lost_real = sorted(old_real - new_real)
        gained_real = sorted(new_real - old_real)
        dropped_decoys = sorted(set(old_map) - set(new_map) - old_real)

        print(f"{name:<34} {old_n:>5} {new_n:>5} {new_n - old_n:>+5} {len(old_real):>7} {len(new_real):>7}")
        if gained_real:
            print(f"    + real articles that gained an own chunk: {gained_real}")
        if dropped_decoys:
            print(f"    · index decoys removed (were never article text): {dropped_decoys}")
        if lost_real:
            print(f"    - REAL articles that LOST their own chunk: {lost_real}")
            regressions.append(f"{name}: lost {lost_real}")

    print()
    if regressions:
        print("REGRESSION — a real article that had its own chunk no longer does:")
        for r in regressions:
            print(f"  - {r}")
        return 1
    print("No real article lost its own chunk in any document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
