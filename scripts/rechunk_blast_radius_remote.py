"""Full-corpus blast-radius comparison, run INSIDE the hr-ai container (Sprint 10a, CP-1).

Same question as `rechunk_blast_radius.py` — does the F1/F2 detector change
alter any document other than the Estatuto? — but asked of the WHOLE prose
corpus rather than a hand-picked sample, and asked where the documents already
are. Nothing is copied out of AWS, nothing is written: no DB, no S3, no
embedding, no re-chunk. It reads each PDF, runs both chunkers in memory, and
prints a table.

Both chunker versions are supplied as files so the container needs no git
checkout: `--baseline` is `app/chunking/chunker.py` as committed on `main`,
`--candidate` is the working-tree version. Neither is installed; both are
imported from a path, so the running service is untouched.

Usage (from the staging host):
    docker compose -f docker-compose.staging.yml cp baseline_chunker.py hr-ai:/tmp/
    docker compose -f docker-compose.staging.yml cp candidate_chunker.py hr-ai:/tmp/
    docker compose -f docker-compose.staging.yml cp rechunk_blast_radius_remote.py hr-ai:/tmp/
    docker compose -f docker-compose.staging.yml exec -T hr-ai python /tmp/rechunk_blast_radius_remote.py \
        --baseline /tmp/baseline_chunker.py --candidate /tmp/candidate_chunker.py \
        13:<key> 18:<key> ...
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys

sys.path.insert(0, "/app")

from app.chunking.extract_columns import extract_language_streams  # noqa: E402
from app.storage import get_object_bytes  # noqa: E402

_TOC = re.compile(r"\.{6,}\s*\d")


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokens(s: str) -> int:
    """Whitespace word count. The detector under test is positional and lexical;
    tokenization affects only where an OVERSIZED article sub-splits, which is
    identical between the two versions because neither touches the cap. Using a
    stub keeps this off the shared BGE-M3 model the live service is using."""
    return len(s.split())


def articles(chunks: list[dict]) -> dict[int, bool]:
    """article number → is this REAL article text, or an index decoy?"""
    out: dict[int, bool] = {}
    for c in chunks:
        m = re.match(r"(?:ART[ÍI]CULO|Art[íi]culo|ART?\.?)\s+(\d{1,3})", c["content"].lstrip(), re.IGNORECASE)
        if not m:
            continue
        lines = [ln for ln in c["content"].splitlines() if ln.strip()]
        is_real = not lines or (sum(1 for ln in lines if _TOC.search(ln)) / len(lines)) < 0.5
        n = int(m.group(1))
        out[n] = out.get(n, False) or is_real
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("docs", nargs="+", help="id:storage_key pairs")
    args = ap.parse_args()

    base = load(args.baseline, "baseline_chunker")
    cand = load(args.candidate, "candidate_chunker")

    header = f"{'doc':>5} {'old':>5} {'new':>5} {'Δ':>5} {'oldArt':>7} {'newArt':>7}  notes"
    print(header)
    print("-" * (len(header) + 20))

    regressions: list[str] = []
    changed = 0
    for spec in args.docs:
        label, key = spec.split(":", 1)
        try:
            pdf = get_object_bytes(key)
        except Exception as exc:  # noqa: BLE001
            print(f"{label:>5}  SKIP  could not read {key}: {exc}")
            continue

        streams = extract_language_streams(pdf)
        s = streams["streams"] if "streams" in streams else streams

        old_n = new_n = 0
        old_map: dict[int, bool] = {}
        new_map: dict[int, bool] = {}
        for units in s.values():
            if not units:
                continue
            oc = base.chunk_stream(units, tokens)
            nc = cand.chunk_stream(units, tokens)
            old_n += len(oc)
            new_n += len(nc)
            for k, v in articles(oc).items():
                old_map[k] = old_map.get(k, False) or v
            for k, v in articles(nc).items():
                new_map[k] = new_map.get(k, False) or v

        old_real = {k for k, v in old_map.items() if v}
        new_real = {k for k, v in new_map.items() if v}
        lost = sorted(old_real - new_real)
        gained = sorted(new_real - old_real)
        decoys = sorted(set(old_map) - set(new_map) - old_real)

        notes = []
        if gained:
            notes.append(f"+real {gained}")
        if decoys:
            notes.append(f"-decoy {decoys}")
        if lost:
            notes.append(f"LOST REAL {lost}")
            regressions.append(f"doc {label}: lost {lost}")
        if old_n != new_n:
            changed += 1

        print(f"{label:>5} {old_n:>5} {new_n:>5} {new_n - old_n:>+5} "
              f"{len(old_real):>7} {len(new_real):>7}  {'; '.join(notes)}")

    print(f"\ndocuments whose chunk count changed: {changed} of {len(args.docs)}")
    if regressions:
        print("\nREGRESSION — a real article lost its own chunk:")
        for r in regressions:
            print(f"  - {r}")
        return 1
    print("No real article lost its own chunk in any document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
