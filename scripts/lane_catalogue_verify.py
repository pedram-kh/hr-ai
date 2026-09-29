"""Sprint 13 CP-1 — verify candidate general-lane catalogue pages with REAL
fetches through the lane's own fetcher (`fetch_source`: allowlist, SSRF guard,
certifi + bundled-intermediate TLS, non-2xx = error, topic-windowed excerpt,
no-topic-match = no material). READ-ONLY; nothing is written anywhere.

For each candidate it prints one `ROW {json}` line: status, bytes, full-page
text length, windows, excerpt length, error, matched topic terms, the number of
topic-term occurrences in the excerpt, and CONTEXT snippets (±200 chars around
the first two occurrences) so a human can judge whether the excerpt really
explains the topic rather than merely containing the word (a page title or
breadcrumb also 'contains the word').

Candidates come from the CANDIDATES_B64 env var (base64 JSON list of
{id, url, topics, probe?, expect?}) — `probe` is the question phrase used as
the priority topic; `expect` is a list of phrases the excerpt should contain
(reported as `expect_hits`, accent/case-insensitive) — e.g. "Artículo 169". Run inside the hr-ai container (PYTHONPATH=/app) or locally
from hr-ai/:

    CANDIDATES_B64=$(base64 < candidates.json) python scripts/lane_catalogue_verify.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.general_lane import _fold, fetch_source, matching_topics  # noqa: E402

ALLOWED = ["boe.es", "mites.gob.es", "seg-social.es", "sepe.es"]


def _contexts(text: str, terms: list[str], n: int = 2, radius: int = 200) -> tuple[int, list[str]]:
    folded = _fold(text)
    spans: list[int] = []
    for t in terms:
        ft = _fold(t)
        spans += [m.start() for m in re.finditer(r"(?<![a-z0-9])" + re.escape(ft), folded)]
    spans.sort()
    out, last = [], -10**9
    for s in spans:
        if s - last < 2 * radius:
            continue
        out.append(text[max(0, s - radius): s + radius].replace("\n", " "))
        last = s
        if len(out) >= n:
            break
    return len(spans), out


def main() -> int:
    cands = json.loads(base64.b64decode(os.environ["CANDIDATES_B64"]).decode())
    for c in cands:
        entry = {"topics": c["topics"]}
        priority = matching_topics(c.get("probe", ""), entry) if c.get("probe") else []
        r = fetch_source(c["url"], ALLOWED, topics=c["topics"], priority_topics=priority)
        occurrences, contexts = _contexts(r.text, c["topics"])
        print("ROW " + json.dumps({
            "id": c["id"],
            "url": c["url"],
            "final_url": r.url,
            "status": r.status,
            "bytes": r.bytes,
            "ms": r.ms,
            "error": r.error,
            "text_chars": r.text_chars,
            "windows": r.windows,
            "excerpt_chars": len(r.text),
            "matched_terms": r.matched_terms,
            "expect_hits": {e: (_fold(e) in _fold(r.text)) for e in c.get("expect", [])},
            "occurrences": occurrences,
            "contexts": contexts,
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
