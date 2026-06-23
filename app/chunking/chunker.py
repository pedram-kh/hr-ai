"""Article-boundary chunking (ADR-0013 + ADR-0017, Sprint 2c).

Sprint 2a chunked by *packing* small consecutive articles up to a token target.
That packing was the root cause of the buried-grant artifact behind 2b-2
Correction-03: a short governing article (Navarra Art. 9.º Vacaciones, "37 días
laborables") got merged into a chunk dominated by a neighbouring article (Art.
8.ºbis horas complementarias), so its grant sat at the chunk tail and embedded
weakly against a vacaciones query → ranked #15 → the Estatuto baseline reached
synthesis instead.

Sprint 2c makes each detected article header start its **own** chunk and
**removes cross-article packing** for the article path. A small article (e.g.
Vacaciones) is now its own small, topically-clean chunk that retrieves on its own
merits. Only an article that exceeds the size cap is sub-split — on a sub-clause /
paragraph / sentence boundary, never mid-sentence — and every sub-chunk carries
its `Artículo N.º <título>` header so it still resolves to (and cites as) its
article. The pre-Article-1 preamble and any anchor-less annex still use the
size-capped paragraph fallback.

Because packing no longer hides loose detection, the header **detector is now
load-bearing** and runs with three precision guards (line-anchored, case-aware,
monotonic-number) so an inline lowercase cross-reference ("…según el artículo 22
del Estatuto…") can never be mistaken for a header and spawn a spurious chunk.

This stage composes with — never replaces — the 2a extraction front-end
(`extract_columns.py`): de-spacing, repetition/margin-band furniture stripping,
positive-evidence two-column detection, the Spanish-function-word language gate,
and language tagging all run BEFORE this, and are untouched. The chunker still
receives one already-separated language stream at a time.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..config import settings

# --- Header detection -------------------------------------------------------
# Candidate patterns deliberately OVER-detect; the per-candidate guards in
# `_find_anchors` reject everything that is not a real header.

# Spelled-out ordinals — defensive only (no active prose convenio uses them
# today; survey §1.1), so a future "ARTÍCULO NOVENO" doc still splits.
_ORDINAL_WORDS = (
    r"primero|segundo|tercero|cuarto|quinto|sexto|s[eé]ptimo|octavo|noveno|"
    r"d[eé]cimo|und[eé]cimo|duod[eé]cimo|decimo\w*|vig[eé]simo\w*|trig[eé]simo\w*"
)

# es article header with a numeric id. Covers survey variants V1–V7:
#   Artículo 1.  /  Artículo 1.º  /  Art. 2.º  /  Artículo 1.—  /  Artículo 1.–
#   /  Artículo 1.-  /  ART 7.-  (Salamanca, uppercase bare ART, §7-Q6).
# `lead` is captured so the guard can test capitalization (headers are
# capitalised; inline references are lowercase `artículo`/`art.`).
_ES_NUM = re.compile(
    r"(?P<lead>ART[ÍI]CULO|Art[íi]culo|Art\.|ART\.|art\.|art[íi]culo|ART)"
    r"\s+(?P<num>\d{1,3})"
)
# es article header spelled out (defensive).
_ES_ORD = re.compile(
    rf"(?P<lead>ART[ÍI]CULO|Art[íi]culo)\s+(?P<ord>{_ORDINAL_WORDS})\b",
    re.IGNORECASE,
)
# Euskara header: `N. artikulua` (V8). Always a header — `artikulua` does not
# occur as an inline cross-reference token the way `artículo` does.
_EU = re.compile(r"(?P<num>\d{1,3})\.?\s*artikulua", re.IGNORECASE)
# Estatuto / convenio back-matter dispositions (kept as boundaries, after
# articles).
_DISP = re.compile(
    r"Disposici[óo]n\s+(?:adicional|transitoria|final|derogatoria)", re.IGNORECASE
)

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.;:])\s+")


def _build_text_and_pagemap(units: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Join (page, text) units into one string; return (text, offset→page map).

    The map is a list of (offset_start, page) sorted by offset; the page of any
    character is the page of the last entry whose offset_start <= the char.
    """
    parts: list[str] = []
    pagemap: list[tuple[int, int]] = []
    offset = 0
    sep = "\n\n"
    for page, text in units:
        pagemap.append((offset, page))
        parts.append(text)
        offset += len(text) + len(sep)
    return sep.join(parts), pagemap


def _page_for_offset(pagemap: list[tuple[int, int]], offset: int) -> int:
    page = pagemap[0][1] if pagemap else 1
    for off, pg in pagemap:
        if off <= offset:
            page = pg
        else:
            break
    return page


def _at_line_start(text: str, s: int) -> bool:
    """Guard 1 — the anchor must start a line (allowing leading indentation),
    not appear mid-sentence (kills inline `…del artículo 22…` wrapped onto a new
    line only if it is genuinely line-leading)."""
    i = s - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i < 0 or text[i] == "\n"


def _subclause_after_num(text: str, num_end: int) -> bool:
    """True when the integer is immediately followed by a decimal sub-clause
    (`9.1`) or a single-letter sub-article (`13.a`) — which must stay WITH the
    parent article, not start a new chunk. `bis` and the ordinal markers
    `.º/.ª` and the separators `.—/.–/.-` are NOT sub-clauses."""
    tail = text[num_end : num_end + 4]
    if re.match(r"\.\d", tail):  # 9.1 / 13.2 decimal sub-clause
        return True
    # 13.a / 5.b letter sub-article (a real ascii letter, not º/ª, not 'bis').
    m = re.match(r"\.([a-zA-Z])(?![a-zA-Z])", tail)
    if m and not tail.lower().startswith(".bis"):
        return True
    return False


def _find_anchors(text: str) -> list[int]:
    """Return sorted start offsets of REAL article headers in one language
    stream, after the three precision guards. Empty if the stream has no
    articles (→ caller uses the paragraph fallback)."""
    cands: list[tuple[int, str, int | None]] = []  # (start, kind, num|None)

    for m in _ES_NUM.finditer(text):
        cands.append((m.start(), "es_num", int(m.group("num"))))
    for m in _ES_ORD.finditer(text):
        cands.append((m.start(), "es_ord", None))
    for m in _EU.finditer(text):
        cands.append((m.start(), "eu", int(m.group("num"))))
    for m in _DISP.finditer(text):
        cands.append((m.start(), "disp", None))

    cands.sort(key=lambda c: c[0])

    accepted: list[int] = []
    last_num = 0
    for start, kind, num in cands:
        if not _at_line_start(text, start):
            continue

        if kind in ("es_ord", "disp"):
            # Spelled-out article / disposition: capitalised + line-anchored is
            # enough (these never collide with inline references).
            if text[start] == text[start].upper():
                accepted.append(start)
            continue

        if kind == "eu":
            # `N. artikulua` — always a header in the eu stream.
            accepted.append(start)
            if num is not None:
                last_num = max(last_num, num)
            continue

        # es_num: locate the integer end to test the sub-clause / monotonic guards.
        mnum = re.match(
            r"(?:ART[ÍI]CULO|Art[íi]culo|Art\.|ART\.|art\.|art[íi]culo|ART)\s+\d{1,3}",
            text[start : start + 24],
        )
        num_end = start + mnum.end() if mnum else start
        if _subclause_after_num(text, num_end):
            continue  # 9.1 / 13.a → keep with parent article

        # Guard 2 (case-aware): headers are capitalised; inline refs are
        # lowercase. A capitalised, line-anchored header is accepted regardless
        # of its number (tolerates the real duplicate/skip quirks — two
        # `Artículo 16`, `9→11`). A lowercase candidate (OCR-lowered header, or
        # an inline ref that happens to lead a line) is accepted ONLY if it is
        # also monotonic (Guard 3) — so `artículo 22 del ET` inside Art. 40 is
        # rejected (22 < 40), while a genuinely sequential lowercase header is
        # kept.
        is_cap = text[start] == text[start].upper()
        if is_cap:
            accepted.append(start)
            if num is not None:
                last_num = max(last_num, num)
        elif num is not None and num >= last_num:
            accepted.append(start)
            last_num = num

    # De-dup (an offset can match >1 candidate pattern) and keep order.
    return sorted(set(accepted))


def _pack_to_cap(text: str, count_tokens: Callable[[str], int], cap: int) -> list[str]:
    """Split text into pieces each <= cap tokens (paragraph → sentence → word
    granularity). Used for the preamble, anchor-less annexes, and the size
    fallback inside an oversized article — never to pack two articles together."""
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= cap:
        return [text]

    pieces: list[str] = []
    units = [p for p in _PARA.split(text) if p.strip()] or [text]
    buf = ""
    for unit in units:
        candidate = (buf + "\n\n" + unit).strip() if buf else unit.strip()
        if count_tokens(candidate) <= cap:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        if count_tokens(unit) <= cap:
            buf = unit.strip()
        else:
            # Paragraph itself too big → sentence granularity.
            sbuf = ""
            for sent in _SENT.split(unit):
                scand = (sbuf + " " + sent).strip() if sbuf else sent.strip()
                if count_tokens(scand) <= cap:
                    sbuf = scand
                elif sbuf:
                    pieces.append(sbuf)
                    sbuf = sent.strip()
                else:
                    pieces.extend(_hard_split_words(sent, count_tokens, cap))
                    sbuf = ""
            if sbuf:
                buf = sbuf
    if buf:
        pieces.append(buf)
    return pieces


def _hard_split_words(text: str, count_tokens: Callable[[str], int], cap: int) -> list[str]:
    words = text.split()
    pieces, buf = [], ""
    for w in words:
        cand = (buf + " " + w).strip() if buf else w
        if count_tokens(cand) <= cap:
            buf = cand
        elif buf:
            pieces.append(buf)
            buf = w
        else:
            pieces.append(w)  # single token over cap — unavoidable
            buf = ""
    if buf:
        pieces.append(buf)
    return pieces


def _article_header_line(seg: str) -> str:
    """First non-empty line of an article segment — its `Artículo N.º <título>`
    header, carried onto continuation sub-chunks of an oversized article."""
    for line in seg.splitlines():
        if line.strip():
            return line.strip()
    return ""


def chunk_stream(
    units: list[tuple[int, str]],
    count_tokens: Callable[[str], int],
    target: int | None = None,
    cap: int | None = None,
) -> list[dict]:
    """Chunk one language stream → list of {content, page_from, page_to, token_count}.

    One chunk per article (NO cross-article packing). The preamble and anchor-less
    streams use the size-capped paragraph fallback; an oversized article is
    sub-split on a sub-clause/paragraph/sentence boundary with its header carried."""
    target = target or settings.chunk_token_target
    cap = cap or settings.chunk_token_cap
    text, pagemap = _build_text_and_pagemap(units)
    if not text.strip():
        return []

    chunks: list[dict] = []

    def emit(content: str, start_off: int, end_off: int) -> None:
        content = content.strip()
        if not content:
            return
        chunks.append(
            {
                "content": content,
                "page_from": _page_for_offset(pagemap, start_off),
                "page_to": _page_for_offset(pagemap, max(start_off, end_off - 1)),
                "token_count": count_tokens(content),
            }
        )

    def emit_fallback(segment: str, base_off: int) -> None:
        for piece in _pack_to_cap(segment, count_tokens, target):
            rel = segment.find(piece[:40]) if piece else -1
            off = base_off + (rel if rel >= 0 else 0)
            emit(piece, off, off + len(piece))

    anchors = _find_anchors(text)

    if not anchors:
        # No article anchors (e.g. an annex / non-article doc) → paragraph fallback.
        emit_fallback(text, 0)
        return chunks

    # Preamble before Article 1 (title page, CAPÍTULO heading, recitals).
    if anchors[0] > 0 and text[: anchors[0]].strip():
        emit_fallback(text[: anchors[0]], 0)

    # One chunk per article; sub-split only an oversized article.
    for i, s in enumerate(anchors):
        e = anchors[i + 1] if i + 1 < len(anchors) else len(text)
        seg = text[s:e].strip()
        if not seg:
            continue
        if count_tokens(seg) <= cap:
            emit(seg, s, e)
            continue
        # Oversized article → sub-clause/paragraph/sentence sub-split, carrying
        # the article header onto each continuation sub-chunk so it still
        # resolves to (and cites as) its article.
        header = _article_header_line(seg)
        pieces = _pack_to_cap(seg, count_tokens, cap)
        for j, piece in enumerate(pieces):
            if j > 0 and header and not piece.startswith(header):
                piece = f"{header}\n{piece}"
            emit(piece, s, e)

    return chunks


def chunk_document(streams: dict[str, list[tuple[int, str]]], count_tokens: Callable[[str], int]) -> list[dict]:
    """Chunk both language streams and return a single deterministically-ordered
    list (by first page, then es before eu). `chunk_index` is assigned by the
    caller after embedding."""
    out: list[dict] = []
    for lang in ("es", "eu"):
        for c in chunk_stream(streams.get(lang, []), count_tokens):
            c["language"] = lang  # internal only — NOT stored on document_chunks
            out.append(c)
    out.sort(key=lambda c: (c["page_from"], 0 if c["language"] == "es" else 1))
    return out
