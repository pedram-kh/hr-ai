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

# Sprint 10a F1 — a table-of-contents / index entry: a dot leader (6+ dots) run
# followed by a page number. The Estatuto's front matter (pages 1-11) is 32 such
# chunks, 25 of which open with `Artículo N.` and are therefore INDISTINGUISHABLE
# from a real article chunk by their first line — e.g.
#   "Artículo 14. Periodo de prueba.................................. 40"
# They carry the topical keywords of the article and none of its substance. On
# the normal path real convenio chunks out-compete them; on Sprint 10a's
# national-law-ONLY fallback path there is no convenio side, so they become
# first-class decoys for exactly the questions the fallback exists to answer.
# NOT end-of-line anchored, deliberately. The Estatuto's index frequently packs
# several entries onto one extracted line —
#   "Artículo 45. Causas y efectos de la suspensión.......... 95 Artículo 46. …"
# — so an `…$` anchor matches none of them. Worse than merely missing a chunk:
# an accepted TOC anchor advances the monotonic counter (guard 3) to the highest
# article number in the index, which then rejects every later SENTENCE-initial
# body header as non-monotonic. That coupling is what made F2 look like it had
# no effect until F1 was correct.
_TOC_LINE = re.compile(r"\.{6,}\s*\d")

# Sprint 10a F2, guard 5 — the minimum body a SENTENCE-initial header must have.
#
# Not every index has dot leaders. Staging doc 89 (COEAS Andalucía) opens with
# four pages of run-on contents —
#   "Artículo 1. Ámbito territorial. Artículo 2. Ámbito funcional. Artículo 3. …"
# — with no leaders at all, so `_on_toc_line` cannot see it. Every entry is
# sentence-initial, capitalised and perfectly sequential, which is to say it
# satisfies F2 exactly. Measured: F2 without this guard turns those four pages
# into 76 chunks of 28-51 characters that are pure title and zero content —
# strictly worse retrieval decoys than the packed chunks they replace.
#
# The separating fact, measured across all 33 chunked prose documents on
# staging: an index entry's segment is 28-51 characters, while the smallest
# genuine article F2 recovers is 306 (doc 67 `Art. 2.- Ámbito territorial`).
# The threshold sits near the geometric midpoint of that gap, calibrated to
# neither edge, with ~3x margin below and ~2x above.
#
# Measured cost over that same corpus: 75 index segments dropped from doc 89,
# and exactly one genuine article anywhere else — doc 67 `Art. 51.- De la
# jubilación. Según la legislación vigente en cada momento.` (77 chars), a stub
# that defers to the law and carries nothing retrievable of its own.
#
# It fails safe. Dropping an anchor never deletes text — the segment stays
# merged into the preceding chunk, which is exactly what the pre-Sprint-10a
# chunker did with it. The worst case for a genuinely short article is
# "no better than before", never "lost".
_MIN_SENTENCE_ANCHOR_BODY = 150


def _line_bounds(text: str, s: int) -> tuple[int, int]:
    """(start, end) offsets of the physical line containing offset `s`."""
    start = text.rfind("\n", 0, s) + 1
    end = text.find("\n", s)
    return start, (len(text) if end == -1 else end)


def _on_toc_line(text: str, s: int) -> bool:
    """Guard 4 (Sprint 10a F1) — is this candidate on a table-of-contents line?

    Line-scoped, not page-scoped or offset-scoped: a real article header never
    ends in a dot leader + page number, and an index entry always does. This
    rejects the anchor only; the index text itself still flows into the
    paragraph fallback rather than being deleted (the extract_columns rule —
    keep a stray bit of furniture rather than risk deleting an article)."""
    start, end = _line_bounds(text, s)
    return _TOC_LINE.search(text[start:end]) is not None


def _at_sentence_start(text: str, s: int) -> bool:
    """Sprint 10a F2 — is this candidate the first token of a sentence?

    Guard 1 (`_at_line_start`) exists to kill inline cross-references, and it is
    right to. But it also drops a REAL header whose line break did not survive
    PDF text extraction. Measured on staging doc 75 (ESTATUTO julio2025), that
    cost exactly three articles their own chunk — 26, 27 and **37**. Article 37
    is `Descanso semanal, fiestas y permisos`, i.e. THE permisos article, and it
    was buried mid-way through a 4 613-char chunk that opens with art. 36
    (Trabajo nocturno), spanning pages 66-74. That is precisely the Correction-03
    buried-grant shape the 2c re-chunk was built to eliminate. In the extracted
    stream it reads:
        "…durante la jornada de trabajo. Artículo 37. Descanso semanal, …"

    So: accept a candidate that opens a sentence (preceded by `.`/`;`/`:` and
    whitespace). This is a WEAKER positional signal than a line start, so the
    caller compensates by requiring BOTH remaining guards — capitalised AND
    monotonic — where a line-anchored candidate needs only capitalisation.
    A lowercase inline reference ("…según el artículo 22 del Estatuto…") is
    still rejected on case; a capitalised backward reference is still rejected
    on monotonicity. Deliberately NOT a per-article special case for 26/27/37:
    hardcoding known article numbers is the digit-regex failure class deleted in
    7f (ADR-0028)."""
    i = s - 1
    while i >= 0 and text[i] in " \t\n\r":
        i -= 1
    return i >= 0 and text[i] in ".;:"


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
    # Sprint 10a F2: the number of the most recently ACCEPTED numeric header —
    # distinct from `last_num`, which is a running HIGH-WATER MARK. Guard 3 uses
    # the high-water mark, which is right for its job (rejecting a lowercase
    # backward reference) but far too weak to license a sentence-initial header:
    # measured on doc 75, `last_num` is already 58 by the time the body reaches
    # article 23, so `num >= last_num` rejects every real sentence-initial
    # header in the document. `prev_num` supports the much stronger and
    # non-tunable test F2 actually wants — "is this the NEXT header in a running
    # sequence?".
    prev_num = 0
    # Offsets accepted via F2 rather than a line start — the only ones guard 5
    # is allowed to reconsider. A line-anchored header keeps 2c's behaviour
    # exactly, however short its article is.
    sentence_anchored: set[int] = set()
    for start, kind, num in cands:
        # Guard 4 (Sprint 10a F1): never anchor on a table-of-contents entry,
        # whatever else is true about it. Checked before the positional guards
        # because a TOC line IS line-anchored and capitalised and monotonic —
        # it passes all three original guards, which is why the Estatuto's index
        # produced 25 article-lookalike chunks.
        if _on_toc_line(text, start):
            continue

        line_anchored = _at_line_start(text, start)

        # Guard 1, relaxed (Sprint 10a F2): a numeric es header may also open a
        # SENTENCE, for the case where the PDF's line break did not survive
        # extraction. Only `es_num` — it is the only kind with both remaining
        # guards (case AND number) available to compensate for the weaker
        # position. `es_ord`/`disp`/`eu` stay strictly line-anchored: they have
        # no number to check monotonicity against, so relaxing them would trade
        # a real precision guard for nothing.
        if not line_anchored:
            if kind != "es_num" or not _at_sentence_start(text, start):
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

        if not line_anchored:
            # Sprint 10a F2: a SENTENCE-initial candidate has the weakest
            # position, so it must clear BOTH remaining guards — capitalised AND
            # in-sequence — not either one, and "in sequence" means the strict
            # SUCCESSOR of the last accepted header, not merely "not smaller".
            # This is what keeps the 2c false-positive corpus at 100% rejected:
            # a lowercase inline ref fails the case guard, and any capitalised
            # cross-reference — backward OR forward — fails the successor test
            # unless it happens to name exactly the next article, in a sentence
            # that opens with it, in a document that has not already emitted it.
            if is_cap and num is not None and num == prev_num + 1:
                accepted.append(start)
                sentence_anchored.add(start)
                prev_num = num
                last_num = max(last_num, num)
            continue

        if is_cap:
            accepted.append(start)
            if num is not None:
                last_num = max(last_num, num)
                prev_num = num
        elif num is not None and num >= last_num:
            accepted.append(start)
            last_num = num
            prev_num = num

    # De-dup (an offset can match >1 candidate pattern) and keep order.
    ordered = sorted(set(accepted))

    # Guard 5 (Sprint 10a F2): a sentence-initial header must actually have an
    # article under it. Applied here rather than inside the loop because the
    # segment's extent is only known once the NEXT anchor is known.
    #
    # Single pass, gaps measured against the pre-removal list. That is the
    # conservative direction for the case this exists for: a run-on index is a
    # run of consecutive tiny gaps, so every entry in it is dropped together and
    # the whole index collapses back into one packed chunk. Re-measuring after
    # each removal would instead let the first entry survive by inheriting the
    # gap of everything dropped after it.
    if sentence_anchored:
        kept = []
        for i, off in enumerate(ordered):
            if off in sentence_anchored:
                end = ordered[i + 1] if i + 1 < len(ordered) else len(text)
                if end - off < _MIN_SENTENCE_ANCHOR_BODY:
                    continue
            kept.append(off)
        return kept

    return ordered


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


def _strip_toc_lines(content: str) -> str:
    """Sprint 10a F1, second half — drop table-of-contents lines from a chunk's
    body, keeping every other line.

    Guard 4 stops an index entry from STARTING a chunk, but the index text still
    flows into the preamble's paragraph fallback. On doc 75 that leaves six
    front-matter chunks (pages 1-11) that are 52-93% dot-leader lines: no
    answerable content, but the full topical keyword set of every article in the
    document. Harmless while convenio chunks out-compete them; on the
    national-law-only fallback path they compete directly for the synthesis cap
    against the very questions the fallback exists to answer.

    Line-level, so there is no ratio threshold to tune and no whole chunk is
    deleted — the same granularity as the guard, and the same treatment
    `extract_columns` already gives repeating headers/footers. It cannot delete
    an article: a rule of law never ends in a dot leader followed by a page
    number. A chunk left empty by the strip is dropped by `emit()`'s existing
    empty check, which is how the pure-index chunks disappear without a
    special case."""
    kept = [ln for ln in content.splitlines() if not _TOC_LINE.search(ln)]
    return "\n".join(kept)


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
        # Sprint 10a F1: index lines are furniture — strip them before the empty
        # check, so a chunk that was nothing but index disappears here rather
        # than being special-cased anywhere downstream.
        content = _strip_toc_lines(content).strip()
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
