"""Article-aware chunking with a size cap (ADR-0013, plan §3.3).

Split a single-language stream on article boundaries (`Artículo N` / `N.
artikulua`, plus the Estatuto's `Disposición adicional/transitoria/…`);
sub-split oversized articles on paragraph/sentence boundaries; pack tiny
consecutive articles together up to the cap. Where no article anchors exist
(e.g. an annex), fall back to a size-capped paragraph chunker. Each chunk
records `page_from`/`page_to` and `token_count`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from ..config import settings

# Article / disposition anchors (case-insensitive). `artikulua` is Euskara;
# `Artículo` Spanish; the Disposición forms appear in the Estatuto.
_CORE = re.compile(
    r"(Art[íi]culo\s+\d+|Art\.\s*\d+|\d{1,3}\.?\s*artikulua|"
    r"Disposici[óo]n\s+(?:adicional|transitoria|final|derogatoria))",
    re.IGNORECASE,
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


def _article_spans(text: str) -> list[tuple[int, int]]:
    """Character spans [start, end) for each article; [] if no anchors."""
    starts: list[int] = []
    for m in _CORE.finditer(text):
        s = m.start()
        if s == 0 or text[s - 1].isspace() or text[s - 1] in "\n.;)º":
            starts.append(s)
    if not starts:
        return []
    starts = sorted(set(starts))
    if starts[0] > 0:
        starts = [0] + starts  # preamble before Article 1 is its own segment
    spans = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        spans.append((s, e))
    return spans


def _pack_to_cap(text: str, count_tokens: Callable[[str], int], cap: int) -> list[str]:
    """Greedily split text into pieces each <= cap tokens (paragraph → sentence
    → word granularity)."""
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


def chunk_stream(
    units: list[tuple[int, str]],
    count_tokens: Callable[[str], int],
    target: int | None = None,
    cap: int | None = None,
) -> list[dict]:
    """Chunk one language stream → list of {content, page_from, page_to, token_count}."""
    target = target or settings.chunk_token_target
    cap = cap or settings.chunk_token_cap
    text, pagemap = _build_text_and_pagemap(units)
    if not text.strip():
        return []

    spans = _article_spans(text)
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

    if not spans:
        # Fallback: size-capped paragraph chunker over the whole stream.
        for piece in _pack_to_cap(text, count_tokens, target):
            off = text.find(piece[:40]) if piece else -1
            off = off if off >= 0 else 0
            emit(piece, off, off + len(piece))
        return chunks

    # Article-aware: pack tiny consecutive articles up to `target`; sub-split
    # oversized ones to `cap`.
    buf_text = ""
    buf_start = spans[0][0]
    for s, e in spans:
        seg = text[s:e].strip()
        if not seg:
            continue
        if count_tokens(seg) > cap:
            if buf_text:
                emit(buf_text, buf_start, s)
                buf_text = ""
            for piece in _pack_to_cap(seg, count_tokens, cap):
                emit(piece, s, e)
            buf_start = e
            continue
        candidate = (buf_text + "\n\n" + seg).strip() if buf_text else seg
        if buf_text and count_tokens(candidate) > target:
            emit(buf_text, buf_start, s)
            buf_text = seg
            buf_start = s
        else:
            if not buf_text:
                buf_start = s
            buf_text = candidate
    if buf_text:
        emit(buf_text, buf_start, spans[-1][1])
    return chunks


def chunk_document(streams: dict[str, list[tuple[int, str]]], count_tokens: Callable[[str], int]) -> list[dict]:
    """Chunk both language streams and return a single deterministically-ordered
    list (by first page, then es before eu, then stream order). `chunk_index` is
    assigned by the caller after embedding."""
    out: list[dict] = []
    for lang in ("es", "eu"):
        for c in chunk_stream(streams.get(lang, []), count_tokens):
            c["language"] = lang  # internal only — NOT stored on document_chunks
            out.append(c)
    out.sort(key=lambda c: (c["page_from"], 0 if c["language"] == "es" else 1))
    return out
