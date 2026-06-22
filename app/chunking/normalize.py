"""De-spacing the intra-word spacing artifact (ADR-0013, plan §3.2).

PyMuPDF extraction of the justified two-column BOG layout pervasively splits
tokens: ``hi tzar men`` → ``hitzarmen``, ``Gi puz koa`` → ``Gipuzkoa``,
``rea li za do`` → ``realizado``. Justification inflates the *inter-word* spaces
but leaves the *intra-word* fragment gaps near zero — so the signal is
geometric, not lexical. We reconstruct each line from per-glyph geometry: a gap
counts as a real space only when it exceeds a per-line, median-relative
threshold; smaller gaps are the artifact and get merged.

Dictionary-free on purpose (works identically for `eu` and `es`, and only has
to be good enough to *embed* — display/citation stays on the untouched
`document_pages`). Guards below avoid over-merging legitimate spaces.
"""

from __future__ import annotations

from statistics import median

# Punctuation that must never be glued onto a following glyph without respecting
# the gap (kept as its own glyph; the gap rule still applies around it).
_SEPARATORS = set(" \t\n")


def _despace_glyphs(glyphs: list[tuple[str, float, float]], ratio: float) -> tuple[str, int]:
    """glyphs: list of (char, x0, x1) in reading order for ONE line.

    Returns (text, long_token_flag_count) where the flag counts suspiciously
    long merged tokens (a weak over-merge signal surfaced for the eyes-on gate).
    """
    glyphs = [g for g in glyphs if not g[0].isspace()]
    if not glyphs:
        return "", 0
    if len(glyphs) == 1:
        return glyphs[0][0], 0

    widths = [max(x1 - x0, 0.0) for _, x0, x1 in glyphs]
    positive = [w for w in widths if w > 0]
    med = median(positive) if positive else 1.0
    threshold = ratio * med

    out: list[str] = [glyphs[0][0]]
    run_len = 1
    long_tokens = 0
    for i in range(1, len(glyphs)):
        c, x0, _ = glyphs[i]
        prev_x1 = glyphs[i - 1][2]
        gap = x0 - prev_x1
        if gap >= threshold:
            out.append(" ")
            if run_len > 28:
                long_tokens += 1
            run_len = 1
        else:
            # Merge (drop the justification artifact gap).
            run_len += 1
        out.append(c)
    if run_len > 28:
        long_tokens += 1
    return "".join(out), long_tokens


def despace_block(block: dict, ratio: float) -> tuple[str, int]:
    """Reconstruct a PyMuPDF rawdict text block into de-spaced text.

    Joins glyphs per line by geometry; joins lines with spaces and undoes
    line-end hyphenation (``traba-\\njadores`` → ``trabajadores``). Returns
    (text, long_token_flag_count).
    """
    lines_out: list[str] = []
    long_tokens = 0
    for line in block.get("lines", []):
        glyphs: list[tuple[str, float, float]] = []
        for span in line.get("spans", []):
            for ch in span.get("chars", []):
                bbox = ch.get("bbox")
                if not bbox:
                    continue
                glyphs.append((ch.get("c", ""), float(bbox[0]), float(bbox[2])))
        text, flags = _despace_glyphs(glyphs, ratio)
        long_tokens += flags
        text = text.strip()
        if text:
            lines_out.append(text)

    # Join lines, undoing end-of-line hyphenation.
    assembled: list[str] = []
    for ln in lines_out:
        if assembled and assembled[-1].endswith("-"):
            assembled[-1] = assembled[-1][:-1] + ln
        else:
            assembled.append(ln)
    return " ".join(assembled).strip(), long_tokens
