"""Column-aware re-extraction: split Euskara (left) / Spanish (right) and strip
page furniture BEFORE column assignment (ADR-0013, plan §3.1 + catch 1).

Furniture stripping is a *never-blend correctness requirement*, not hygiene: the
real Gipuzkoa BOG footer is itself bilingual
(``…Gipuzkoako Aldizkari Ofiziala / Boletín Oficial de Gipuzkoa…``). Leaving it
in a column would create a blended-language chunk, which ADR-0013 forbids.

Sprint 2a Correction-01 — bias every uncertain call toward KEEPING TEXT and
SINGLE COLUMN. For a legal-weight system the safe failure direction is to keep a
stray bit of furniture (mild retrieval noise) rather than delete an article (a
silent legal gap). Two governing rules follow from that:

1. **Two-column only on positive evidence of a real two-column prose body** — two
   tall, balanced, vertically-overlapping text columns separated by a clear
   central gutter, and NOT a row-aligned table. An article index/TOC (titles left,
   thin page-numbers right → unbalanced) and ``GRUPO/CATEGORÍA``/salary tables
   (short, row-paired cells) must NOT trip it. When evidence is weak → single
   column (one Spanish stream). So spurious ``eu`` on monolingual docs disappears;
   ``eu`` appears only on genuinely bilingual pages (Gipuzkoa).
2. **Full-width alone is NOT furniture.** A block is furniture only when it
   *repeats* across pages in a top/bottom margin band, or matches known gazette
   boilerplate in that band. A non-repeating full-width block in the body (e.g. a
   preámbulo paragraph spanning the page) is prose → keep it.
"""

from __future__ import annotations

import re

import fitz  # PyMuPDF

from ..config import settings
from .normalize import despace_block

_WS = re.compile(r"\s+")

# Known gazette/official-bulletin boilerplate (matched on normalized lowercased
# text, and ONLY inside the top/bottom margin band). Repetition is the primary
# furniture signal; this is the secondary "OR matches known boilerplate" clause
# (catches a masthead that varies per page — date/number — so it never repeats
# identically). Deliberately narrow; it never matches body prose.
_BOILERPLATE = re.compile(
    r"(bolet[íi]n\s+oficial|aldizkari\s+ofiziala|d\.?\s*l\.?\s*:|issn|"
    r"www\.[\w.-]+\.(?:eus|es|com|net)|n[úu]m\.\s*\d|"
    r"^\s*\d+\s*/\s*\d+\s*$|p[áa]g(?:ina)?s?\.?\s*\d)",
    re.IGNORECASE,
)


# Very common Spanish function words that essentially never appear as standalone
# tokens in Basque. Used ONLY to decide stream routing on a geometrically
# two-column page (is the left column actually Basque, or is this a monolingual
# two-column Spanish layout?) — NOT a retrieval filter (ADR-0006 forbids that).
_ES_STOP = {
    "de", "la", "el", "los", "las", "que", "en", "por", "para", "con", "del",
    "al", "un", "una", "se", "su", "sus", "lo", "como", "más", "es", "son",
    "será", "ser", "este", "esta", "estos", "estas", "no", "y", "o", "le",
    "les", "entre", "sobre", "cuando", "donde", "cada", "dicho", "dicha",
}
_WORD = re.compile(r"[a-záéíóúüñ]+", re.IGNORECASE)


def _norm_key(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def _es_ratio(text: str) -> float:
    """Fraction of tokens that are common Spanish function words. Spanish prose
    runs ~0.12–0.20; Basque prose runs ~0 (it has none of these). Used to tell a
    real bilingual page (one Basque column, one Spanish) from a monolingual
    two-column Spanish page (both columns Spanish)."""
    toks = _WORD.findall(text.lower())
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _ES_STOP) / len(toks)


def _vol(blocks: list[dict]) -> int:
    return sum(len(b["text"]) for b in blocks)


def _yspan(blocks: list[dict]) -> float:
    if not blocks:
        return 0.0
    return max(b["y1"] for b in blocks) - min(b["y0"] for b in blocks)


def _classify_page(pbs: list[dict], pw: float, ph: float) -> dict:
    """Decide a page's layout from POSITIVE evidence (Correction-01, change 1).

    Returns {"two_column": bool, "tabular": bool, "has_straddle": bool}. Defaults
    to single column whenever the two-column evidence is weak.
    """
    center = pw / 2.0
    gutter = 0.04 * pw

    left: list[dict] = []
    right: list[dict] = []
    straddle: list[dict] = []
    for b in pbs:
        if b["x0"] < center - gutter and b["x1"] > center + gutter:
            straddle.append(b)  # crosses the central gutter ⇒ full-width-ish
        elif b["x_mid"] < center:
            left.append(b)
        else:
            right.append(b)

    lvol, rvol, svol = _vol(left), _vol(right), _vol(straddle)

    # Row-aligned SHORT-CELL table test. A data table / TOC is (a) row-paired —
    # most left blocks have a right block at ~the same y — AND (b) made of short
    # cells (labels, codes, numbers). Crucially this must NOT fire on a genuine
    # parallel-bilingual page (Gipuzkoa: Basque left / Spanish right), whose rows
    # ARE aligned by translation but whose cells are long PROSE. So we require
    # short cells too. Tolerance ~ a line height.
    paired = 0
    for lb in left:
        if any(abs(lb["y0"] - rb["y0"]) < 6.0 for rb in right):
            paired += 1
    row_aligned = len(left) >= 3 and (paired / len(left)) > 0.6
    side_texts = [b["text"] for b in left] + [b["text"] for b in right]
    short_cells = (
        bool(side_texts)
        and sum(1 for t in side_texts if len(t) < 30) / len(side_texts) > 0.5
    )
    tabular = row_aligned and short_cells

    # Tall, balanced, vertically-overlapping columns with a clear gutter.
    lspan, rspan = _yspan(left), _yspan(right)
    overlap = 0.0
    if left and right:
        top = max(min(b["y0"] for b in left), min(b["y0"] for b in right))
        bottom = min(max(b["y1"] for b in left), max(b["y1"] for b in right))
        overlap = max(0.0, bottom - top)

    max_vol = max(lvol, rvol)
    balanced = max_vol > 0 and min(lvol, rvol) >= 0.55 * max_vol
    tall = lspan >= 0.5 * ph and rspan >= 0.5 * ph
    overlapping = overlap >= 0.4 * ph
    gutter_clear = (lvol + rvol) > 0 and svol <= 0.15 * (lvol + rvol)

    two_column = (
        len(left) >= 2
        and len(right) >= 2
        and balanced
        and tall
        and overlapping
        and gutter_clear
        and not tabular
    )
    return {"two_column": two_column, "tabular": tabular, "has_straddle": svol > 0}


def extract_language_streams(pdf_bytes: bytes) -> dict:
    """Return per-language page-tagged text units + extraction stats.

    {
      "streams": {"es": [(page_number, text), ...], "eu": [(page_number, text), ...]},
      "stats": {...},
    }
    """
    ratio = settings.chunk_space_gap_ratio
    repeat_fraction = settings.chunk_repeat_furniture_min_page_fraction

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_count = doc.page_count

        # --- Pass 1: collect de-spaced text blocks with geometry ---
        blocks: list[dict] = []
        long_token_flags = 0
        for index in range(page_count):
            page = doc.load_page(index)
            page_number = index + 1
            pw = float(page.rect.width) or 1.0
            ph = float(page.rect.height) or 1.0
            raw = page.get_text("rawdict")
            for blk in raw.get("blocks", []):
                if blk.get("type", 0) != 0:
                    continue  # image block, no text
                text, flags = despace_block(blk, ratio)
                long_token_flags += flags
                if not text:
                    continue
                x0, y0, x1, y1 = blk.get("bbox", (0, 0, 0, 0))
                blocks.append(
                    {
                        "page": page_number,
                        "pw": pw,
                        "ph": ph,
                        "x0": float(x0),
                        "x1": float(x1),
                        "y0": float(y0),
                        "y1": float(y1),
                        "x_mid": (float(x0) + float(x1)) / 2.0,
                        "width": float(x1) - float(x0),
                        "text": text,
                        "furniture": False,
                    }
                )

        blocks_total = len(blocks)

        # --- Pass 2: mark furniture (Correction-01, change 2) ---
        # Furniture is REPETITION at a margin band — NOT bare full width. A
        # full-width body paragraph that does not repeat stays as prose.
        furniture_stripped = 0

        def _in_margin_band(b: dict) -> bool:
            return b["y0"] < 0.12 * b["ph"] or b["y1"] > 0.88 * b["ph"]

        # (a) Repeating top/bottom-band lines (same normalized text on
        #     >= repeat_fraction of pages). This catches the bilingual Gipuzkoa
        #     footer (recurs on all pages) — the catch-1 correctness win.
        band_counts: dict[str, set[int]] = {}
        for b in blocks:
            if _in_margin_band(b):
                key = _norm_key(b["text"])
                if 0 < len(key) <= 160:  # short-ish lines only (real furniture)
                    band_counts.setdefault(key, set()).add(b["page"])
        repeating = {
            k for k, pages in band_counts.items()
            if page_count > 1 and len(pages) >= max(2, repeat_fraction * page_count)
        }
        repeating_furniture_lines: list[str] = sorted(repeating) if repeating else []

        for b in blocks:
            if not _in_margin_band(b):
                continue
            key = _norm_key(b["text"])
            # (a) repeating, or (b) known gazette boilerplate in the band.
            if key in repeating or (len(key) <= 160 and _BOILERPLATE.search(b["text"])):
                b["furniture"] = True
                furniture_stripped += 1

        column_blocks = [b for b in blocks if not b["furniture"]]

        # --- Pass 3: per-page layout (positive-evidence two-column) + assign ---
        # Stream units carry a column index so a monolingual two-column page is
        # read left-column-then-right-column (not interleaved by y) when both
        # columns land in the same stream.
        es: list[tuple[int, int, float, str]] = []  # (page, col, y0, text)
        eu: list[tuple[int, int, float, str]] = []
        two_column_pages = 0
        single_column_pages = 0
        bilingual_pages = 0
        tabular_pages = 0
        pages_not_cleanly_split: list[int] = []

        by_page: dict[int, list[dict]] = {}
        for b in column_blocks:
            by_page.setdefault(b["page"], []).append(b)

        for page_number in sorted(by_page):
            pbs = by_page[page_number]
            pw = pbs[0]["pw"]
            ph = pbs[0]["ph"]
            center = pw / 2.0
            layout = _classify_page(pbs, pw, ph)

            if layout["tabular"]:
                tabular_pages += 1

            if layout["two_column"]:
                two_column_pages += 1
                left_blocks = [b for b in pbs if b["x_mid"] < center]
                right_blocks = [b for b in pbs if b["x_mid"] >= center]
                lr = _es_ratio(" ".join(b["text"] for b in left_blocks))
                rr = _es_ratio(" ".join(b["text"] for b in right_blocks))
                # Bilingual ONLY when one column reads as Basque (≈0 Spanish
                # function words) and the other clearly as Spanish. Otherwise it
                # is a monolingual two-column Spanish layout (e.g. Valencia) —
                # KEEP BOTH in the Spanish stream, ordered by column, never tag
                # Spanish as eu.
                bilingual = (lr < 0.05 and rr > 0.07) or (rr < 0.05 and lr > 0.07)
                if bilingual:
                    bilingual_pages += 1
                    eu_blocks, es_blocks = (
                        (left_blocks, right_blocks) if lr <= rr else (right_blocks, left_blocks)
                    )
                    for b in eu_blocks:
                        eu.append((b["page"], 0, b["y0"], b["text"]))
                    for b in es_blocks:
                        es.append((b["page"], 0, b["y0"], b["text"]))
                    if layout["has_straddle"]:
                        pages_not_cleanly_split.append(page_number)
                else:
                    for b in left_blocks:
                        es.append((b["page"], 0, b["y0"], b["text"]))
                    for b in right_blocks:
                        es.append((b["page"], 1, b["y0"], b["text"]))
            else:
                # Weak evidence ⇒ single column (one Spanish stream): KEEP TEXT.
                single_column_pages += 1
                for b in pbs:
                    es.append((b["page"], 0, b["y0"], b["text"]))
                # A row-aligned table/TOC is genuine non-prose — surface it for
                # eyes-on (its text is still kept in the es stream).
                if layout["tabular"]:
                    pages_not_cleanly_split.append(page_number)

        def _stream(units: list[tuple[int, int, float, str]]) -> list[tuple[int, str]]:
            units.sort(key=lambda u: (u[0], u[1], u[2]))
            return [(p, t) for p, _, _, t in units]

        return {
            "streams": {"es": _stream(es), "eu": _stream(eu)},
            "stats": {
                "page_count": page_count,
                "two_column_pages": two_column_pages,
                "bilingual_pages": bilingual_pages,
                "single_column_pages": single_column_pages,
                "tabular_pages": tabular_pages,
                "blocks_total": blocks_total,
                "column_blocks_kept": len(column_blocks),
                "furniture_blocks_stripped": furniture_stripped,
                "repeating_furniture_lines": repeating_furniture_lines,
                "pages_not_cleanly_split": sorted(set(pages_not_cleanly_split)),
                "long_token_flags": long_token_flags,
            },
        }
    finally:
        doc.close()
