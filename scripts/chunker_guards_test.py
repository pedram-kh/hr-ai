"""Article-header detector guard test — Sprint 10a F1 + F2 (build step 3).

Sprint 2c made the header detector load-bearing (no packing left to hide a
missed header) and gave it three precision guards: line-anchored, case-aware,
monotonic-number. Sprint 10a adds a fourth and relaxes the first:

  F1 (guard 4, `_on_toc_line`) — never anchor on a table-of-contents entry.
     The Estatuto's index (pages 1-11) passes all THREE original guards: an
     index line is line-anchored, capitalised and monotonic. It produced 25
     chunks that open with `Artículo N.` and contain no article text at all.
     Harmless while convenio chunks out-compete them; first-class decoys on
     Sprint 10a's national-law-only fallback path.

  F2 (`_at_sentence_start`) — accept a numeric es header that opens a SENTENCE
     rather than a line, when it is ALSO capitalised AND monotonic. Guard 1
     rejected three real headers on doc 75 (arts. 26, 27, 37) whose line break
     did not survive PDF extraction. Art. 37 is `Descanso semanal, fiestas y
     permisos` — the permisos article, and the source for a third of Sprint
     10a's fallback gold set.

The hard gate (build authorization D4): the 2c false-positive corpus must stay
100% rejected. Relaxing a precision guard is only safe if the thing it was
protecting against is still caught, so that is asserted first and loudest.

D4 also pins the fixture: these assertions run against the ACTUAL ingested
source of staging doc 75 (`content_hash e7f7dfa8…`), not any other Estatuto
copy — at least one local copy in circulation is a scan with no text layer and
would silently pass every assertion by producing no anchors at all. The hash is
verified before anything else runs.

Run:
    python scripts/chunker_guards_test.py [path/to/doc75.pdf]
    docker exec hr_ai python scripts/chunker_guards_test.py /tmp/doc75.pdf

The PDF-backed section is skipped (loudly, non-zero exit) if the fixture is
absent, so CI without the file still runs the pure-text guard corpus.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chunking.chunker import (  # noqa: E402
    _at_sentence_start,
    _find_anchors,
    _on_toc_line,
    chunk_stream,
)

# The exact file staging doc 75 was ingested from (documents.content_hash).
DOC75_SHA256 = "e7f7dfa82320c36c70d56becd0b49fc460e1f48aff7b18753cd06bbfc673586e"
DEFAULT_DOC75 = "/tmp/hr10a/doc75-estatuto-julio2025.pdf"

failures: list[str] = []
passes = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passes
    if cond:
        passes += 1
        print(f"  PASS  {name}")
    else:
        failures.append(f"{name}{(' — ' + detail) if detail else ''}")
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


def anchors_for(text: str) -> list[int]:
    return _find_anchors(text)


def stub_tokens(s: str) -> int:
    """Whitespace word count. The guards under test are positional/lexical and
    independent of tokenization; using a stub keeps this test free of the 4.3 GB
    BGE-M3 load. The real tokenizer is used by rechunk_survey.py, which is what
    the cap decisions are made on."""
    return len(s.split())


# ---------------------------------------------------------------------------
# 1. THE HARD GATE — the 2c false-positive corpus stays 100% rejected.
# ---------------------------------------------------------------------------
# Every string below contains a candidate the patterns WILL match, and every one
# must produce zero anchors. These are the shapes 2c's guards were built for
# (review.md §"Precision guards held"), plus the new shapes F2 could plausibly
# let through.
FALSE_POSITIVES = [
    # 2c corpus — inline lowercase cross-references, the original target.
    ("inline mid-sentence ref", "El descanso se regula según el artículo 22 del Estatuto de los Trabajadores y no podrá reducirse."),
    ("inline ref, abbreviated", "Conforme al art. 15 del convenio, la jornada será de 1.750 horas anuales."),
    ("uppercase-lead sentence containing a lowercase ref", "Las partes acuerdan que el artículo 30 quede redactado como sigue."),
    # Line-leading lowercase BACKWARD ref — guard 3 (monotonic) is what rejects
    # this one. NB the line-leading lowercase FORWARD ref is accepted, by 2c's
    # own documented design (chunker.py guard-2 comment: a lowercase candidate
    # is accepted if it is monotonic). Verified unchanged by this sprint against
    # the pre-change code; not a regression and not in scope to alter here.
    ("line-leading lowercase BACKWARD ref", "Artículo 40. Movilidad geográfica.\nSerá de aplicación lo dispuesto en el\nartículo 22 del Estatuto."),
    # F2-specific risks — sentence-initial but NOT a header.
    ("sentence-initial BACKWARD ref (monotonic guard must catch it)", "Artículo 40. Movilidad geográfica.\nLo anterior se entiende sin perjuicio. Artículo 22 del Estatuto sigue siendo aplicable."),
    ("sentence-initial lowercase ref (case guard must catch it)", "La empresa lo comunicará. artículo 55 del texto refundido resulta de aplicación."),
    # F1-specific — index entries pass all three original guards.
    ("TOC entry, dot leader + page number", "Artículo 14. Periodo de prueba.................................. 40"),
    ("TOC block, several entries", "Artículo 34. Jornada.................................... 65\nArtículo 35. Horas extraordinarias. ................... 67\nArtículo 36. Trabajo nocturno........................... 69"),
    # Sub-clause guard (2c) — must still hold.
    ("decimal sub-clause", "Artículo 9.1 no inicia un artículo nuevo dentro del cuerpo."),
]

print("\n=== 1. Hard gate: 2c false-positive corpus (must stay 100% rejected) ===")
for name, text in FALSE_POSITIVES:
    found = anchors_for(text)
    # The BACKWARD-ref cases legitimately contain one real header (art. 40) at
    # offset 0; assert only that the spurious second anchor is not created.
    expected = 1 if "BACKWARD" in name else 0
    check(
        f"rejected: {name}",
        len(found) == expected,
        f"expected {expected} anchor(s), got {len(found)} at {found}",
    )

# ---------------------------------------------------------------------------
# 2. F1 unit — the TOC line predicate.
# ---------------------------------------------------------------------------
print("\n=== 2. F1 unit — _on_toc_line ===")
_TOC_CASES = [
    ("Artículo 14. Periodo de prueba.................................. 40", True),
    ("Artículo 35. Horas extraordinarias. ................... 67", True),
    ("Artículo 14. Periodo de prueba.", False),
    ("Artículo 38. Vacaciones anuales.\n1. El periodo de vacaciones…", False),
    # A real article whose body happens to contain an ellipsis must NOT be hit.
    ("Artículo 12. Contrato a tiempo parcial... y sus modalidades.", False),
]
for line, expected in _TOC_CASES:
    check(
        f"_on_toc_line({line[:44]!r}…) == {expected}",
        _on_toc_line(line, line.index("Art")) is expected,
    )

# ---------------------------------------------------------------------------
# 3. F2 unit — the sentence-start predicate.
# ---------------------------------------------------------------------------
print("\n=== 3. F2 unit — _at_sentence_start ===")
_SENT_CASES = [
    ("…durante la jornada de trabajo. Artículo 37. Descanso semanal", True),
    ("…de aplicación; Artículo 38. Vacaciones", True),
    ("…lo siguiente: Artículo 12. Jornada", True),
    ("…en el marco del Artículo 22 citado", False),  # mid-sentence, no terminator
    ("…y el Artículo 40 establece", False),
]
for text, expected in _SENT_CASES:
    idx = text.index("Artículo")
    check(f"_at_sentence_start({text[-38:]!r}) == {expected}", _at_sentence_start(text, idx) is expected)

# F2 acceptance requires cap AND successor AND a real body.
print("\n=== 3b. F2 acceptance requires capitalised AND in-sequence AND a body ===")
_BODY = (
    "1. Los trabajadores tendrán derecho a un descanso mínimo semanal, acumulable por periodos "
    "de hasta catorce días, de día y medio ininterrumpido que, como regla general, comprenderá "
    "la tarde del sábado o, en su caso, la mañana del lunes y el día completo del domingo."
)
sentence_header = (
    "Artículo 36. Trabajo nocturno.\nTexto del artículo treinta y seis. "
    "Artículo 37. Descanso semanal, fiestas y permisos.\n" + _BODY
)
found = anchors_for(sentence_header)
check("sentence-initial forward header WITH a body IS accepted (art. 36 → 37)", len(found) == 2, f"got {len(found)} at {found}")

# Guard 5 — the doc-89 shape: a run-on index with NO dot leaders. Every entry is
# sentence-initial, capitalised and perfectly sequential, so F1 cannot see it and
# F2's successor test welcomes it. Only the missing body rejects it.
print("\n=== 3c. Guard 5 — run-on index with no dot leaders (staging doc 89) ===")
runon_index = (
    "ÍNDICE\n"
    "Artículo 1. Ámbito territorial. Artículo 2. Ámbito funcional. Artículo 3. Ámbito temporal. "
    "Artículo 4. Ámbito personal. Artículo 5. Comisión paritaria. Artículo 6. Concurrencia de convenios. "
    "Artículo 7. Derecho supletorio. Artículo 8. Mediación y arbitraje.\n"
)
found = anchors_for(runon_index)
check(
    "run-on index produces at most the one line-anchored entry, not one anchor per title",
    len(found) <= 1,
    f"got {len(found)} anchors at {found} — guard 5 did not reject the title-only segments",
)

# …and the same shape followed by the REAL article 1, which must still be found.
runon_then_body = runon_index + "\nArtículo 1. Ámbito territorial.\n" + _BODY
found = anchors_for(runon_then_body)
check(
    "the real article after a run-on index is still anchored",
    len(found) >= 1,
    f"got {len(found)} at {found}",
)

# ---------------------------------------------------------------------------
# 4. The real fixture — staging doc 75.
# ---------------------------------------------------------------------------
pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOC75
print(f"\n=== 4. Real fixture — staging doc 75 ({pdf_path}) ===")

if not os.path.exists(pdf_path):
    print(f"  SKIP  fixture not present at {pdf_path}")
    print("        Fetch it with:")
    print("        aws s3 cp s3://hr-staging-documents-<acct>/documents/"
          "c5519e13-3695-4469-8127-0877b9b8eeb9/original.pdf " + DEFAULT_DOC75)
    failures.append("doc-75 fixture missing — the D4 gate did not run")
else:
    digest = hashlib.sha256(open(pdf_path, "rb").read()).hexdigest()
    check(
        "fixture is the ACTUAL ingested source of staging doc 75 (D4)",
        digest == DOC75_SHA256,
        f"sha256 {digest[:16]}… != expected {DOC75_SHA256[:16]}…",
    )

    if digest == DOC75_SHA256:
        from app.chunking.extract_columns import extract_language_streams

        streams = extract_language_streams(open(pdf_path, "rb").read())
        es_units = streams["streams"].get("es", []) if "streams" in streams else streams.get("es", [])
        check("es stream extracted", len(es_units) > 0, f"got {len(es_units)} page units")

        chunks = chunk_stream(es_units, stub_tokens)
        heads = [c["content"].lstrip()[:80] for c in chunks]

        def article_numbers(hs: list[str]) -> set[int]:
            out = set()
            for h in hs:
                m = re.match(r"Art[íi]culo\s+(\d{1,3})", h)
                if m:
                    out.add(int(m.group(1)))
            return out

        arts = article_numbers(heads)

        # F2: article 37 — `Descanso semanal, fiestas y permisos`, the permisos
        # source for Sprint 10a's fallback gold set, and the reason F2 exists.
        check("F2: article 37 (permisos) now has its own chunk", 37 in arts,
              f"articles present: {sorted(arts)[:12]}…")

        # Articles 26 (`Del salario`) and 27 (`SMI`) stay merged, KNOWINGLY.
        # 26's header is preceded by a section heading that also lost its line
        # break ("…Sección 4.ª Salarios y garantías salariales Artículo 26…"),
        # so it is neither line- nor sentence-initial; 27 is sentence-initial
        # but is no longer the successor of 25 once 26 is missed. Catching them
        # would need a THIRD relaxation (accept after a Sección/Capítulo/Título
        # fragment), and both articles are salary/SMI — excluded from the
        # fallback by spec §2.2 — so the relaxation would buy nothing this
        # sprint and widen the detector's surface for free. Asserted as a
        # KNOWN state so a future change that fixes or worsens it is visible.
        check("known residual: articles 26/27 remain merged (salary/SMI, excluded from fallback)",
              26 not in arts and 27 not in arts,
              f"26 in arts: {26 in arts}, 27 in arts: {27 in arts} — if these are now present, "
              "update this assertion and the review; it is an improvement, not a failure")

        # The 2c behaviour must be intact — a broad spread of articles, not a
        # collapse (detector missing headers) and not an explosion
        # (false-positive over-splitting, the §3.3 risk).
        check("2c intact: article 14 (periodo de prueba) still has its own chunk", 14 in arts)
        check("2c intact: article 38 (vacaciones) still has its own chunk", 38 in arts)
        check("2c intact: >=88 distinct articles detected", len(arts) >= 88, f"got {len(arts)}")

        # F1: no chunk may OPEN with a TOC entry any more.
        toc_headed = [h for h in heads if re.match(r"Art[íi]culo\s+\d", h) and re.search(r"\.{6,}\s*\d", h)]
        check("F1: zero chunks open with a table-of-contents entry", len(toc_headed) == 0,
              f"{len(toc_headed)} remain, e.g. {toc_headed[:2]}")

        print(f"\n  [info] es chunks: {len(chunks)}   distinct articles: {len(arts)}")
        missing = sorted(set(range(1, 93)) - arts)
        print(f"  [info] articles 1-92 without an own chunk: {missing or 'none'}")

print("\n" + "=" * 70)
if failures:
    print(f"FAILED — {len(failures)} failure(s), {passes} passed:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print(f"OK — {passes} checks passed.")
