"""Slice 13d (ADR-0037) — `/synthesise` fact-identity contract, no live API.

hr-ai has no pytest suite (see planner_contract_test.py). This script fakes the
Anthropic client and asserts the ADDITIVE `fact_id` on synthesis sources:

  H1. Two fact sources from ONE document, each with its own `fact_id`, stay TWO
      citations (each carrying its `fact_id`); the in-text markers stay 1:1.
  H2. The same `fact_id` cited twice collapses to ONE citation (same source).
  H3. No `fact_id` anywhere (every prose turn, every single-fact composition):
      the legacy key applies (two null-chunk sources of one document collapse to
      one) and NO `fact_id` key appears in any returned citation.
  H4. `fact_id` never reaches the model prompt (prompt text is unchanged).
  H5. A vector-chunk citation is unchanged next to fact sources (no `fact_id`).

Run:
    python3.11 scripts/synthesis_factset_contract_test.py
"""

from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeMessages:
    def __init__(self, envelope: dict):
        self.envelope = envelope
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.envelope))],
            stop_reason="end_turn",
            model=kwargs.get("model"),
            id="msg_test",
            usage=SimpleNamespace(input_tokens=100, output_tokens=40),
        )


def _run(chunks, envelope):
    from app.providers import ProviderConfig, get_provider

    messages = _FakeMessages(envelope)
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda **_: SimpleNamespace(messages=messages)  # type: ignore[attr-defined]
    sys.modules["anthropic"] = fake_anthropic

    result = get_provider("claude").synthesise(
        "¿Cuál es mi jornada máxima anual?",
        chunks,
        "test-key",
        ProviderConfig(provider="claude", model="claude-sonnet-5", endpoint=None),
    )
    return result, messages.last_kwargs or {}


def main() -> int:
    from app.providers import ChunkInput

    failures: list[str] = []

    def fact(fact_id, content, *, with_id=True):
        return ChunkInput(
            chunk_id=None,
            document_id=50,
            page_from=None,
            page_to=None,
            content=content,
            score=1.0,
            authority_level="structured_reference",
            source_type="reference_fact",
            fact_id=fact_id if with_id else None,
        )

    chunk = ChunkInput(
        chunk_id=3558, document_id=50, page_from=9, page_to=10, content="La jornada máxima anual será de 1704 horas.",
        score=0.93, authority_level="official_convenio", source_type="chunk",
    )

    # --- H1: two facts, one document, two fact_ids → two citations ---
    chunks = [chunk, fact(140, "Año 2025: 1704 horas"), fact(143, "2 días de libre disposición")]
    envelope = {
        "answer": "Tienes 1704 horas [Fuente 2] y 2 días de libre disposición [Fuente 3].",
        "confidence": 0.9,
        "cited_sources": [2, 3],
    }
    result, kwargs = _run(chunks, envelope)
    fact_ids = [c.get("fact_id") for c in result.citations]
    if fact_ids != [140, 143]:
        failures.append(f"H1 FAILED: expected fact_ids [140, 143], got {fact_ids}: {result.citations}")
    if result.answer != "Tienes 1704 horas [Fuente 1] y 2 días de libre disposición [Fuente 2].":
        failures.append(f"H1 FAILED: markers not renumbered 1:1: {result.answer!r}")
    print("H1 two same-document facts stay two citations, each with its fact_id:", "FAIL" if any("H1" in f for f in failures) else "OK")

    # --- H2: the same fact cited twice → one citation ---
    envelope2 = dict(envelope, cited_sources=[2, 2, 3], answer="A [Fuente 2] B [Fuente 2] C [Fuente 3].")
    result2, _ = _run(chunks, envelope2)
    if [c.get("fact_id") for c in result2.citations] != [140, 143]:
        failures.append(f"H2 FAILED: {result2.citations}")
    dup = [chunk, fact(140, "Año 2025: 1704 horas"), fact(140, "Año 2025: 1704 horas")]
    result2b, _ = _run(dup, dict(envelope, cited_sources=[2, 3], answer="A [Fuente 2] B [Fuente 3]."))
    if len(result2b.citations) != 1 or result2b.answer != "A [Fuente 1] B [Fuente 1].":
        failures.append(f"H2 FAILED: same fact_id twice must collapse to one citation: {result2b.citations} / {result2b.answer!r}")
    print("H2 the same fact_id collapses to one citation:", "FAIL" if any("H2" in f for f in failures) else "OK")

    # --- H3: no fact_id → the legacy key (collapse) and no fact_id key in the output ---
    legacy = [chunk, fact(None, "Año 2025: 1704 horas", with_id=False), fact(None, "2 días de libre disposición", with_id=False)]
    result3, kwargs3 = _run(legacy, envelope)
    if len(result3.citations) != 1:
        failures.append(f"H3 FAILED: legacy null-chunk same-document sources must still collapse to 1 citation: {result3.citations}")
    if any("fact_id" in c for c in result3.citations):
        failures.append(f"H3 FAILED: fact_id leaked into a legacy citation: {result3.citations}")
    print("H3 absent fact_id => legacy key, no fact_id in output:", "FAIL" if any("H3" in f for f in failures) else "OK")

    # --- H4: fact_id is not rendered into the model prompt ---
    prompt = kwargs["messages"][0]["content"]
    prompt_legacy = kwargs3["messages"][0]["content"]
    if "140" in prompt.replace("1704", "").replace("2025", "") or "fact_id" in prompt:
        failures.append(f"H4 FAILED: fact_id appears in the model prompt: {prompt!r}")
    if prompt_legacy.split("FUENTES disponibles:")[0] != prompt.split("FUENTES disponibles:")[0]:
        failures.append("H4 FAILED: prompt framing changed")
    print("H4 fact_id never reaches the model prompt:", "FAIL" if any("H4" in f for f in failures) else "OK")

    # --- H5: a chunk citation next to fact sources is unchanged (no fact_id) ---
    result5, _ = _run(chunks, dict(envelope, cited_sources=[1, 2], answer="A [Fuente 1] B [Fuente 2]."))
    if "fact_id" in result5.citations[0] or result5.citations[0]["chunk_id"] != 3558:
        failures.append(f"H5 FAILED: chunk citation changed: {result5.citations[0]}")
    print("H5 chunk citations are unchanged:", "FAIL" if any("H5" in f for f in failures) else "OK")

    if failures:
        print("\n".join(failures))
        return 1
    print("all synthesis fact-set contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
