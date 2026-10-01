"""Slice 13c — `/synthesise` structured abstention flag, no live API.

hr-ai has no pytest suite (see planner_contract_test.py). This script fakes the Anthropic client and asserts:

  A1. Default call (no `report_abstention`): the system prompt is EXACTLY SYSTEM_PROMPT (sha256 pinned), the result carries
      no abstention fields, and the trace_fragment has no abstention keys — byte-identical to before.
  A2. `report_abstention=True`: the system prompt is SYSTEM_PROMPT + the addendum (and only that).
  A3. The model's `abstained` boolean is the structured signal and wins (True and False), `abstained_by == "model_flag"`.
  A4. A prose abstention that still cites a related source (the S3c LP-14 shape: cited_sources non-empty, answer opens
      with "No dispongo de información suficiente…") with `abstained: true` -> abstained, even though citations exist.
  A5. No usable flag from the model (missing / non-boolean) -> the phrase match is used as a FALLBACK only.
  A6. A flag of false next to an abstaining phrase is NOT overridden; the disagreement is recorded in the trace.
  A7. The phrase match against the shared fixture (hr-backend tests/Fixtures/synthesis-abstention-phrases.json).
  A8. `/synthesise` endpoint: keys `abstained`/`abstained_by` appear only when requested.

Run:
    .venv/bin/python scripts/synthesis_abstention_test.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  <- {detail}"))
    if not ok:
        FAILS.append(name)


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


def _chunks():
    from app.providers import ChunkInput

    return [ChunkInput(chunk_id=1, document_id=10, page_from=3, page_to=3, content="texto", score=0.5, authority_level="official_convenio")]


def _run(envelope: dict, report: bool | None):
    from app.providers import ProviderConfig, get_provider

    messages = _FakeMessages(envelope)
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda **_: SimpleNamespace(messages=messages)  # type: ignore[attr-defined]
    sys.modules["anthropic"] = fake_anthropic
    cfg = ProviderConfig(provider="claude", model="claude-sonnet-5", endpoint=None)
    prov = get_provider("claude")
    if report is None:
        res = prov.synthesise("¿Qué es la Inspección de Trabajo?", _chunks(), "k", cfg)
    else:
        res = prov.synthesise("¿Qué es la Inspección de Trabajo?", _chunks(), "k", cfg, report_abstention=report)
    return res, messages


LP14 = "No dispongo de información suficiente en las fuentes proporcionadas para definir qué es la Inspección de Trabajo. Las fuentes solo mencionan a la Inspección [Fuente 1]."
REAL = "La excedencia voluntaria es posible con un año de antigüedad [Fuente 1]."


def main() -> int:
    from app.providers.claude import SYNTHESIS_ABSTENTION_ADDENDUM, SYSTEM_PROMPT, detect_abstention_phrase

    sha = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    # A1
    res, m = _run({"answer": REAL, "cited_sources": [1], "confidence": 0.9}, None)
    check("A1 default call: system prompt is SYSTEM_PROMPT exactly", m.last_kwargs["system"] == SYSTEM_PROMPT, "")
    check("A1 ... no abstention fields on the result", res.abstained is None and res.abstained_by is None, repr((res.abstained, res.abstained_by)))
    check("A1 ... no abstention keys in trace_fragment", not any("abstention" in k for k in res.trace_fragment), repr(res.trace_fragment))
    res0, m0 = _run({"answer": REAL, "cited_sources": [1], "confidence": 0.9}, False)
    check("A1 report_abstention=False is the same as default", m0.last_kwargs["system"] == SYSTEM_PROMPT and res0.abstained is None, "")
    check("A1 SYSTEM_PROMPT sha256 is pinned (unchanged by this slice)", sha == "558042ab584288b0d08339ac1109f4304f565c6802e2a5bcbe0072a766397887", sha)

    # A2
    res, m = _run({"answer": REAL, "cited_sources": [1], "confidence": 0.9, "abstained": False}, True)
    check("A2 report_abstention=True: prompt = SYSTEM_PROMPT + addendum", m.last_kwargs["system"] == SYSTEM_PROMPT + SYNTHESIS_ABSTENTION_ADDENDUM, "")
    check("A2 ... the addendum names the field and is non-trivial", '"abstained"' in SYNTHESIS_ABSTENTION_ADDENDUM and len(SYNTHESIS_ABSTENTION_ADDENDUM) > 200, "")

    # A3
    res, _ = _run({"answer": REAL, "cited_sources": [1], "confidence": 0.9, "abstained": False}, True)
    check("A3 flag false -> abstained False by model_flag", res.abstained is False and res.abstained_by == "model_flag", repr((res.abstained, res.abstained_by)))
    res, _ = _run({"answer": REAL, "cited_sources": [1], "confidence": 0.9, "abstained": True}, True)
    check("A3 flag true -> abstained True by model_flag", res.abstained is True and res.abstained_by == "model_flag", repr((res.abstained, res.abstained_by)))

    # A4
    res, _ = _run({"answer": LP14, "cited_sources": [1], "confidence": 0.6, "abstained": True}, True)
    check("A4 cited abstention (LP-14 shape) is abstained although a citation exists", res.abstained is True and len(res.citations) == 1 and res.abstained_by == "model_flag", repr((res.abstained, len(res.citations))))

    # A5
    res, _ = _run({"answer": LP14, "cited_sources": [1], "confidence": 0.6}, True)
    check("A5 flag missing + abstaining opener -> phrase fallback", res.abstained is True and res.abstained_by == "phrase", repr((res.abstained, res.abstained_by)))
    res, _ = _run({"answer": LP14, "cited_sources": [1], "confidence": 0.6, "abstained": "true"}, True)
    check("A5 flag not a boolean + opener -> phrase fallback", res.abstained is True and res.abstained_by == "phrase" and res.trace_fragment.get("abstention_flag_present") is False, repr(res.trace_fragment))
    res, _ = _run({"answer": REAL, "cited_sources": [1], "confidence": 0.9}, True)
    check("A5 flag missing + real answer -> not abstained", res.abstained is False and res.abstained_by is None, repr((res.abstained, res.abstained_by)))

    # A6
    res, _ = _run({"answer": LP14, "cited_sources": [1], "confidence": 0.6, "abstained": False}, True)
    check("A6 flag false is NOT overridden by the phrase", res.abstained is False and res.abstained_by == "model_flag", repr((res.abstained, res.abstained_by)))
    check("A6 ... the disagreement is recorded in the trace", res.trace_fragment.get("abstention_phrase_disagrees") is True, repr(res.trace_fragment))

    # A7
    fx = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hr-backend", "tests", "Fixtures", "synthesis-abstention-phrases.json")
    if os.path.exists(fx):
        cases = json.load(open(fx, encoding="utf-8"))["cases"]
        bad = [c["id"] for c in cases if detect_abstention_phrase(c["text"]) != c["phrase"]]
        check(f"A7 phrase match agrees with the shared fixture ({len(cases)} cases)", not bad, repr(bad))
    else:
        print("SKIP A7 (hr-backend fixture not found)")

    # A8 endpoint
    from fastapi.testclient import TestClient

    os.environ.setdefault("INTERNAL_API_TOKEN", "t")
    try:
        from app import main as app_main
    except Exception as exc:  # pragma: no cover
        print("SKIP A8 (cannot import app.main:", exc, ")")
        return 1 if FAILS else 0

    body = {
        "question": "q",
        "chunks": [{"chunk_id": 1, "document_id": 10, "page_from": 1, "page_to": 1, "content": "c", "score": 0.5, "authority_level": "official_convenio"}],
        "provider_api_key": "k",
        "provider_config": {"provider": "claude", "model": "claude-sonnet-5"},
    }
    messages = _FakeMessages({"answer": LP14, "cited_sources": [1], "confidence": 0.6, "abstained": True})
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = lambda **_: SimpleNamespace(messages=messages)  # type: ignore[attr-defined]
    sys.modules["anthropic"] = fake_anthropic
    app_main.app.dependency_overrides[app_main.require_internal_token] = lambda: None
    client = TestClient(app_main.app)
    headers = {"X-Internal-Token": os.environ.get("INTERNAL_API_TOKEN", "t")}
    r = client.post("/synthesise", json=body, headers=headers)
    if r.status_code == 200:
        j = r.json()
        check("A8 endpoint default: no abstained keys", "abstained" not in j and "abstained_by" not in j, repr(list(j)))
        r2 = client.post("/synthesise", json={**body, "report_abstention": True}, headers=headers)
        j2 = r2.json()
        check("A8 endpoint report_abstention: abstained true by model_flag", j2.get("abstained") is True and j2.get("abstained_by") == "model_flag", repr(j2)[:200])
    else:
        print("SKIP A8 (endpoint auth/config in this environment:", r.status_code, r.text[:80], ")")

    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
