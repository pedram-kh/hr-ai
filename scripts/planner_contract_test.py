"""Sprint 13, build step 6 (plan.md §E.15) — `/plan` contract, no live API.

hr-ai has no pytest suite (see ocr_sidecar_test.py / sanity_test.py). This
script mocks the Anthropic client and asserts:

  1. Response normalization: tool_use blocks become {id, tool, input}.
  2. An unknown tool the model invents is DROPPED, not forwarded.
  3. The Anthropic call never sends a `temperature` parameter (§C.10).
  4. `tool_choice` is `any` and thinking is disabled.
  5. `prompt_version` is a stable sha256 of system + the enabled tools.

Run:
    python3 scripts/planner_contract_test.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeToolUse:
    def __init__(self, name: str, tool_id: str, input_data: dict):
        self.type = "tool_use"
        self.name = name
        self.id = tool_id
        self.input = input_data


class _CapturingClient:
    def __init__(self, content):
        self.content = content
        self.last_kwargs: dict | None = None
        self.messages = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            content=self.content,
            stop_reason="tool_use",
            model=kwargs.get("model"),
            id="msg_test",
            usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        )


def main() -> int:
    from app.planner.plan import normalize_tool_calls, plan, prompt_version, select_tools
    from app.planner.tools import TOOLS

    failures: list[str] = []

    # --- (1) normalize: well-shaped calls pass through, junk is dropped ---
    allowed = {"salary_lookup", "finalize"}
    normalized = normalize_tool_calls(
        [
            {"id": "t1", "tool": "salary_lookup", "input": {}},
            {"id": "t2", "name": "finalize", "input": {"use": ["t1"]}},
            {"id": "t3", "tool": "invented_tool", "input": {}},  # unknown — drop
            {"tool": "salary_lookup", "input": "not-an-object"},  # bad input — drop
            "not-a-dict",
        ],
        allowed,
    )
    if [c["tool"] for c in normalized] != ["salary_lookup", "finalize"]:
        failures.append(f"(1) FAILED: expected salary_lookup+finalize, got {normalized}")
    if normalized[0]["id"] != "t1" or normalized[1]["input"] != {"use": ["t1"]}:
        failures.append(f"(1) FAILED: id/input not preserved: {normalized}")
    print("(1) normalize keeps known well-shaped calls:", "FAIL" if any("(1)" in f for f in failures) else "OK")

    # --- (2) unknown tool in a live-shaped Anthropic response is dropped ---
    client = _CapturingClient(
        [
            _FakeToolUse("convenio_search", "tu_1", {}),
            _FakeToolUse("not_a_real_tool", "tu_2", {"foo": 1}),
        ]
    )
    result = plan(
        question="¿cuántas vacaciones tengo?",
        scope_summary={"convenio_name": "Hostelería Navarra"},
        window={"exchanges": [], "message_ids": []},
        enabled_tools=["convenio_search", "finalize"],
        prior_steps=[],
        api_key="test-key",
        provider_config={"provider": "claude", "model": "claude-sonnet-5", "endpoint": None},
        client=client,
    )
    tools_returned = [c["tool"] for c in result["calls"]]
    if tools_returned != ["convenio_search"]:
        failures.append(f"(2) FAILED: unknown tool leaked through: {result['calls']}")
    if result["prompt_version"] != prompt_version(select_tools(["convenio_search", "finalize"])):
        failures.append("(2) FAILED: prompt_version did not match the enabled-tool subset")
    print("(2) unknown tool dropped from Anthropic content:", "FAIL" if any("(2)" in f for f in failures) else "OK")

    # --- (3)+(4) the Anthropic kwargs honour §C.10 ---
    kwargs = client.last_kwargs or {}
    if "temperature" in kwargs:
        failures.append("(3) FAILED: temperature was sent — §C.10 forbids the parameter entirely")
    if kwargs.get("tool_choice") != {"type": "any"}:
        failures.append(f"(4) FAILED: tool_choice={kwargs.get('tool_choice')!r}, expected any")
    if kwargs.get("thinking") != {"type": "disabled"}:
        failures.append(f"(4) FAILED: thinking={kwargs.get('thinking')!r}, expected disabled")
    sent_names = [t["name"] for t in kwargs.get("tools", [])]
    if "general_knowledge" in sent_names:
        failures.append("(4) FAILED: general_knowledge was sent even though it was not enabled")
    if sent_names != ["convenio_search", "finalize"]:
        failures.append(f"(4) FAILED: tools not in spec-order subset: {sent_names}")
    print("(3) no temperature parameter:", "FAIL" if any("(3)" in f for f in failures) else "OK")
    print("(4) tool_choice any + thinking disabled + enabled subset:", "FAIL" if any("(4)" in f for f in failures) else "OK")

    # --- (5) prompt_version is stable for the same tool set ---
    v1 = prompt_version(select_tools(["salary_lookup", "finalize"]))
    v2 = prompt_version(select_tools(["salary_lookup", "finalize"]))
    v3 = prompt_version(select_tools(["finalize", "salary_lookup"]))  # select_tools reorders to spec order
    if not v1.startswith("sha256:") or v1 != v2 or v1 != v3:
        failures.append(f"(5) FAILED: prompt_version unstable: {v1} / {v2} / {v3}")
    if prompt_version(TOOLS) == v1:
        failures.append("(5) FAILED: full TOOLS hash collided with a 2-tool subset")
    print("(5) prompt_version stable + order-independent:", "FAIL" if any("(5)" in f for f in failures) else "OK")

    if failures:
        for f in failures:
            print(f)
        return 1
    print("all planner contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
