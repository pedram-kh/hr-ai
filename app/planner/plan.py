"""The `/plan` call (plan.md §C.7 / §C.10).

Native Anthropic tool use, `tool_choice: any`, thinking disabled, NO
`temperature` parameter (the step-0 probe: claude-sonnet-5 rejects
`temperature` under forced tool choice). Repeatability rests on those
constraints plus a fixed prompt/tool order — not on a sampling knob.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .tools import SYSTEM_PROMPT, TOOLS, TOOLS_BY_NAME


def prompt_version(tools: list[dict]) -> str:
    """sha256 of system prompt + the tools actually sent, in list order (§C.8)."""
    payload = json.dumps(
        {"system": SYSTEM_PROMPT, "tools": tools},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_tools(enabled: list[str]) -> list[dict]:
    """Preserve TOOLS' fixed spec order; drop names this build does not know."""
    wanted = {name for name in enabled if name in TOOLS_BY_NAME}
    return [t for t in TOOLS if t["name"] in wanted]


def compact_sorted(value: Any) -> str:
    """§C.10 — tool results / summaries as compact sorted JSON."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_user_content(
    question: str,
    scope_summary: dict,
    window: dict,
    prior_steps: list,
) -> str:
    # Scope keys sorted so two identical summaries hash the same (§C.10).
    scope = compact_sorted(scope_summary)
    window_json = compact_sorted(window)
    parts = [
        f"Pregunta: {question}",
        f"Alcance: {scope}",
        f"Ventana: {window_json}",
    ]
    if prior_steps:
        parts.append(f"Pasos previos de este turno: {compact_sorted(prior_steps)}")
    return "\n\n".join(parts)


def normalize_tool_calls(raw_calls: list[dict], allowed_names: set[str]) -> list[dict]:
    """Keep only well-shaped calls whose tool is in the enabled set.

    Unknown tools are DROPPED (not an error) — §E.15 step 6. A missing id
    is synthesized so hr-backend still has a stable call_id for the next
    round's tool_result reconstruction.
    """
    out: list[dict] = []
    for i, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            continue
        name = raw.get("tool") or raw.get("name")
        if not isinstance(name, str) or name not in allowed_names:
            continue
        raw_input = raw.get("input")
        if raw_input is None:
            raw_input = {}
        if not isinstance(raw_input, dict):
            continue
        call_id = raw.get("id")
        if not isinstance(call_id, str) or call_id == "":
            call_id = f"call_{i + 1}"
        out.append({"id": call_id, "tool": name, "input": raw_input})
    return out


def _calls_from_anthropic_content(content: Any, allowed_names: set[str]) -> list[dict]:
    raw: list[dict] = []
    for block in content or []:
        block_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if block_type != "tool_use":
            continue
        if isinstance(block, dict):
            raw.append(
                {
                    "id": block.get("id"),
                    "tool": block.get("name"),
                    "input": block.get("input") or {},
                }
            )
        else:
            raw.append(
                {
                    "id": getattr(block, "id", None),
                    "tool": getattr(block, "name", None),
                    "input": getattr(block, "input", None) or {},
                }
            )
    return normalize_tool_calls(raw, allowed_names)


def plan(
    question: str,
    scope_summary: dict,
    window: dict,
    enabled_tools: list[str],
    prior_steps: list,
    api_key: str,
    provider_config: dict,
    client: Any | None = None,
) -> dict:
    """One `/plan` round. `client` is injectable for the contract test."""
    tools = select_tools(enabled_tools)
    if not tools:
        raise ValueError("no enabled tools remain after the allowlist filter")

    allowed = {t["name"] for t in tools}
    user_content = build_user_content(question, scope_summary, window, prior_steps)

    if client is None:
        import anthropic

        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=provider_config.get("endpoint") or None,
        )

    started = time.monotonic()
    # Thinking disabled + tool_choice any + NO temperature (§C.10).
    resp = client.messages.create(
        model=provider_config["model"],
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        tools=tools,
        tool_choice={"type": "any"},
        thinking={"type": "disabled"},
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)

    calls = _calls_from_anthropic_content(getattr(resp, "content", None), allowed)
    usage = getattr(resp, "usage", None)
    request_id = None
    if getattr(resp, "_request_id", None):
        request_id = resp._request_id
    elif getattr(resp, "id", None):
        request_id = resp.id

    return {
        "stop_reason": getattr(resp, "stop_reason", None) or "tool_use",
        "calls": calls,
        "model": getattr(resp, "model", None) or provider_config.get("model"),
        "request_id": request_id,
        "prompt_version": prompt_version(tools),
        "tokens": {
            "prompt": getattr(usage, "input_tokens", None) if usage else None,
            "completion": getattr(usage, "output_tokens", None) if usage else None,
        },
        "ms": elapsed_ms,
        "thinking": False,
        "tool_choice": "any",
    }
