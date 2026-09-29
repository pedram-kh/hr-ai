"""Sprint 13, build step 6 (plan.md §C.7–C.10) — the `/plan` planner.

hr-ai is the stateless transport (ADR-0007/0015): it calls Anthropic native
tool use and returns the tool calls. hr-backend decides and writes.
"""

from .plan import normalize_tool_calls, plan, prompt_version
from .tools import SYSTEM_PROMPT, TOOLS, tools_by_name

__all__ = [
    "SYSTEM_PROMPT",
    "TOOLS",
    "normalize_tool_calls",
    "plan",
    "prompt_version",
    "tools_by_name",
]
