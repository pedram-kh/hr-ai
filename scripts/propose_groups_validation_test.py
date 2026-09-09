#!/usr/bin/env python3
"""Verify the /propose-groups response validation (Sprint 7f, ADR-0028).

hr-ai has no pytest suite (this repo's established pattern — see sanity_test.py,
ocr_sidecar_test.py — is a standalone, directly-run verification script), so this
follows the same convention rather than introducing one for these invariants.

What this pins is the gap between what the prompt ASKS for and what the code
ENFORCES. The prompt asks the model not to invent categories, not to split a
group without a citation, and not to emit an orphan area; this script proves that
a model which does all three anyway cannot get any of it past the provider — the
payload degrades to fewer nodes, never to a wrong tree and never to a minted
category. The Anthropic call is stubbed, so this makes no network request and
costs nothing.

Run:  python3 scripts/propose_groups_validation_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers import ClaudeProvider, ConvenioCandidate, JobCategoryCandidate, ProviderConfig

CONFIG = ProviderConfig(provider="claude", model="claude-sonnet-4-5", endpoint=None)

# Two real categories; id 999 is NOT in the set and must never survive.
CONVENIO = ConvenioCandidate(
    id=21,
    name="HOSTELERIA",
    numero="31003805011981",
    territory_name="Navarra",
    sector_name="Hostelería",
    job_categories=[
        JobCategoryCandidate(id=1, name="Camarero", group_code="Grupo 2"),
        JobCategoryCandidate(id=2, name="Jefe de cocina", group_code="1.234,56"),
    ],
)


class _FakeUsage:
    input_tokens = 1000
    output_tokens = 500


class _FakeBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage()


def run(model_output: str) -> tuple[list[dict], dict]:
    """Call propose_groups with the Anthropic client stubbed to return
    `model_output` verbatim — no network, no key, no cost."""

    class _FakeMessages:
        def create(self, **_kwargs: object) -> _FakeMessage:
            return _FakeMessage(model_output)

    class _FakeClient:
        messages = _FakeMessages()

    class _FakeAnthropicModule:
        @staticmethod
        def Anthropic(**_kwargs: object) -> _FakeClient:  # noqa: N802 - mirrors the SDK name
            return _FakeClient()

    with patch.dict(sys.modules, {"anthropic": _FakeAnthropicModule}):
        result = ClaudeProvider().propose_groups(
            CONVENIO, "texto del convenio", ["Grupo 1", "Grupo 2"], "unused-key", CONFIG
        )
    return result.groups, result.trace_fragment


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))
        FAILURES.append(name)


def node(groups: list[dict], label: str, parent: str | None = None) -> dict | None:
    for g in groups:
        if g["code_label"] == label and g["parent_code_label"] == parent:
            return g
    return None


print("\n[1] The happy path — Hostelería Navarra's real shape (G2 split by value)")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {
                    "code_label": "Grupo 1",
                    "parent_code_label": None,
                    "job_category_ids": [2],
                    "source_excerpt": "Grupo 1: ... 6 meses",
                    "source_locator": "p.12",
                    "confidence": 0.9,
                    "uncertainty": None,
                },
                {
                    "code_label": "Grupo 2",
                    "parent_code_label": None,
                    "job_category_ids": [1],
                    "source_excerpt": "Grupo 2: ...",
                    "source_locator": "p.12",
                    "confidence": 0.9,
                    "uncertainty": None,
                },
                {
                    "code_label": "área 5",
                    "parent_code_label": "Grupo 2",
                    "job_category_ids": [],
                    "source_excerpt": "área 5: 90 días",
                    "source_locator": "p.12",
                    "confidence": 0.85,
                    "uncertainty": None,
                },
                {
                    "code_label": "resto áreas",
                    "parent_code_label": "Grupo 2",
                    "job_category_ids": [],
                    "source_excerpt": "resto de áreas: 60 días",
                    "source_locator": "p.12",
                    "confidence": 0.85,
                    "uncertainty": None,
                },
            ],
            "notes": "",
        }
    )
)
check("4 nodes survive", len(groups) == 4, f"got {len(groups)}")
check("2 roots, 2 sub-areas", trace["group_count"] == 2 and trace["sub_area_count"] == 2, str(trace))
check("labels are passed through UNNORMALIZED", node(groups, "área 5", "Grupo 2") is not None)
check("valid category id kept", (node(groups, "Grupo 2") or {}).get("job_category_ids") == [1])
check("cost is reported", trace["cost_usd"] > 0, str(trace.get("cost_usd")))

print("\n[2] A hallucinated category id can never reach hr-backend (ADR-0011)")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {
                    "code_label": "Grupo 1",
                    "parent_code_label": None,
                    "job_category_ids": [1, 999, 12345],
                    "source_excerpt": "Grupo 1",
                    "source_locator": "p.1",
                }
            ]
        }
    )
)
check("only the real id survives", (node(groups, "Grupo 1") or {}).get("job_category_ids") == [1])
check("the drop is counted for the trace", trace["dropped_category_ids"] == 2, str(trace))

print("\n[3] An UNCITED split is dropped — the granularity guard")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {
                    "code_label": "Grupo 2",
                    "parent_code_label": None,
                    "job_category_ids": [],
                    "source_excerpt": "Grupo 2",
                    "source_locator": "p.1",
                },
                {
                    "code_label": "área 3",
                    "parent_code_label": "Grupo 2",
                    "job_category_ids": [],
                    "source_excerpt": "",
                    "source_locator": "p.1",
                },
            ]
        }
    )
)
check("the group survives", node(groups, "Grupo 2") is not None)
check("the uncited area does not", node(groups, "área 3", "Grupo 2") is None)
check("counted as unsupported", trace["dropped_unsupported_areas"] == 1, str(trace))

# Rewritten after the first live run on Hostelería Navarra, which is where this
# behaviour was decided rather than guessed. The model read article 19 correctly
# and said so in its notes, but returned Grupo 2's two areas WITHOUT Grupo 2 —
# so both areas were orphans, both were dropped, and the result was exactly the
# under-split the eval gates on. A cited orphan now gets its parent
# reconstructed: the label comes verbatim from the child's `parent_code_label`,
# so nothing is invented, and the node is flagged so the reviewer knows it came
# from its children rather than from a line of its own.
print("\n[4] A CITED orphan area reconstructs its parent instead of losing the split")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {
                    "code_label": "Grupo 1",
                    "parent_code_label": None,
                    "job_category_ids": [],
                    "source_excerpt": "Grupo 1",
                    "source_locator": "p.1",
                },
                {
                    "code_label": "área 5",
                    "parent_code_label": "Grupo 7",  # never proposed
                    "job_category_ids": [],
                    "source_excerpt": "área 5: 90 días",
                    "source_locator": "p.1",
                },
            ]
        }
    )
)
check("the split survives", len(groups) == 3, str([g["code_label"] for g in groups]))
check("the missing parent was reconstructed", node(groups, "Grupo 7") is not None)
check(
    "reconstructed with the label its child cited, not an invented one",
    (node(groups, "Grupo 7") or {}).get("parent_code_label") is None,
)
check("the area stays an AREA under it, not a second root", node(groups, "área 5", "Grupo 7") is not None)
check("and it is NOT also emitted as a root", node(groups, "área 5") is None)
check(
    "the reconstruction is flagged for the reviewer",
    "reconstruido" in ((node(groups, "Grupo 7") or {}).get("uncertainty") or {}).get("reason", ""),
)
check("counted", trace["synthesised_parents"] == 1, str(trace))

print("\n[4b] An UNCITED orphan area is still dropped, and conjures nothing")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {
                    "code_label": "Grupo 1",
                    "parent_code_label": None,
                    "job_category_ids": [],
                    "source_excerpt": "Grupo 1",
                    "source_locator": "p.1",
                },
                {
                    "code_label": "área 5",
                    "parent_code_label": "Grupo 7",  # never proposed, and uncited
                    "job_category_ids": [],
                    "source_excerpt": "",
                },
            ]
        }
    )
)
check("only the real group remains", len(groups) == 1 and groups[0]["code_label"] == "Grupo 1")
check("no group was conjured for it", node(groups, "Grupo 7") is None)
check("nothing was synthesised", trace["synthesised_parents"] == 0, str(trace))

print("\n[5] Depth stays at 2 — an area under an area cannot exist")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {
                    "code_label": "Grupo 2",
                    "parent_code_label": None,
                    "job_category_ids": [],
                    "source_excerpt": "Grupo 2",
                    "source_locator": "p.1",
                },
                {
                    "code_label": "área 5",
                    "parent_code_label": "Grupo 2",
                    "job_category_ids": [],
                    "source_excerpt": "área 5: 90 días",
                    "source_locator": "p.1",
                },
                {
                    "code_label": "subárea 5.1",
                    "parent_code_label": "área 5",  # a grandchild
                    "job_category_ids": [],
                    "source_excerpt": "subárea 5.1: 30 días",
                    "source_locator": "p.1",
                },
            ]
        }
    )
)
check("the grandchild is dropped", node(groups, "subárea 5.1", "área 5") is None)
check("group + area survive", len(groups) == 2, f"got {len(groups)}")

print("\n[6] A root with no citation survives but is FLAGGED, never silently confident")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {
                    "code_label": "Técnicos titulados",
                    "parent_code_label": None,
                    "job_category_ids": [],
                    "source_excerpt": "",
                    "source_locator": "",
                }
            ]
        }
    )
)
n = node(groups, "Técnicos titulados")
check("an unnumbered group is a valid group", n is not None)
check("it carries an uncertainty flag", bool(n and n["uncertainty"]), str(n))
check("counted", trace["flagged_missing_excerpt"] == 1, str(trace))

print("\n[7] Malformed output yields NO nodes and a visible parse_error")
groups, trace = run("lo siento, no puedo determinar la estructura de este convenio")
check("no nodes", groups == [])
check("parse_error surfaced for retry", trace.get("parse_error") is True, str(trace))
check("cost still reported", trace["cost_usd"] > 0)

print("\n[8] Duplicate and empty labels collapse")
groups, trace = run(
    json.dumps(
        {
            "groups": [
                {"code_label": "Grupo 1", "parent_code_label": None, "source_excerpt": "a", "job_category_ids": []},
                {"code_label": "GRUPO 1", "parent_code_label": None, "source_excerpt": "b", "job_category_ids": []},
                {"code_label": "  ", "parent_code_label": None, "source_excerpt": "c", "job_category_ids": []},
                "not a dict",
            ]
        }
    )
)
check("one node only", len(groups) == 1, f"got {len(groups)}")

print()
if FAILURES:
    print(f"FAILED — {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("All /propose-groups validation checks passed.")
