"""The answer-provider interface and its plain data shapes (ADR-0015).

These dataclasses are deliberately framework-free (not pydantic): the provider
layer is a thin port that knows nothing about FastAPI or hr-backend. `app/main.py`
validates the HTTP body with pydantic and maps it onto these.

`api_key` is a per-call argument — never an instance field — so a key can never
outlive the single synthesis call it was handed for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkInput:
    """One eligible retrieved chunk passed to synthesis.

    `authority_level` is load-bearing for the precedence rule: the employee's
    convenio governs the topics it addresses; `national_law` (the Estatuto) is
    the baseline that applies only where the convenio is silent. hr-backend
    orders convenio chunks before national_law chunks before handing them here.
    """

    chunk_id: int
    document_id: int
    page_from: int | None
    page_to: int | None
    content: str
    score: float
    authority_level: str | None


@dataclass
class ProviderConfig:
    """Non-secret provider settings (which model, which EU endpoint). Comes from
    hr-ai's own config — NOT from hr-backend, and NOT a secret."""

    provider: str
    model: str
    endpoint: str | None = None


@dataclass
class SynthesisResult:
    """What a provider returns for one synthesis call.

    `authority_used` is computed deterministically from the authority levels of
    the cited chunks (not trusted from the model) so an auditor can see whether an
    answer was drawn from the convenio or from the national-law baseline.
    """

    answer: str
    citations: list[dict] = field(default_factory=list)
    grounding_signal: dict = field(default_factory=dict)
    confidence: float = 0.0
    authority_used: list[str] = field(default_factory=list)
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class GroundChunk:
    """One CITED chunk passed to the per-claim grounding check (Sprint 2b-2, §5).

    `is_tabular` lets the grounding prompt be table-aware: digit-presence in a
    tabular/columnar chunk is NOT entailment of a claim (Q5's lesson) — the
    salary path is SQL, but a prose answer that happens to cite a wage-table
    chunk must not be ruled "grounded" just because the digits appear.
    """

    chunk_id: int
    content: str
    authority_level: str | None = None
    is_tabular: bool = False


@dataclass
class RouterResult:
    """The router's classification of ONE question (Sprint 2b-2, ADR-0016).

    `label` ∈ salary | prose | off_domain. `subqueries` is non-empty only for a
    compound question (used by hr-backend's recall hardening: one /retrieve per
    sub-query, unioned — §6). The router sees the QUESTION only, never the chunks.
    """

    label: str
    confidence: float = 0.0
    subqueries: list[str] = field(default_factory=list)
    reason: str = ""
    trace_fragment: dict = field(default_factory=dict)


@dataclass
class GroundingResult:
    """The per-claim entailment verdict for one prose answer (Sprint 2b-2, §5).

    `grounded` is the gate signal: True only when EVERY load-bearing claim is
    entailed by a cited chunk. `claims` records each claim + its verdict +
    the supporting chunk (for the audit trail / the trace). hr-backend escalates
    the whole turn (low_confidence) when `grounded` is False — never edits.
    """

    grounded: bool
    claims: list[dict] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    trace_fragment: dict = field(default_factory=dict)


class AnswerProvider(ABC):
    """A pluggable answer provider (synthesis + routing + grounding). Default: Claude.

    All three calls reuse the SAME hr-backend-owned key path (ADR-0015): the key
    is a per-call argument, never an instance field, never persisted. The MODEL
    differs by call (ADR-0016 / Sprint 2b-2 §5): `route` uses the small/fast
    ROUTER_MODEL; `synthesise` and `ground` use the capable ANSWER_MODEL
    (entailment is subtle — never judged by the cheap classifier).
    """

    @abstractmethod
    def synthesise(
        self,
        question: str,
        chunks: list[ChunkInput],
        api_key: str,
        config: ProviderConfig,
    ) -> SynthesisResult:
        """Compose a cited answer grounded ONLY in `chunks`, honouring the
        convenio-over-baseline precedence rule. `api_key` is used for this one
        call and never persisted."""
        raise NotImplementedError

    @abstractmethod
    def classify(
        self,
        question: str,
        api_key: str,
        config: ProviderConfig,
    ) -> RouterResult:
        """Classify the question salary | prose | off_domain and, for a compound
        question, decompose it into `subqueries` (ADR-0016). Small/fast model."""
        raise NotImplementedError

    @abstractmethod
    def ground(
        self,
        question: str,
        answer: str,
        chunks: list[GroundChunk],
        api_key: str,
        config: ProviderConfig,
    ) -> GroundingResult:
        """Per-claim entailment of `answer` against the CITED `chunks`. The real
        grounding gate (Sprint 2b-2 §5). Table-aware. Capable answer model."""
        raise NotImplementedError
