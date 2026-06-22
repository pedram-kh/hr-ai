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


class AnswerProvider(ABC):
    """A pluggable answer-synthesis provider. Default impl: Claude."""

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
