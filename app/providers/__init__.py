"""Pluggable answer-synthesis providers (ADR-0015).

The answer model is external (quality-dominant trade, unlike the self-hosted
embedding model — ADR-0006). It sits behind one interface so swapping providers,
or falling back to a self-hosted answer model if compliance ever demands, is an
adapter/config change, not a rewrite. Default provider: Claude.

The API key is NEVER owned here — hr-backend owns it (encrypted at rest), decrypts
it, and passes it per call. A provider adapter receives the key as a call argument,
uses it for that one request, and never stores, logs, or persists it.
"""

from .base import (
    AnswerProvider,
    ChunkInput,
    ConvenioCandidate,
    ExplainResult,
    GroundChunk,
    GroundingResult,
    GroupProposalResult,
    JobCategoryCandidate,
    OcrPageResult,
    ProviderConfig,
    RouterResult,
    SegmentedFactsResult,
    SynthesisResult,
    TagProposalResult,
    VocabularyCandidate,
)
from .claude import ClaudeProvider

__all__ = [
    "AnswerProvider",
    "ChunkInput",
    "ConvenioCandidate",
    "ExplainResult",
    "GroundChunk",
    "GroundingResult",
    "GroupProposalResult",
    "JobCategoryCandidate",
    "OcrPageResult",
    "ProviderConfig",
    "RouterResult",
    "SegmentedFactsResult",
    "SynthesisResult",
    "TagProposalResult",
    "VocabularyCandidate",
    "ClaudeProvider",
    "get_provider",
]


def get_provider(name: str) -> AnswerProvider:
    """Resolve a provider by its non-secret config name (default: claude)."""
    providers: dict[str, type[AnswerProvider]] = {
        "claude": ClaudeProvider,
    }
    impl = providers.get(name.lower())
    if impl is None:
        raise ValueError(f"Unknown answer provider '{name}' (known: {list(providers)})")
    return impl()
